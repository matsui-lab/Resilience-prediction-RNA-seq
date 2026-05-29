# Script cleaned for public release. Edit /path/to/... inputs before running.
rm(list = ls())
options(stringsAsFactors = FALSE)

path <- "/path/to/project"
setwd(path)

outdir <- "out/clustering/linear_model_clinical_from_merged_combat_both_cohorts_exploratory"
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(outdir, "ROSMAP"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(outdir, "MSBB"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(outdir, "comparison"), recursive = TRUE, showWarnings = FALSE)

expr_merged_combat_file <- "/path/to/project/out/combined/exp_merged_combat_exclude89.csv"
meta_merged_file <- "/path/to/project/out/combined/meta_merged_exclude89.csv"

selected_gene_file <- "out/clustering/positice_gene_annotation.csv"

gene_anno_file <- "/path/to/annotation/genelist_human_2101gene.txt"
somascan_file <- "/path/to/ADNI/ADNI_Cruchaga_lab_CSF_SOMAscan7k_analyte_information_20_06_2023.csv"

rosmap_meta_file <- "/path/to/input_data/resilience/RNAseq_Harmonization_ROSMAP_combined_metadata.csv"
msbb_meta_file <- "/path/to/input_data/resilience/RNAseq_Harmonization_MSBB_combined_metadata.csv"

rosmap_continuous_vars <- c("cts_mmse30_lv", "educ", "age_death_clean", "braaksc", "ceradsc")
rosmap_categorical_vars <- c("dcfdx_lv", "msex")
rosmap_clinical_vars <- c(rosmap_continuous_vars, rosmap_categorical_vars)

rosmap_base_covariates_main <- c("age_death_clean", "msex", "dcfdx_lv")
rosmap_base_covariates_batch_sensitivity <- c("age_death_clean", "msex", "dcfdx_lv", "sequencingBatch")

msbb_continuous_vars <- c("CDR", "age_death_clean", "Braak", "CERAD")
msbb_categorical_vars <- c("msex")
msbb_clinical_vars <- c(msbb_continuous_vars, msbb_categorical_vars)

msbb_base_covariates_main_preferred <- c(
  "age_death_clean",
  "msex", "sex", "Sex", "gender", "Gender",
  "diagnosis", "Diagnosis", "dx", "Dx",
  "sequencingBatch"
)

run_batch_sensitivity_model <- TRUE

comparison_variable_map <- data.frame(
  common_variable = c("cognition", "age_at_death", "braak", "cerad", "sex"),
  ROSMAP = c("cts_mmse30_lv", "age_death_clean", "braaksc", "ceradsc", "msex"),
  MSBB = c("CDR", "age_death_clean", "Braak", "CERAD", "msex"),
  note = c(
    "MMSE and CDR have opposite clinical direction; consider sign reversal for biological interpretation.",
    "Same broad variable.",
    "Same broad neuropathology variable.",
    "Same broad neuropathology variable.",
    "Categorical sex variable; compare p-values, not beta direction."
  ),
  stringsAsFactors = FALSE
)

suppressPackageStartupMessages({
  library(dplyr)
  library(data.table)
  library(tidyr)
  library(tibble)
  library(ggplot2)
  library(pheatmap)
  library(RColorBrewer)
})

clean_numeric <- function(x) {
  x <- as.character(x)
  x[x %in% c("", "NA", "NaN", "NULL", "null")] <- NA
  x <- ifelse(x == "90+", "99", x)
  suppressWarnings(as.numeric(x))
}

p_to_stars <- function(p) {
  if (is.na(p)) return("NA")
  if (p > 0.05) return("ns")
  if (p > 0.01) return("*")
  if (p > 0.001) return("**")
  if (p > 0.0001) return("***")
  return("****")
}

sanitize_colnames <- function(x) {
  x <- sub("^X", "", x)
  x <- gsub("-", ".", x)
  x
}

sanitize_sample_ids <- function(x) {
  x <- as.character(x)
  x <- gsub("-", ".", x)
  x
}

safe_factor <- function(x) {
  as.factor(as.character(x))
}

first_existing_col <- function(df, candidates) {
  hit <- candidates[candidates %in% colnames(df)]
  if (length(hit) == 0) return(NA_character_)
  hit[1]
}

remove_bad_genes <- function(expr_mat) {
  expr_mat <- as.matrix(expr_mat)
  storage.mode(expr_mat) <- "numeric"
  keep <- apply(expr_mat, 1, function(z) {
    all(is.finite(z)) && !any(is.na(z)) && stats::sd(z) > 0
  })
  expr_mat[keep, , drop = FALSE]
}

align_expr_meta <- function(expr_mat, meta_df, sample_col = "specimenID") {
  common <- intersect(colnames(expr_mat), meta_df[[sample_col]])
  if (length(common) == 0) {
    stop("No common samples between expression matrix and metadata after ID normalization.")
  }
  expr_mat <- expr_mat[, common, drop = FALSE]
  meta_df <- meta_df[match(common, meta_df[[sample_col]]), , drop = FALSE]
  stopifnot(identical(colnames(expr_mat), meta_df[[sample_col]]))
  list(expr = expr_mat, meta = meta_df)
}

test_batch_clinical_association <- function(meta_df, batch_col, clinical_vars, outfile) {
  if (!(batch_col %in% colnames(meta_df))) {
    warning("Batch column not found: ", batch_col)
    return(NULL)
  }

  out <- lapply(clinical_vars, function(v) {
    if (!(v %in% colnames(meta_df))) {
      return(data.frame(variable = v, test = NA, p.value = NA, n = NA, note = "variable_not_found"))
    }

    df <- meta_df[, c(batch_col, v), drop = FALSE]
    names(df) <- c("batch", "var")
    df <- df[complete.cases(df), , drop = FALSE]
    df$batch <- as.factor(df$batch)

    if (nrow(df) < 4 || nlevels(df$batch) < 2 || length(unique(df$var)) < 2) {
      return(data.frame(variable = v, test = NA, p.value = NA, n = nrow(df), note = "insufficient_variation"))
    }

    if (is.numeric(df$var)) {
      fit <- tryCatch(summary(aov(var ~ batch, data = df)), error = function(e) NULL)
      p <- if (is.null(fit)) NA_real_ else fit[[1]]$`Pr(>F)`[1]
      data.frame(variable = v, test = "aov_by_batch", p.value = p, n = nrow(df), note = "")
    } else {
      tab <- table(df$var, df$batch)
      test <- tryCatch(chisq.test(tab), error = function(e) NULL)
      p <- if (is.null(test)) NA_real_ else test$p.value
      data.frame(variable = v, test = "chisq_by_batch", p.value = p, n = nrow(df), note = "")
    }
  }) %>% bind_rows()

  out$FDR <- p.adjust(out$p.value, method = "BH")
  write.csv(out, outfile, row.names = FALSE, quote = FALSE)
  out
}

gene_wise_lm <- function(expr_mat, genes_tbl, meta_df, continuous_vars, categorical_vars,
                         base_covariates, sample_col = "specimenID", model_label = "main") {
  expr_mat <- remove_bad_genes(expr_mat)
  clinical_vars <- c(continuous_vars, categorical_vars)
  out <- list()

  for (gene in rownames(expr_mat)) {
    symbol <- genes_tbl$Symbol[match(gene, genes_tbl$gene)]
    if (is.na(symbol) || length(symbol) == 0) symbol <- gene

    df0 <- data.frame(
      specimenID = colnames(expr_mat),
      expr = as.numeric(expr_mat[gene, ]),
      stringsAsFactors = FALSE
    ) %>% left_join(meta_df, by = sample_col)

    for (v in clinical_vars) {
      if (!(v %in% colnames(df0))) next

      covars <- setdiff(base_covariates, v)
      covars <- covars[covars %in% colnames(df0)]
      rhs <- c(v, covars)

      df <- df0[, c("expr", rhs), drop = FALSE]
      df <- df[complete.cases(df), , drop = FALSE]

      if (nrow(df) < 10 || length(unique(df$expr)) < 2 || length(unique(df[[v]])) < 2) {
        out[[length(out) + 1]] <- data.frame(
          model = model_label,
          gene = gene,
          Symbol = symbol,
          variable = v,
          variable_type = ifelse(v %in% categorical_vars, "categorical", "continuous"),
          beta = NA_real_,
          statistic = NA_real_,
          p.value = NA_real_,
          n = nrow(df),
          covariates = paste(covars, collapse = ";"),
          formula = NA_character_,
          test = "insufficient_variation",
          stringsAsFactors = FALSE
        )
        next
      }

      categorical_like_terms <- unique(c(
        categorical_vars,
        "msex", "sex", "Sex", "gender", "Gender",
        "dcfdx_lv", "diagnosis", "Diagnosis", "dx", "Dx",
        "sequencingBatch", "cohort"
      ))
      for (cc in names(df)) {
        if (cc %in% categorical_like_terms) {
          df[[cc]] <- droplevels(as.factor(df[[cc]]))
        }
      }

      usable_rhs <- c()
      for (term in rhs) {
        z <- df[[term]]
        if (is.factor(z) && nlevels(droplevels(z)) < 2) next
        if (!is.factor(z) && length(unique(z)) < 2) next
        usable_rhs <- c(usable_rhs, term)
      }
      if (!(v %in% usable_rhs)) {
        out[[length(out) + 1]] <- data.frame(
          model = model_label,
          gene = gene,
          Symbol = symbol,
          variable = v,
          variable_type = ifelse(v %in% categorical_vars, "categorical", "continuous"),
          beta = NA_real_,
          statistic = NA_real_,
          p.value = NA_real_,
          n = nrow(df),
          covariates = paste(covars, collapse = ";"),
          formula = NA_character_,
          test = "tested_variable_dropped",
          stringsAsFactors = FALSE
        )
        next
      }

      full_form_txt <- paste("expr ~", paste(usable_rhs, collapse = " + "))
      full_form <- as.formula(full_form_txt)
      fit <- tryCatch(lm(full_form, data = df), error = function(e) NULL)

      if (is.null(fit)) {
        out[[length(out) + 1]] <- data.frame(
          model = model_label,
          gene = gene,
          Symbol = symbol,
          variable = v,
          variable_type = ifelse(v %in% categorical_vars, "categorical", "continuous"),
          beta = NA_real_,
          statistic = NA_real_,
          p.value = NA_real_,
          n = nrow(df),
          covariates = paste(setdiff(usable_rhs, v), collapse = ";"),
          formula = full_form_txt,
          test = "lm_failed",
          stringsAsFactors = FALSE
        )
        next
      }

      if (v %in% categorical_vars || is.factor(df[[v]])) {
        reduced_rhs <- setdiff(usable_rhs, v)
        reduced_form <- if (length(reduced_rhs) == 0) {
          as.formula("expr ~ 1")
        } else {
          as.formula(paste("expr ~", paste(reduced_rhs, collapse = " + ")))
        }
        fit0 <- tryCatch(lm(reduced_form, data = df), error = function(e) NULL)
        an <- if (is.null(fit0)) NULL else tryCatch(anova(fit0, fit), error = function(e) NULL)
        p <- if (is.null(an)) NA_real_ else an$`Pr(>F)`[2]
        stat <- if (is.null(an)) NA_real_ else an$F[2]
        beta <- NA_real_
        test_name <- "lm_nested_anova_categorical"
        var_type <- "categorical"
      } else {
        sm <- summary(fit)$coefficients
        if (!(v %in% rownames(sm))) {
          beta <- NA_real_
          stat <- NA_real_
          p <- NA_real_
        } else {
          beta <- sm[v, "Estimate"]
          stat <- sm[v, "t value"]
          p <- sm[v, "Pr(>|t|)"]
        }
        test_name <- "lm_continuous_term"
        var_type <- "continuous"
      }

      out[[length(out) + 1]] <- data.frame(
        model = model_label,
        gene = gene,
        Symbol = symbol,
        variable = v,
        variable_type = var_type,
        beta = beta,
        statistic = stat,
        p.value = p,
        n = nrow(df),
        covariates = paste(setdiff(usable_rhs, v), collapse = ";"),
        formula = full_form_txt,
        test = test_name,
        stringsAsFactors = FALSE
      )
    }
  }

  bind_rows(out) %>%
    group_by(model, variable) %>%
    mutate(FDR = p.adjust(p.value, method = "BH")) %>%
    ungroup()
}

lm_to_matrices <- function(lm_df, genes_tbl, vars) {
  symbols <- genes_tbl$Symbol[genes_tbl$gene %in% lm_df$gene]
  symbols <- unique(symbols)
  symbols <- symbols[!is.na(symbols)]

  beta <- matrix(NA_real_, nrow = length(symbols), ncol = length(vars), dimnames = list(symbols, vars))
  pval <- matrix(NA_real_, nrow = length(symbols), ncol = length(vars), dimnames = list(symbols, vars))
  fdr <- matrix(NA_real_, nrow = length(symbols), ncol = length(vars), dimnames = list(symbols, vars))
  stars <- matrix("", nrow = length(symbols), ncol = length(vars), dimnames = list(symbols, vars))
  neg_log10p <- matrix(NA_real_, nrow = length(symbols), ncol = length(vars), dimnames = list(symbols, vars))
  signed_log10p <- matrix(NA_real_, nrow = length(symbols), ncol = length(vars), dimnames = list(symbols, vars))

  for (i in seq_len(nrow(lm_df))) {
    s <- lm_df$Symbol[i]
    v <- lm_df$variable[i]
    if (!(s %in% rownames(beta)) || !(v %in% colnames(beta))) next

    beta[s, v] <- lm_df$beta[i]
    pval[s, v] <- lm_df$p.value[i]
    fdr[s, v] <- lm_df$FDR[i]
    stars[s, v] <- p_to_stars(lm_df$p.value[i])

    if (!is.na(lm_df$p.value[i])) {
      neg_log10p[s, v] <- -log10(lm_df$p.value[i])
    }
    if (!is.na(lm_df$beta[i]) && !is.na(lm_df$p.value[i])) {
      signed_log10p[s, v] <- sign(lm_df$beta[i]) * (-log10(lm_df$p.value[i]))
    }
  }

  list(beta = beta, pval = pval, fdr = fdr, stars = stars,
       neg_log10p = neg_log10p, signed_log10p = signed_log10p)
}

save_matrix_set <- function(mats, prefix) {
  write.csv(mats$beta, paste0(prefix, "_beta_matrix.csv"), quote = FALSE)
  write.csv(mats$pval, paste0(prefix, "_pval_matrix.csv"), quote = FALSE)
  write.csv(mats$fdr, paste0(prefix, "_FDR_matrix.csv"), quote = FALSE)
  write.csv(mats$stars, paste0(prefix, "_pvalue_stars_matrix.csv"), quote = FALSE)
  write.csv(mats$neg_log10p, paste0(prefix, "_neg_log10p_matrix.csv"), quote = FALSE)
  write.csv(mats$signed_log10p, paste0(prefix, "_signed_log10p_matrix.csv"), quote = FALSE)
}

save_continuous_heatmap_beta <- function(mats, continuous_vars, prefix, title_prefix) {
  vars <- continuous_vars[continuous_vars %in% colnames(mats$beta)]
  if (length(vars) == 0) return(invisible(NULL))

  beta_mat <- mats$beta[, vars, drop = FALSE]
  stars_mat <- mats$stars[, vars, drop = FALSE]

  max_abs <- max(abs(beta_mat), na.rm = TRUE)
  if (!is.finite(max_abs) || max_abs == 0) max_abs <- 1
  brks <- seq(-max_abs, max_abs, length.out = 101)

  svg(paste0(prefix, "_continuous_heatmap_beta_with_raw_pstars.svg"), width = 7, height = 8)
  pheatmap(
    beta_mat,
    display_numbers = stars_mat,
    cluster_rows = TRUE,
    cluster_cols = FALSE,
    main = paste0(title_prefix, ": continuous variables, LM beta; text = raw p stars"),
    fontsize_number = 10,
    color = colorRampPalette(rev(brewer.pal(n = 11, name = "RdBu")))(100),
    breaks = brks
  )
  dev.off()
}

save_categorical_heatmap_pvalue <- function(mats, categorical_vars, prefix, title_prefix) {
  vars <- categorical_vars[categorical_vars %in% colnames(mats$neg_log10p)]
  if (length(vars) == 0) return(invisible(NULL))

  logp_mat <- mats$neg_log10p[, vars, drop = FALSE]
  stars_mat <- mats$stars[, vars, drop = FALSE]
  logp_mat[!is.finite(logp_mat)] <- NA_real_

  svg(paste0(prefix, "_categorical_heatmap_neglog10_raw_p_with_raw_pstars.svg"), width = 5 + length(vars), height = 8)
  pheatmap(
    logp_mat,
    display_numbers = stars_mat,
    cluster_rows = TRUE,
    cluster_cols = FALSE,
    main = paste0(title_prefix, ": categorical variables, -log10(raw p); text = raw p stars"),
    fontsize_number = 10,
    color = colorRampPalette(brewer.pal(n = 9, name = "YlOrRd"))(100)
  )
  dev.off()
}

make_simple_correlation_heatmap <- function(expr_mat, genes_tbl, meta_df, continuous_vars, outfile_prefix, title) {
  continuous_vars <- continuous_vars[continuous_vars %in% colnames(meta_df)]
  if (length(continuous_vars) == 0) return(invisible(NULL))

  expr_long <- expr_mat %>%
    as.data.frame() %>%
    rownames_to_column("gene") %>%
    pivot_longer(-gene, names_to = "specimenID", values_to = "expr") %>%
    left_join(genes_tbl, by = "gene") %>%
    mutate(Symbol = ifelse(is.na(Symbol), gene, Symbol)) %>%
    left_join(meta_df %>% select(specimenID, all_of(continuous_vars)), by = "specimenID")

  symbols <- unique(expr_long$Symbol)
  cor_matrix <- matrix(NA_real_, nrow = length(symbols), ncol = length(continuous_vars), dimnames = list(symbols, continuous_vars))
  pval_matrix <- matrix(NA_real_, nrow = length(symbols), ncol = length(continuous_vars), dimnames = list(symbols, continuous_vars))
  label_matrix <- matrix("", nrow = length(symbols), ncol = length(continuous_vars), dimnames = list(symbols, continuous_vars))

  for (symbol in symbols) {
    df_gene <- expr_long %>% filter(Symbol == symbol)
    for (var in continuous_vars) {
      ok <- complete.cases(df_gene$expr, df_gene[[var]])
      x <- df_gene$expr[ok]
      y <- df_gene[[var]][ok]
      if (length(x) < 4 || length(unique(x)) < 2 || length(unique(y)) < 2) next

      method <- ifelse(var %in% c("braaksc", "ceradsc", "CDR", "Braak", "CERAD"), "spearman", "pearson")
      test <- tryCatch(cor.test(x, y, method = method), error = function(e) NULL)
      if (is.null(test)) next
      cor_matrix[symbol, var] <- as.numeric(test$estimate)
      pval_matrix[symbol, var] <- test$p.value
      label_matrix[symbol, var] <- p_to_stars(test$p.value)
    }
  }

  write.csv(cor_matrix, paste0(outfile_prefix, "_cor_matrix.csv"), quote = FALSE)
  write.csv(pval_matrix, paste0(outfile_prefix, "_pval_matrix.csv"), quote = FALSE)
  write.csv(label_matrix, paste0(outfile_prefix, "_pvalue_stars_matrix.csv"), quote = FALSE)

  svg(paste0(outfile_prefix, "_heatmap_correlation.svg"), width = 7, height = 8)
  pheatmap(
    cor_matrix,
    display_numbers = label_matrix,
    cluster_rows = TRUE,
    cluster_cols = FALSE,
    main = title,
    fontsize_number = 10,
    color = colorRampPalette(rev(brewer.pal(n = 11, name = "RdBu")))(100),
    breaks = seq(-1, 1, length.out = 101)
  )
  dev.off()
}

run_cohort_lm_analysis <- function(cohort_name, expr_mat, meta_df, genes_tbl,
                                   continuous_vars, categorical_vars,
                                   base_covariates_main,
                                   cohort_outdir,
                                   run_batch_sensitivity = TRUE) {
  dir.create(cohort_outdir, recursive = TRUE, showWarnings = FALSE)

  continuous_vars <- continuous_vars[continuous_vars %in% colnames(meta_df)]
  categorical_vars <- categorical_vars[categorical_vars %in% colnames(meta_df)]
  clinical_vars <- c(continuous_vars, categorical_vars)
  base_covariates_main <- base_covariates_main[base_covariates_main %in% colnames(meta_df)]

  if (length(clinical_vars) == 0) {
    stop("No clinical variables found for cohort: ", cohort_name)
  }

  settings <- data.frame(
    cohort = cohort_name,
    continuous_vars = paste(continuous_vars, collapse = ";"),
    categorical_vars = paste(categorical_vars, collapse = ";"),
    base_covariates_main = paste(base_covariates_main, collapse = ";"),
    heatmap_continuous_value = "beta",
    heatmap_categorical_value = "-log10(raw p.value)",
    heatmap_text = "raw p-value stars",
    stringsAsFactors = FALSE
  )
  write.csv(settings, file.path(cohort_outdir, paste0(cohort_name, "_model_settings.csv")), row.names = FALSE, quote = FALSE)

  if ("sequencingBatch" %in% colnames(meta_df)) {
    batch_diag <- test_batch_clinical_association(
      meta_df = meta_df,
      batch_col = "sequencingBatch",
      clinical_vars = clinical_vars,
      outfile = file.path(cohort_outdir, paste0(cohort_name, "_batch_vs_clinical_diagnostic.csv"))
    )
    print(batch_diag)
  }

  lm_main <- gene_wise_lm(
    expr_mat = expr_mat,
    genes_tbl = genes_tbl,
    meta_df = meta_df,
    continuous_vars = continuous_vars,
    categorical_vars = categorical_vars,
    base_covariates = base_covariates_main,
    sample_col = "specimenID",
    model_label = paste0(cohort_name, "_merged_ComBat_main")
  )

  write.csv(lm_main, file.path(cohort_outdir, paste0(cohort_name, "_LM_results_long.csv")), row.names = FALSE, quote = FALSE)

  mats_main <- lm_to_matrices(lm_main, genes_tbl, clinical_vars)
  save_matrix_set(mats_main, file.path(cohort_outdir, paste0(cohort_name, "_LM")))

  save_continuous_heatmap_beta(
    mats = mats_main,
    continuous_vars = continuous_vars,
    prefix = file.path(cohort_outdir, cohort_name),
    title_prefix = paste0(cohort_name, " merged-ComBat expression")
  )

  save_categorical_heatmap_pvalue(
    mats = mats_main,
    categorical_vars = categorical_vars,
    prefix = file.path(cohort_outdir, cohort_name),
    title_prefix = paste0(cohort_name, " merged-ComBat expression")
  )

  make_simple_correlation_heatmap(
    expr_mat = expr_mat,
    genes_tbl = genes_tbl,
    meta_df = meta_df,
    continuous_vars = continuous_vars,
    outfile_prefix = file.path(cohort_outdir, paste0(cohort_name, "_merged_ComBat_simple_correlation")),
    title = paste0(cohort_name, " simple correlation: merged-ComBat expression")
  )

  lm_batch_sens <- NULL
  if (run_batch_sensitivity && "sequencingBatch" %in% colnames(meta_df)) {
    covars_batch <- unique(c(base_covariates_main, "sequencingBatch"))
    covars_batch <- covars_batch[covars_batch %in% colnames(meta_df)]

    lm_batch_sens <- gene_wise_lm(
      expr_mat = expr_mat,
      genes_tbl = genes_tbl,
      meta_df = meta_df,
      continuous_vars = continuous_vars,
      categorical_vars = categorical_vars,
      base_covariates = covars_batch,
      sample_col = "specimenID",
      model_label = paste0(cohort_name, "_merged_ComBat_sensitivity_with_batch_covariate")
    )

    write.csv(lm_batch_sens,
              file.path(cohort_outdir, paste0(cohort_name, "_LM_results_long_with_batch_covariate.csv")),
              row.names = FALSE, quote = FALSE)

    mats_batch <- lm_to_matrices(lm_batch_sens, genes_tbl, clinical_vars)
    save_matrix_set(mats_batch, file.path(cohort_outdir, paste0(cohort_name, "_LM_with_batch_covariate")))

    save_continuous_heatmap_beta(
      mats = mats_batch,
      continuous_vars = continuous_vars,
      prefix = file.path(cohort_outdir, paste0(cohort_name, "_with_batch_covariate")),
      title_prefix = paste0(cohort_name, " merged-ComBat expression + residual batch covariate")
    )

    save_categorical_heatmap_pvalue(
      mats = mats_batch,
      categorical_vars = categorical_vars,
      prefix = file.path(cohort_outdir, paste0(cohort_name, "_with_batch_covariate")),
      title_prefix = paste0(cohort_name, " merged-ComBat expression + residual batch covariate")
    )
  }

  list(main = lm_main, batch_sensitivity = lm_batch_sens)
}

selected_genes_raw <- read.csv(selected_gene_file)

gene_anno <- fread(gene_anno_file)
gene_anno <- gene_anno[, c(2, 5)]
names(gene_anno) <- c("Symbol", "gene")
gene_anno <- as.data.frame(gene_anno)

somascan <- read.csv(somascan_file)
somascan <- somascan[, c("Analytes", "EntrezGeneSymbol")]

selected_genes <- merge(selected_genes_raw, gene_anno, by = "gene", all.x = TRUE)
selected_genes$Symbol[is.na(selected_genes$Symbol)] <- selected_genes$gene[is.na(selected_genes$Symbol)]
selected_genes <- selected_genes[selected_genes$Symbol %in% somascan$EntrezGeneSymbol, ]
selected_genes <- selected_genes %>% select(gene, Symbol) %>% distinct()

write.csv(selected_genes, file.path(outdir, "selected_genes_used.csv"), row.names = FALSE, quote = FALSE)

expr_mat <- read.csv(expr_merged_combat_file, row.names = 1, check.names = FALSE)
expr_mat <- as.matrix(expr_mat)
storage.mode(expr_mat) <- "numeric"
colnames(expr_mat) <- sanitize_colnames(colnames(expr_mat))

meta <- fread(meta_merged_file) %>% as.data.frame()
if (!("cohort" %in% colnames(meta))) {
  stop("The merged metadata does not contain a 'cohort' column.")
}

possible_sample_cols <- c("specimenID", "sampleID", "sample_id", "SampleID", "IID", "id")
sample_col_merged <- first_existing_col(meta, possible_sample_cols)
if (is.na(sample_col_merged)) {
  stop("Could not find a sample ID column in merged metadata. Tried: ", paste(possible_sample_cols, collapse = ", "))
}
meta$specimenID <- sanitize_sample_ids(meta[[sample_col_merged]])

meta_rosmap_from_merged <- meta[meta$cohort == "rosmap", , drop = FALSE]
meta_msbb_from_merged <- meta[meta$cohort == "msbb", , drop = FALSE]

if (nrow(meta_rosmap_from_merged) == 0) {
  meta_rosmap_from_merged <- meta[tolower(meta$cohort) == "rosmap", , drop = FALSE]
}
if (nrow(meta_msbb_from_merged) == 0) {
  meta_msbb_from_merged <- meta[tolower(meta$cohort) == "msbb", , drop = FALSE]
}

if (nrow(meta_rosmap_from_merged) == 0) stop("No ROSMAP samples found in merged metadata.")
if (nrow(meta_msbb_from_merged) == 0) stop("No MSBB samples found in merged metadata.")

meta_rosmap_detail <- read.csv(rosmap_meta_file)
meta_rosmap_detail$specimenID <- sanitize_sample_ids(meta_rosmap_detail$specimenID)

meta_rosmap_detail <- meta_rosmap_detail %>%
  filter(tissue %in% c("dorsolateral prefrontal cortex", "frontal cortex")) %>%
  filter(dcfdx_lv %in% c(1, 2, 4)) %>%
  filter(assay == "rnaSeq")

meta_rosmap_detail$age_death_clean <- clean_numeric(meta_rosmap_detail$age_death)
meta_rosmap_detail$cts_mmse30_lv <- clean_numeric(meta_rosmap_detail$cts_mmse30_lv)
meta_rosmap_detail$educ <- clean_numeric(meta_rosmap_detail$educ)
meta_rosmap_detail$braaksc <- clean_numeric(meta_rosmap_detail$braaksc)
meta_rosmap_detail$ceradsc <- clean_numeric(meta_rosmap_detail$ceradsc)
meta_rosmap_detail$dcfdx_lv <- safe_factor(meta_rosmap_detail$dcfdx_lv)
meta_rosmap_detail$msex <- safe_factor(meta_rosmap_detail$msex)
if ("sequencingBatch" %in% colnames(meta_rosmap_detail)) {
  meta_rosmap_detail$sequencingBatch <- safe_factor(meta_rosmap_detail$sequencingBatch)
}

meta_rosmap <- meta_rosmap_from_merged %>%
  select(specimenID, cohort, everything()) %>%
  left_join(meta_rosmap_detail, by = "specimenID", suffix = c(".merged", ".detail"))

if (!("sequencingBatch" %in% colnames(meta_rosmap))) {
  if ("sequencingBatch.detail" %in% colnames(meta_rosmap)) {
    meta_rosmap$sequencingBatch <- meta_rosmap$sequencingBatch.detail
  } else if ("sequencingBatch.merged" %in% colnames(meta_rosmap)) {
    meta_rosmap$sequencingBatch <- meta_rosmap$sequencingBatch.merged
  }
}
if ("sequencingBatch" %in% colnames(meta_rosmap)) {
  meta_rosmap$sequencingBatch <- safe_factor(meta_rosmap$sequencingBatch)
}

if (!("msex" %in% colnames(meta_rosmap))) {
  if ("msex.detail" %in% colnames(meta_rosmap)) meta_rosmap$msex <- meta_rosmap$msex.detail
  if ("msex.merged" %in% colnames(meta_rosmap)) meta_rosmap$msex <- meta_rosmap$msex.merged
}
if ("msex" %in% colnames(meta_rosmap)) meta_rosmap$msex <- safe_factor(meta_rosmap$msex)

if (!("dcfdx_lv" %in% colnames(meta_rosmap))) {
  if ("dcfdx_lv.detail" %in% colnames(meta_rosmap)) meta_rosmap$dcfdx_lv <- meta_rosmap$dcfdx_lv.detail
  if ("dcfdx_lv.merged" %in% colnames(meta_rosmap)) meta_rosmap$dcfdx_lv <- meta_rosmap$dcfdx_lv.merged
}
if ("dcfdx_lv" %in% colnames(meta_rosmap)) meta_rosmap$dcfdx_lv <- safe_factor(meta_rosmap$dcfdx_lv)

meta_msbb_detail <- read.csv(msbb_meta_file)

if (!("specimenID" %in% colnames(meta_msbb_detail))) {
  sample_col_msbb_detail <- first_existing_col(meta_msbb_detail, possible_sample_cols)
  if (is.na(sample_col_msbb_detail)) {
    stop("Could not find sample ID column in MSBB detailed metadata.")
  }
  meta_msbb_detail$specimenID <- meta_msbb_detail[[sample_col_msbb_detail]]
}
meta_msbb_detail$specimenID <- sanitize_sample_ids(meta_msbb_detail$specimenID)

meta_msbb_detail <- meta_msbb_detail %>%
  filter(tissue %in% c("frontal pole", "inferior frontal gyrus", "prefrontal cortex"))

if ("ageDeath" %in% colnames(meta_msbb_detail)) {
  meta_msbb_detail$age_death_clean <- clean_numeric(meta_msbb_detail$ageDeath)
} else if ("age_death" %in% colnames(meta_msbb_detail)) {
  meta_msbb_detail$age_death_clean <- clean_numeric(meta_msbb_detail$age_death)
} else if ("ageAtDeath" %in% colnames(meta_msbb_detail)) {
  meta_msbb_detail$age_death_clean <- clean_numeric(meta_msbb_detail$ageAtDeath)
}

for (v in c("CDR", "Braak", "CERAD")) {
  if (v %in% colnames(meta_msbb_detail)) {
    meta_msbb_detail[[v]] <- clean_numeric(meta_msbb_detail[[v]])
  }
}

sex_col <- first_existing_col(meta_msbb_detail, c("msex", "sex", "Sex", "gender", "Gender"))
if (!is.na(sex_col)) {
  meta_msbb_detail$msex <- safe_factor(meta_msbb_detail[[sex_col]])
}

dx_col <- first_existing_col(meta_msbb_detail, c("diagnosis", "Diagnosis", "dx", "Dx", "diseaseStatus", "clinicalDiagnosis"))
if (!is.na(dx_col)) {
  meta_msbb_detail$diagnosis <- safe_factor(meta_msbb_detail[[dx_col]])
}

if ("sequencingBatch" %in% colnames(meta_msbb_detail)) {
  meta_msbb_detail$sequencingBatch <- safe_factor(meta_msbb_detail$sequencingBatch)
}

meta_msbb <- meta_msbb_from_merged %>%
  select(specimenID, cohort, everything()) %>%
  left_join(meta_msbb_detail, by = "specimenID", suffix = c(".merged", ".detail"))

if (!("sequencingBatch" %in% colnames(meta_msbb))) {
  if ("sequencingBatch.detail" %in% colnames(meta_msbb)) {
    meta_msbb$sequencingBatch <- meta_msbb$sequencingBatch.detail
  } else if ("sequencingBatch.merged" %in% colnames(meta_msbb)) {
    meta_msbb$sequencingBatch <- meta_msbb$sequencingBatch.merged
  }
}
if ("sequencingBatch" %in% colnames(meta_msbb)) {
  meta_msbb$sequencingBatch <- safe_factor(meta_msbb$sequencingBatch)
}

if (!("msex" %in% colnames(meta_msbb))) {
  if ("msex.detail" %in% colnames(meta_msbb)) meta_msbb$msex <- meta_msbb$msex.detail
  if ("msex.merged" %in% colnames(meta_msbb)) meta_msbb$msex <- meta_msbb$msex.merged
  if ("sex.detail" %in% colnames(meta_msbb)) meta_msbb$msex <- meta_msbb$sex.detail
  if ("sex.merged" %in% colnames(meta_msbb)) meta_msbb$msex <- meta_msbb$sex.merged
  if ("Sex.detail" %in% colnames(meta_msbb)) meta_msbb$msex <- meta_msbb$Sex.detail
  if ("Sex.merged" %in% colnames(meta_msbb)) meta_msbb$msex <- meta_msbb$Sex.merged
}
if ("msex" %in% colnames(meta_msbb)) meta_msbb$msex <- safe_factor(meta_msbb$msex)

if (!("diagnosis" %in% colnames(meta_msbb))) {
  if ("diagnosis.detail" %in% colnames(meta_msbb)) meta_msbb$diagnosis <- meta_msbb$diagnosis.detail
  if ("diagnosis.merged" %in% colnames(meta_msbb)) meta_msbb$diagnosis <- meta_msbb$diagnosis.merged
}
if ("diagnosis" %in% colnames(meta_msbb)) meta_msbb$diagnosis <- safe_factor(meta_msbb$diagnosis)

msbb_base_covariates_main <- unique(msbb_base_covariates_main_preferred[msbb_base_covariates_main_preferred %in% colnames(meta_msbb)])
if ("msex" %in% msbb_base_covariates_main) {
  msbb_base_covariates_main <- setdiff(msbb_base_covariates_main, c("sex", "Sex", "gender", "Gender"))
}
if ("diagnosis" %in% msbb_base_covariates_main) {
  msbb_base_covariates_main <- setdiff(msbb_base_covariates_main, c("Diagnosis", "dx", "Dx"))
}

expr_selected <- expr_mat[rownames(expr_mat) %in% selected_genes$gene, , drop = FALSE]
expr_selected <- remove_bad_genes(expr_selected)

aligned_rosmap <- align_expr_meta(expr_selected, meta_rosmap, sample_col = "specimenID")
expr_rosmap_selected <- aligned_rosmap$expr
meta_rosmap_aligned <- aligned_rosmap$meta
selected_genes_rosmap <- selected_genes %>% filter(gene %in% rownames(expr_rosmap_selected))

aligned_msbb <- align_expr_meta(expr_selected, meta_msbb, sample_col = "specimenID")
expr_msbb_selected <- aligned_msbb$expr
meta_msbb_aligned <- aligned_msbb$meta
selected_genes_msbb <- selected_genes %>% filter(gene %in% rownames(expr_msbb_selected))

message("ROSMAP samples used: ", ncol(expr_rosmap_selected))
message("ROSMAP selected genes used: ", nrow(expr_rosmap_selected))
message("MSBB samples used: ", ncol(expr_msbb_selected))
message("MSBB selected genes used: ", nrow(expr_msbb_selected))

if (!("msex" %in% colnames(meta_msbb_aligned))) {
  warning("MSBB msex was requested but could not be found or standardized. MSBB msex will be skipped.")
}
if (!("dcfdx_lv" %in% colnames(meta_rosmap_aligned))) {
  warning("ROSMAP dcfdx_lv was requested but could not be found. ROSMAP dcfdx_lv will be skipped.")
}
if (!("msex" %in% colnames(meta_rosmap_aligned))) {
  warning("ROSMAP msex was requested but could not be found. ROSMAP msex will be skipped.")
}

write.csv(data.frame(specimenID = colnames(expr_rosmap_selected)),
          file.path(outdir, "ROSMAP", "ROSMAP_samples_used.csv"), row.names = FALSE, quote = FALSE)
write.csv(selected_genes_rosmap,
          file.path(outdir, "ROSMAP", "ROSMAP_selected_genes_present.csv"), row.names = FALSE, quote = FALSE)
write.csv(data.frame(specimenID = colnames(expr_msbb_selected)),
          file.path(outdir, "MSBB", "MSBB_samples_used.csv"), row.names = FALSE, quote = FALSE)
write.csv(selected_genes_msbb,
          file.path(outdir, "MSBB", "MSBB_selected_genes_present.csv"), row.names = FALSE, quote = FALSE)

rosmap_results <- run_cohort_lm_analysis(
  cohort_name = "ROSMAP",
  expr_mat = expr_rosmap_selected,
  meta_df = meta_rosmap_aligned,
  genes_tbl = selected_genes_rosmap,
  continuous_vars = rosmap_continuous_vars,
  categorical_vars = rosmap_categorical_vars,
  base_covariates_main = rosmap_base_covariates_main,
  cohort_outdir = file.path(outdir, "ROSMAP"),
  run_batch_sensitivity = run_batch_sensitivity_model
)

msbb_results <- run_cohort_lm_analysis(
  cohort_name = "MSBB",
  expr_mat = expr_msbb_selected,
  meta_df = meta_msbb_aligned,
  genes_tbl = selected_genes_msbb,
  continuous_vars = msbb_continuous_vars,
  categorical_vars = msbb_categorical_vars,
  base_covariates_main = msbb_base_covariates_main,
  cohort_outdir = file.path(outdir, "MSBB"),
  run_batch_sensitivity = run_batch_sensitivity_model
)

ros_lm <- rosmap_results$main %>% mutate(cohort = "ROSMAP")
msbb_lm <- msbb_results$main %>% mutate(cohort = "MSBB")
combined_lm <- bind_rows(ros_lm, msbb_lm)
write.csv(combined_lm,
          file.path(outdir, "comparison", "ROSMAP_MSBB_LM_combined_long.csv"),
          row.names = FALSE, quote = FALSE)

comparison_rows <- list()
for (i in seq_len(nrow(comparison_variable_map))) {
  common_var <- comparison_variable_map$common_variable[i]
  rv <- comparison_variable_map$ROSMAP[i]
  mv <- comparison_variable_map$MSBB[i]
  note <- comparison_variable_map$note[i]

  rsub <- ros_lm %>%
    filter(variable == rv) %>%
    select(gene, Symbol, beta_ROSMAP = beta, p_ROSMAP = p.value, FDR_ROSMAP = FDR, n_ROSMAP = n, variable_type_ROSMAP = variable_type)

  msub <- msbb_lm %>%
    filter(variable == mv) %>%
    select(gene, Symbol, beta_MSBB = beta, p_MSBB = p.value, FDR_MSBB = FDR, n_MSBB = n, variable_type_MSBB = variable_type)

  cmp <- full_join(rsub, msub, by = c("gene", "Symbol")) %>%
    mutate(
      common_variable = common_var,
      ROSMAP_variable = rv,
      MSBB_variable = mv,
      note = note,
      same_beta_direction = ifelse(!is.na(beta_ROSMAP) & !is.na(beta_MSBB), sign(beta_ROSMAP) == sign(beta_MSBB), NA),
      same_beta_direction_cognition_sign_reversed = ifelse(
        common_variable == "cognition" & !is.na(beta_ROSMAP) & !is.na(beta_MSBB),
        sign(beta_ROSMAP) == sign(-beta_MSBB),
        NA
      )
    ) %>%
    select(common_variable, ROSMAP_variable, MSBB_variable, note, gene, Symbol,
           variable_type_ROSMAP, beta_ROSMAP, p_ROSMAP, FDR_ROSMAP, n_ROSMAP,
           variable_type_MSBB, beta_MSBB, p_MSBB, FDR_MSBB, n_MSBB,
           same_beta_direction, same_beta_direction_cognition_sign_reversed)

  comparison_rows[[length(comparison_rows) + 1]] <- cmp
}
comparison_overlap <- bind_rows(comparison_rows)
write.csv(comparison_overlap,
          file.path(outdir, "comparison", "ROSMAP_MSBB_LM_comparison_overlap_mapped_variables.csv"),
          row.names = FALSE, quote = FALSE)

comparison_summary <- comparison_overlap %>%
  group_by(common_variable, ROSMAP_variable, MSBB_variable) %>%
  summarise(
    n_genes_compared = sum(!is.na(beta_ROSMAP) & !is.na(beta_MSBB)),
    n_same_direction = sum(same_beta_direction %in% TRUE, na.rm = TRUE),
    fraction_same_direction = ifelse(n_genes_compared > 0, n_same_direction / n_genes_compared, NA_real_),
    n_same_direction_cognition_sign_reversed = sum(same_beta_direction_cognition_sign_reversed %in% TRUE, na.rm = TRUE),
    fraction_same_direction_cognition_sign_reversed = ifelse(
      common_variable == "cognition" & n_genes_compared > 0,
      n_same_direction_cognition_sign_reversed / n_genes_compared,
      NA_real_
    ),
    n_ROSMAP_raw_p05 = sum(p_ROSMAP < 0.05, na.rm = TRUE),
    n_MSBB_raw_p05 = sum(p_MSBB < 0.05, na.rm = TRUE),
    n_both_raw_p05 = sum(p_ROSMAP < 0.05 & p_MSBB < 0.05, na.rm = TRUE),
    .groups = "drop"
  )
write.csv(comparison_summary,
          file.path(outdir, "comparison", "ROSMAP_MSBB_LM_comparison_summary.csv"),
          row.names = FALSE, quote = FALSE)

sink(file.path(outdir, "sessionInfo.txt"))
print(sessionInfo())
sink()

message("Done. Results saved in: ", outdir)
