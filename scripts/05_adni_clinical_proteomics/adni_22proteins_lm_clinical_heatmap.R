#!/usr/bin/env Rscript
# Script cleaned for public release. Edit /path/to/... inputs before running.

rm(list = ls())
options(stringsAsFactors = FALSE)

project_dir <- "/path/to/project"
setwd(project_dir)

adni_input_file <- "out/ADNI/projected_rosmap_msbb_22gene_signature/tables/ADNI_projected_cluster_scores_with_clinical_all.csv"
marker_table_file <- "out/clustering/adni_projection_signature/tables/ROSMAP_MSBB_22gene_cluster_marker_table.csv"

outdir <- "out/ADNI/projected_rosmap_msbb_22gene_signature/lm_clinical_heatmap_22proteins_ALL"
table_dir <- file.path(outdir, "tables")
heatmap_dir <- file.path(outdir, "heatmaps")
dir.create(table_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(heatmap_dir, recursive = TRUE, showWarnings = FALSE)

continuous_vars <- c(
  "MMSCORE",
  "resilience_score",
  "ABETA42",
  "TAU",
  "PTAU",
  "age_at_visit",
  "PTEDUCAT",
  "max_primary_score",
  "score_margin"
)

categorical_vars <- c(
  "DIAGNOSIS",
  "sex",
  "ADpatho",
  "resilience",
  "predicted_cluster_like"
)

base_covariates_main <- c(
  "age_at_visit",
  "sex",
  "PTEDUCAT",
  "DIAGNOSIS"
)

categorical_like_terms <- c(
  "DIAGNOSIS",
  "sex",
  "ADpatho",
  "resilience",
  "predicted_cluster_like"
)

variable_label_map <- c(
  MMSCORE = "MMSE",
  resilience_score = "Resilience score",
  ABETA42 = "Aβ42",
  TAU = "Tau",
  PTAU = "pTau",
  age_at_visit = "Age",
  PTEDUCAT = "Education",
  max_primary_score = "Max projection score",
  score_margin = "Score margin",
  DIAGNOSIS = "Diagnosis",
  sex = "Sex",
  ADpatho = "AD pathology",
  resilience = "Resilience group",
  predicted_cluster_like = "Projected cluster-like"
)

reference_cluster_for_direction <- "cluster1"

cluster_rows_heatmap <- FALSE
cluster_cols_heatmap <- FALSE
show_raw_p_stars <- TRUE

continuous_width_base <- 3.2
continuous_width_per_col <- 0.52
continuous_width_min <- 4.6
continuous_width_max <- 7.4

categorical_width_base <- 2.5
categorical_width_per_col <- 0.65
categorical_width_min <- 3.4
categorical_width_max <- 5.2

height_base <- 2.2
height_per_gene <- 0.23
height_min <- 5.0
height_max <- 12.0

fontsize_row <- 14
fontsize_col <- 16
fontsize_number <- 14
fontsize_main <- 20

continuous_beta_abs_limit <- NULL

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(tibble)
  library(pheatmap)
  library(RColorBrewer)
  library(grid)
})

clean_numeric <- function(x) {
  x <- as.character(x)
  x[x %in% c("", "NA", "NaN", "NULL", "null")] <- NA
  suppressWarnings(as.numeric(x))
}

safe_factor <- function(x) {
  as.factor(as.character(x))
}

p_to_stars <- function(p) {
  if (is.na(p)) return("")
  if (p > 0.05) return("")
  if (p > 0.01) return("*")
  if (p > 0.001) return("**")
  if (p > 0.0001) return("***")
  return("****")
}

label_columns <- function(mat) {
  old <- colnames(mat)
  new <- ifelse(old %in% names(variable_label_map), variable_label_map[old], old)
  colnames(mat) <- new
  mat
}

clip_matrix <- function(mat, lim) {
  mat <- as.matrix(mat)
  mat[mat > lim] <- lim
  mat[mat < -lim] <- -lim
  mat
}

compact_width_continuous <- function(n_cols) {
  min(continuous_width_max, max(continuous_width_min, continuous_width_base + continuous_width_per_col * n_cols))
}

compact_width_categorical <- function(n_cols) {
  min(categorical_width_max, max(categorical_width_min, categorical_width_base + categorical_width_per_col * n_cols))
}

compact_height <- function(n_rows) {
  min(height_max, max(height_min, height_base + height_per_gene * n_rows))
}

save_pheatmap_multi <- function(ph, prefix, width, height) {
  svg(paste0(prefix, ".svg"), width = width, height = height)
  grid::grid.newpage()
  grid::grid.draw(ph$gtable)
  dev.off()

  pdf(paste0(prefix, ".pdf"), width = width, height = height)
  grid::grid.newpage()
  grid::grid.draw(ph$gtable)
  dev.off()

  png(paste0(prefix, ".png"), width = width, height = height, units = "in", res = 300)
  grid::grid.newpage()
  grid::grid.draw(ph$gtable)
  dev.off()
}

message("[1] Loading ADNI all-sample projected table and marker table")

if (!file.exists(adni_input_file)) {
  stop("ADNI input file not found: ", adni_input_file)
}
if (!file.exists(marker_table_file)) {
  stop("Marker table file not found: ", marker_table_file)
}

df <- read.csv(adni_input_file, check.names = FALSE)
marker <- read.csv(marker_table_file, check.names = FALSE)

required_marker_cols <- c("cluster", "Symbol", "delta_z_cluster_vs_others", "abs_delta_z")
missing_marker_cols <- setdiff(required_marker_cols, colnames(marker))
if (length(missing_marker_cols) > 0) {
  stop("Marker table is missing columns: ", paste(missing_marker_cols, collapse = ", "))
}

marker_ref <- marker %>%
  filter(cluster == reference_cluster_for_direction) %>%
  group_by(Symbol) %>%
  arrange(desc(abs_delta_z), .by_group = TRUE) %>%
  slice(1) %>%
  ungroup() %>%
  mutate(
    marker_direction = ifelse(delta_z_cluster_vs_others >= 0, "cluster1-high", "cluster2-high"),
    direction_rank = ifelse(marker_direction == "cluster1-high", 1, 2),
    sort_score = ifelse(marker_direction == "cluster1-high",
                        -delta_z_cluster_vs_others,
                        delta_z_cluster_vs_others)
  )

genes_all <- marker_ref %>%
  filter(Symbol %in% colnames(df)) %>%
  arrange(direction_rank, sort_score) %>%
  pull(Symbol) %>%
  unique()

if (length(genes_all) == 0) {
  stop("No marker-table genes were found as protein columns in ADNI input table.")
}

gene_annotation <- marker_ref %>%
  filter(Symbol %in% genes_all) %>%
  select(Symbol, marker_direction, delta_z_cluster_vs_others, abs_delta_z) %>%
  distinct(Symbol, .keep_all = TRUE) %>%
  arrange(match(Symbol, genes_all))

write.csv(gene_annotation,
          file.path(table_dir, "ADNI_ALL_22proteins_marker_direction_order.csv"),
          row.names = FALSE, quote = FALSE)

message("    ADNI samples: ", nrow(df))
message("    Proteins used: ", length(genes_all))
message("    Protein order: ", paste(genes_all, collapse = ", "))
message("    Marker direction counts:")
print(table(gene_annotation$marker_direction, useNA = "ifany"))

message("[2] Preparing clinical variables and covariates")

continuous_vars_present <- continuous_vars[continuous_vars %in% colnames(df)]
categorical_vars_present <- categorical_vars[categorical_vars %in% colnames(df)]
base_covariates_present <- base_covariates_main[base_covariates_main %in% colnames(df)]

for (v in continuous_vars_present) {
  df[[v]] <- clean_numeric(df[[v]])
}

for (v in unique(c(categorical_vars_present, intersect(base_covariates_present, categorical_like_terms)))) {
  if (v %in% colnames(df)) {
    df[[v]] <- droplevels(safe_factor(df[[v]]))
  }
}

for (v in setdiff(base_covariates_present, categorical_like_terms)) {
  if (v %in% colnames(df)) {
    df[[v]] <- clean_numeric(df[[v]])
  }
}

categorical_vars_present <- categorical_vars_present[
  sapply(categorical_vars_present, function(v) {
    if (!(v %in% colnames(df))) return(FALSE)
    length(unique(df[[v]][!is.na(df[[v]])])) >= 2
  })
]

continuous_vars_present <- continuous_vars_present[
  sapply(continuous_vars_present, function(v) {
    if (!(v %in% colnames(df))) return(FALSE)
    length(unique(df[[v]][!is.na(df[[v]])])) >= 2
  })
]

message("    Continuous variables: ", paste(continuous_vars_present, collapse = ", "))
message("    Categorical variables: ", paste(categorical_vars_present, collapse = ", "))
message("    Base covariates: ", paste(base_covariates_present, collapse = ", "))

settings <- data.frame(
  setting = c("input_file", "marker_table_file", "analysis_sample_set", "continuous_vars",
              "categorical_vars", "base_covariates", "genes"),
  value = c(
    adni_input_file,
    marker_table_file,
    "ADNI_all",
    paste(continuous_vars_present, collapse = ";"),
    paste(categorical_vars_present, collapse = ";"),
    paste(base_covariates_present, collapse = ";"),
    paste(genes_all, collapse = ";")
  )
)
write.csv(settings, file.path(table_dir, "ADNI_ALL_22proteins_LM_heatmap_settings.csv"),
          row.names = FALSE, quote = FALSE)

message("[3] Running protein-wise linear models")

protein_wise_lm <- function(df, protein_cols, continuous_vars, categorical_vars, base_covariates) {
  clinical_vars <- c(continuous_vars, categorical_vars)
  out <- list()

  for (gene in protein_cols) {
    for (v in clinical_vars) {
      if (!(v %in% colnames(df))) next

      covars <- setdiff(base_covariates, v)
      covars <- covars[covars %in% colnames(df)]

      rhs <- c(v, covars)
      use_cols <- c(gene, rhs)
      d <- df[, use_cols, drop = FALSE]
      names(d)[names(d) == gene] <- "expr"

      d <- d[complete.cases(d), , drop = FALSE]

      var_type <- ifelse(v %in% categorical_vars, "categorical", "continuous")

      if (nrow(d) < 10 || length(unique(d$expr)) < 2 || length(unique(d[[v]])) < 2) {
        out[[length(out) + 1]] <- data.frame(
          gene = gene,
          Symbol = gene,
          variable = v,
          variable_type = var_type,
          beta = NA_real_,
          statistic = NA_real_,
          p.value = NA_real_,
          n = nrow(d),
          covariates = paste(covars, collapse = ";"),
          formula = NA_character_,
          test = "insufficient_variation",
          stringsAsFactors = FALSE
        )
        next
      }

      for (term in names(d)) {
        if (term %in% categorical_like_terms) {
          d[[term]] <- droplevels(as.factor(d[[term]]))
        }
      }

      usable_rhs <- c()
      for (term in rhs) {
        z <- d[[term]]
        if (is.factor(z) && nlevels(droplevels(z)) < 2) next
        if (!is.factor(z) && length(unique(z)) < 2) next
        usable_rhs <- c(usable_rhs, term)
      }

      if (!(v %in% usable_rhs)) {
        out[[length(out) + 1]] <- data.frame(
          gene = gene,
          Symbol = gene,
          variable = v,
          variable_type = var_type,
          beta = NA_real_,
          statistic = NA_real_,
          p.value = NA_real_,
          n = nrow(d),
          covariates = paste(covars, collapse = ";"),
          formula = NA_character_,
          test = "tested_variable_dropped",
          stringsAsFactors = FALSE
        )
        next
      }

      full_form_txt <- paste("expr ~", paste(usable_rhs, collapse = " + "))
      full_form <- as.formula(full_form_txt)
      fit <- tryCatch(lm(full_form, data = d), error = function(e) NULL)

      if (is.null(fit)) {
        out[[length(out) + 1]] <- data.frame(
          gene = gene,
          Symbol = gene,
          variable = v,
          variable_type = var_type,
          beta = NA_real_,
          statistic = NA_real_,
          p.value = NA_real_,
          n = nrow(d),
          covariates = paste(setdiff(usable_rhs, v), collapse = ";"),
          formula = full_form_txt,
          test = "lm_failed",
          stringsAsFactors = FALSE
        )
        next
      }

      if (v %in% categorical_vars || is.factor(d[[v]])) {
        reduced_rhs <- setdiff(usable_rhs, v)
        reduced_form <- if (length(reduced_rhs) == 0) {
          as.formula("expr ~ 1")
        } else {
          as.formula(paste("expr ~", paste(reduced_rhs, collapse = " + ")))
        }
        fit0 <- tryCatch(lm(reduced_form, data = d), error = function(e) NULL)
        an <- if (is.null(fit0)) NULL else tryCatch(anova(fit0, fit), error = function(e) NULL)

        p <- if (is.null(an)) NA_real_ else an$`Pr(>F)`[2]
        stat <- if (is.null(an)) NA_real_ else an$F[2]
        beta <- NA_real_
        test_name <- "lm_nested_anova_categorical"
        var_type2 <- "categorical"

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
        var_type2 <- "continuous"
      }

      out[[length(out) + 1]] <- data.frame(
        gene = gene,
        Symbol = gene,
        variable = v,
        variable_type = var_type2,
        beta = beta,
        statistic = stat,
        p.value = p,
        n = nrow(d),
        covariates = paste(setdiff(usable_rhs, v), collapse = ";"),
        formula = full_form_txt,
        test = test_name,
        stringsAsFactors = FALSE
      )
    }
  }

  bind_rows(out) %>%
    group_by(variable) %>%
    mutate(FDR = p.adjust(p.value, method = "BH")) %>%
    ungroup() %>%
    mutate(
      p_stars = vapply(p.value, p_to_stars, character(1)),
      neg_log10p = ifelse(is.na(p.value), NA_real_, -log10(p.value))
    )
}

lm_results <- protein_wise_lm(
  df = df,
  protein_cols = genes_all,
  continuous_vars = continuous_vars_present,
  categorical_vars = categorical_vars_present,
  base_covariates = base_covariates_present
)

lm_results <- lm_results %>%
  left_join(gene_annotation, by = c("Symbol" = "Symbol"))

write.csv(lm_results,
          file.path(table_dir, "ADNI_ALL_22proteins_LM_clinical_association_long.csv"),
          row.names = FALSE, quote = FALSE)

message("[4] Building heatmap matrices")

lm_to_matrices <- function(lm_df, vars, row_order) {
  vars_present <- vars[vars %in% unique(lm_df$variable)]

  beta <- matrix(NA_real_, nrow = length(row_order), ncol = length(vars_present),
                 dimnames = list(row_order, vars_present))
  pval <- matrix(NA_real_, nrow = length(row_order), ncol = length(vars_present),
                 dimnames = list(row_order, vars_present))
  fdr <- matrix(NA_real_, nrow = length(row_order), ncol = length(vars_present),
                dimnames = list(row_order, vars_present))
  stars <- matrix("", nrow = length(row_order), ncol = length(vars_present),
                  dimnames = list(row_order, vars_present))
  neg_log10p <- matrix(NA_real_, nrow = length(row_order), ncol = length(vars_present),
                       dimnames = list(row_order, vars_present))

  for (i in seq_len(nrow(lm_df))) {
    g <- lm_df$Symbol[i]
    v <- lm_df$variable[i]
    if (!(g %in% rownames(beta)) || !(v %in% colnames(beta))) next

    beta[g, v] <- lm_df$beta[i]
    pval[g, v] <- lm_df$p.value[i]
    fdr[g, v] <- lm_df$FDR[i]
    stars[g, v] <- lm_df$p_stars[i]
    neg_log10p[g, v] <- lm_df$neg_log10p[i]
  }

  list(beta = beta, pval = pval, fdr = fdr, stars = stars, neg_log10p = neg_log10p)
}

continuous_mats <- lm_to_matrices(lm_results, continuous_vars_present, genes_all)
categorical_mats <- lm_to_matrices(lm_results, categorical_vars_present, genes_all)

write.csv(continuous_mats$beta, file.path(table_dir, "ADNI_ALL_22proteins_continuous_beta_matrix.csv"), quote = FALSE)
write.csv(continuous_mats$pval, file.path(table_dir, "ADNI_ALL_22proteins_continuous_pval_matrix.csv"), quote = FALSE)
write.csv(continuous_mats$fdr, file.path(table_dir, "ADNI_ALL_22proteins_continuous_FDR_matrix.csv"), quote = FALSE)
write.csv(continuous_mats$stars, file.path(table_dir, "ADNI_ALL_22proteins_continuous_raw_pstars_matrix.csv"), quote = FALSE)

write.csv(categorical_mats$neg_log10p, file.path(table_dir, "ADNI_ALL_22proteins_categorical_neglog10p_matrix.csv"), quote = FALSE)
write.csv(categorical_mats$pval, file.path(table_dir, "ADNI_ALL_22proteins_categorical_pval_matrix.csv"), quote = FALSE)
write.csv(categorical_mats$fdr, file.path(table_dir, "ADNI_ALL_22proteins_categorical_FDR_matrix.csv"), quote = FALSE)
write.csv(categorical_mats$stars, file.path(table_dir, "ADNI_ALL_22proteins_categorical_raw_pstars_matrix.csv"), quote = FALSE)

message("[5] Drawing heatmaps")

annotation_row <- gene_annotation %>%
  select(Symbol, marker_direction) %>%
  column_to_rownames("Symbol")
annotation_row <- annotation_row[genes_all, , drop = FALSE]

ann_colors <- list(
  marker_direction = c(
    "cluster1-high" = "#009E73",
    "cluster2-high" = "#E69F00",
    "other" = "gray70"
  )
)

if (length(continuous_vars_present) > 0) {
  beta_mat <- continuous_mats$beta[, continuous_vars_present, drop = FALSE]
  stars_mat <- continuous_mats$stars[, continuous_vars_present, drop = FALSE]

  if (!show_raw_p_stars) {
    stars_mat[,] <- ""
  }

  max_abs <- if (is.null(continuous_beta_abs_limit)) {
    max(abs(beta_mat), na.rm = TRUE)
  } else {
    continuous_beta_abs_limit
  }
  if (!is.finite(max_abs) || max_abs == 0) max_abs <- 1

  beta_plot <- clip_matrix(beta_mat, max_abs)
  beta_plot <- label_columns(beta_plot)

  brks <- seq(-max_abs, max_abs, length.out = 101)
  cols <- colorRampPalette(rev(brewer.pal(n = 11, name = "RdBu")))(100)

  width <- compact_width_continuous(ncol(beta_plot))
  height <- compact_height(nrow(beta_plot))

  ph_cont <- pheatmap(
    beta_plot,
    display_numbers = stars_mat,
    annotation_row = annotation_row,
    annotation_colors = ann_colors,
    cluster_rows = cluster_rows_heatmap,
    cluster_cols = cluster_cols_heatmap,
    main = "ADNI all samples: 22 proteins vs clinical variables\nlinear-model beta; text = raw p-value stars",
    fontsize = 16,
    fontsize_row = fontsize_row,
    fontsize_col = fontsize_col,
    fontsize_number = fontsize_number,
    fontsize_main = fontsize_main,
    color = cols,
    breaks = brks,
    border_color = NA,
    angle_col = 90,
    silent = TRUE
  )

  save_pheatmap_multi(
    ph_cont,
    file.path(heatmap_dir, "ADNI_ALL_22proteins_continuous_LM_beta_heatmap"),
    width = width,
    height = height
  )
}

if (length(categorical_vars_present) > 0) {
  logp_mat <- categorical_mats$neg_log10p[, categorical_vars_present, drop = FALSE]
  stars_mat <- categorical_mats$stars[, categorical_vars_present, drop = FALSE]
  logp_mat[!is.finite(logp_mat)] <- NA_real_

  if (!show_raw_p_stars) {
    stars_mat[,] <- ""
  }

  max_val <- max(logp_mat, na.rm = TRUE)
  if (!is.finite(max_val) || max_val == 0) max_val <- 1

  logp_plot <- label_columns(logp_mat)

  brks <- seq(0, max_val, length.out = 101)
  cols <- colorRampPalette(brewer.pal(n = 9, name = "YlOrRd"))(100)

  width <- compact_width_categorical(ncol(logp_plot))
  height <- compact_height(nrow(logp_plot))

  ph_cat <- pheatmap(
    logp_plot,
    display_numbers = stars_mat,
    annotation_row = annotation_row,
    annotation_colors = ann_colors,
    cluster_rows = cluster_rows_heatmap,
    cluster_cols = FALSE,
    main = "ADNI all samples: categorical associations\n- log10(raw p); text = raw p-value stars",
    fontsize = 8,
    fontsize_row = fontsize_row,
    fontsize_col = fontsize_col,
    fontsize_number = fontsize_number,
    fontsize_main = fontsize_main,
    color = cols,
    breaks = brks,
    border_color = NA,
    angle_col = 90,
    silent = TRUE
  )

  save_pheatmap_multi(
    ph_cat,
    file.path(heatmap_dir, "ADNI_ALL_22proteins_categorical_neglog10p_heatmap"),
    width = width,
    height = height
  )
}

sink(file.path(outdir, "sessionInfo_ADNI_ALL_22proteins_LM_clinical_heatmap.txt"))
print(sessionInfo())
sink()

message("Done. Results saved in: ", outdir)
