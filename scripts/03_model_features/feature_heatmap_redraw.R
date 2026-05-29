# Script cleaned for public release. Edit /path/to/... inputs before running.
rm(list = ls())
options(stringsAsFactors = FALSE)

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(tibble)
  library(pheatmap)
  library(RColorBrewer)
  library(grid)
})

project_dir <- "/path/to/project"
setwd(project_dir)

lm_outdir <- "out/clustering/linear_model_clinical_from_merged_combat_both_cohorts_exploratory"

redraw_outdir <- file.path(lm_outdir, "redrawn_compact_heatmaps_matched_order")
dir.create(redraw_outdir, recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(redraw_outdir, "ROSMAP"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(redraw_outdir, "MSBB"), recursive = TRUE, showWarnings = FALSE)

rosmap_continuous_vars <- c("cts_mmse30_lv", "educ", "age_death_clean", "braaksc", "ceradsc")
rosmap_categorical_vars <- c("dcfdx_lv", "msex")

msbb_continuous_vars <- c("CDR", "age_death_clean", "Braak", "CERAD")
msbb_categorical_vars <- c("msex")

variable_label_map <- c(
  cts_mmse30_lv = "MMSE",
  educ = "Education",
  age_death_clean = "Age at death",
  braaksc = "Braak",
  ceradsc = "CERAD",
  dcfdx_lv = "Diagnosis",
  msex = "Sex",
  CDR = "CDR",
  Braak = "Braak",
  CERAD = "CERAD"
)

gene_order_method <- "input"

keep_all_genes_in_each_panel <- TRUE

continuous_width_base <- 1.8
continuous_width_per_col <- 0.36
continuous_width_min <- 3.0
continuous_width_max <- 4.6

categorical_width_base <- 1.2
categorical_width_per_col <- 0.25
categorical_width_min <- 1.7
categorical_width_max <- 2.4

height_base <- 2.2
height_per_gene <- 0.23
height_min <- 4.2
height_max <- 11.0

fontsize_row <- 10
fontsize_col <- 12
fontsize_number <- 14
fontsize_main <- 10

cluster_rows <- FALSE
cluster_cols <- FALSE

show_raw_p_stars <- TRUE

p_to_stars <- function(p) {
  if (is.na(p)) return("")
  if (p > 0.05) return("")
  if (p > 0.01) return("*")
  if (p > 0.001) return("**")
  if (p > 0.0001) return("***")
  return("****")
}

safe_name <- function(x) {
  x <- gsub("[^A-Za-z0-9_.-]+", "_", x)
  x <- gsub("_+", "_", x)
  x
}

label_columns <- function(mat) {
  old <- colnames(mat)
  new <- ifelse(old %in% names(variable_label_map), variable_label_map[old], old)
  colnames(mat) <- new
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

read_lm_file <- function(path) {
  if (!file.exists(path)) {
    warning("File not found: ", path)
    return(NULL)
  }
  read.csv(path, check.names = FALSE)
}

get_symbol_order <- function(lm_df, continuous_vars, categorical_vars, method = "input") {
  lm_df <- lm_df %>%
    mutate(Symbol = ifelse(is.na(Symbol) | Symbol == "", gene, Symbol))

  all_vars <- c(continuous_vars, categorical_vars)
  sub <- lm_df %>% filter(variable %in% all_vars)

  if (nrow(sub) == 0) {
    return(character(0))
  }

  if (method == "input") {
    return(unique(sub$Symbol))
  }

  if (method == "continuous_mean_abs_beta") {
    cont <- sub %>%
      filter(variable %in% continuous_vars) %>%
      group_by(Symbol) %>%
      summarise(score = mean(abs(beta), na.rm = TRUE), .groups = "drop") %>%
      mutate(score = ifelse(is.nan(score), NA_real_, score))

    input_order <- data.frame(Symbol = unique(sub$Symbol), input_rank = seq_along(unique(sub$Symbol)))
    out <- input_order %>%
      left_join(cont, by = "Symbol") %>%
      arrange(desc(score), input_rank)
    return(out$Symbol)
  }

  if (method == "min_p") {
    ptab <- sub %>%
      group_by(Symbol) %>%
      summarise(score = suppressWarnings(min(p.value, na.rm = TRUE)), .groups = "drop") %>%
      mutate(score = ifelse(is.infinite(score), NA_real_, score))

    input_order <- data.frame(Symbol = unique(sub$Symbol), input_rank = seq_along(unique(sub$Symbol)))
    out <- input_order %>%
      left_join(ptab, by = "Symbol") %>%
      arrange(score, input_rank)
    return(out$Symbol)
  }

  stop("Unknown gene_order_method: ", method)
}

lm_to_matrices_from_long <- function(lm_df, vars, symbol_order) {
  lm_df <- lm_df %>%
    filter(variable %in% vars) %>%
    mutate(
      Symbol = ifelse(is.na(Symbol) | Symbol == "", gene, Symbol),
      p_stars = vapply(p.value, p_to_stars, character(1)),
      neg_log10p = ifelse(is.na(p.value), NA_real_, -log10(p.value)),
      signed_log10p = ifelse(
        is.na(beta) | is.na(p.value),
        NA_real_,
        sign(beta) * (-log10(p.value))
      )
    )

  vars_present <- vars[vars %in% unique(lm_df$variable)]
  symbols <- symbol_order

  beta <- matrix(NA_real_, nrow = length(symbols), ncol = length(vars_present),
                 dimnames = list(symbols, vars_present))
  pval <- matrix(NA_real_, nrow = length(symbols), ncol = length(vars_present),
                 dimnames = list(symbols, vars_present))
  fdr <- matrix(NA_real_, nrow = length(symbols), ncol = length(vars_present),
                dimnames = list(symbols, vars_present))
  stars <- matrix("", nrow = length(symbols), ncol = length(vars_present),
                  dimnames = list(symbols, vars_present))
  neg_log10p <- matrix(NA_real_, nrow = length(symbols), ncol = length(vars_present),
                       dimnames = list(symbols, vars_present))
  signed_log10p <- matrix(NA_real_, nrow = length(symbols), ncol = length(vars_present),
                          dimnames = list(symbols, vars_present))

  for (i in seq_len(nrow(lm_df))) {
    s <- lm_df$Symbol[i]
    v <- lm_df$variable[i]
    if (!(s %in% rownames(beta)) || !(v %in% colnames(beta))) next

    beta[s, v] <- lm_df$beta[i]
    pval[s, v] <- lm_df$p.value[i]
    fdr[s, v] <- lm_df$FDR[i]
    stars[s, v] <- lm_df$p_stars[i]
    neg_log10p[s, v] <- lm_df$neg_log10p[i]
    signed_log10p[s, v] <- lm_df$signed_log10p[i]
  }

  list(
    beta = beta,
    pval = pval,
    fdr = fdr,
    stars = stars,
    neg_log10p = neg_log10p,
    signed_log10p = signed_log10p
  )
}

drop_all_na_rows_if_needed <- function(mat, text_mat = NULL) {
  if (keep_all_genes_in_each_panel) {
    return(list(mat = mat, text = text_mat))
  }

  keep <- rowSums(!is.na(mat)) > 0
  if (!any(keep)) {
    return(list(mat = mat[FALSE, , drop = FALSE],
                text = if (!is.null(text_mat)) text_mat[FALSE, , drop = FALSE] else NULL))
  }

  list(mat = mat[keep, , drop = FALSE],
       text = if (!is.null(text_mat)) text_mat[keep, , drop = FALSE] else NULL)
}

plot_continuous_beta_heatmap <- function(lm_df, continuous_vars, symbol_order, out_prefix, title_prefix) {
  vars <- continuous_vars[continuous_vars %in% unique(lm_df$variable)]
  if (length(vars) == 0) {
    message("No continuous variables found for: ", title_prefix)
    return(invisible(NULL))
  }

  mats <- lm_to_matrices_from_long(lm_df, vars, symbol_order)

  beta_mat <- mats$beta[, vars, drop = FALSE]
  stars_mat <- mats$stars[, vars, drop = FALSE]

  dropped <- drop_all_na_rows_if_needed(beta_mat, stars_mat)
  beta_mat <- dropped$mat
  stars_mat <- dropped$text

  if (nrow(beta_mat) == 0) {
    message("No non-NA beta values for: ", title_prefix)
    return(invisible(NULL))
  }

  if (!show_raw_p_stars) {
    stars_mat[,] <- ""
  }

  max_abs <- max(abs(beta_mat), na.rm = TRUE)
  if (!is.finite(max_abs) || max_abs == 0) max_abs <- 1

  brks <- seq(-max_abs, max_abs, length.out = 101)
  cols <- colorRampPalette(rev(brewer.pal(n = 11, name = "RdBu")))(100)

  beta_plot <- label_columns(beta_mat)

  width <- compact_width_continuous(ncol(beta_plot))
  height <- compact_height(nrow(beta_plot))

  ph <- pheatmap(
    beta_plot,
    display_numbers = stars_mat,
    cluster_rows = FALSE,
    cluster_cols = FALSE,
    main = paste0(title_prefix, "\ncontinuous/ordinal variables: LM beta"),
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

  save_pheatmap_multi(ph, paste0(out_prefix, "_matched_order_continuous_heatmap_beta_raw_pstars"), width, height)

  write.csv(beta_mat, paste0(out_prefix, "_matched_order_continuous_beta_matrix.csv"), quote = FALSE)
  write.csv(stars_mat, paste0(out_prefix, "_matched_order_continuous_raw_pstars_matrix.csv"), quote = FALSE)

  invisible(ph)
}

plot_categorical_pvalue_heatmap <- function(lm_df, categorical_vars, symbol_order, out_prefix, title_prefix) {
  vars <- categorical_vars[categorical_vars %in% unique(lm_df$variable)]
  if (length(vars) == 0) {
    message("No categorical variables found for: ", title_prefix)
    return(invisible(NULL))
  }

  mats <- lm_to_matrices_from_long(lm_df, vars, symbol_order)

  logp_mat <- mats$neg_log10p[, vars, drop = FALSE]
  stars_mat <- mats$stars[, vars, drop = FALSE]

  logp_mat[!is.finite(logp_mat)] <- NA_real_

  dropped <- drop_all_na_rows_if_needed(logp_mat, stars_mat)
  logp_mat <- dropped$mat
  stars_mat <- dropped$text

  if (nrow(logp_mat) == 0) {
    message("No non-NA categorical p-values for: ", title_prefix)
    return(invisible(NULL))
  }

  if (!show_raw_p_stars) {
    stars_mat[,] <- ""
  }

  max_val <- max(logp_mat, na.rm = TRUE)
  if (!is.finite(max_val) || max_val == 0) max_val <- 1

  brks <- seq(0, max_val, length.out = 101)
  cols <- colorRampPalette(brewer.pal(n = 9, name = "YlOrRd"))(100)

  logp_plot <- label_columns(logp_mat)

  width <- compact_width_categorical(ncol(logp_plot))
  height <- compact_height(nrow(logp_plot))

  ph <- pheatmap(
    logp_plot,
    display_numbers = stars_mat,
    cluster_rows = FALSE,
    cluster_cols = FALSE,
    main = paste0(title_prefix, "\ncategorical variables: -log10(raw p)"),
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

  save_pheatmap_multi(ph, paste0(out_prefix, "_matched_order_categorical_heatmap_neglog10_raw_pstars"), width, height)

  write.csv(logp_mat, paste0(out_prefix, "_matched_order_categorical_neglog10p_matrix.csv"), quote = FALSE)
  write.csv(stars_mat, paste0(out_prefix, "_matched_order_categorical_raw_pstars_matrix.csv"), quote = FALSE)

  invisible(ph)
}

redraw_for_cohort <- function(cohort, continuous_vars, categorical_vars,
                              lm_file, out_subdir, model_label = "main") {
  dir.create(out_subdir, recursive = TRUE, showWarnings = FALSE)

  lm_df <- read_lm_file(lm_file)
  if (is.null(lm_df)) return(invisible(NULL))

  required <- c("gene", "Symbol", "variable", "variable_type", "beta", "p.value", "FDR")
  missing <- setdiff(required, colnames(lm_df))
  if (length(missing) > 0) {
    stop("Missing required columns in ", lm_file, ": ", paste(missing, collapse = ", "))
  }

  lm_df <- lm_df %>%
    mutate(Symbol = ifelse(is.na(Symbol) | Symbol == "", gene, Symbol))

  symbol_order <- get_symbol_order(
    lm_df = lm_df,
    continuous_vars = continuous_vars,
    categorical_vars = categorical_vars,
    method = gene_order_method
  )

  if (length(symbol_order) == 0) {
    message("No symbols found for: ", cohort, " ", model_label)
    return(invisible(NULL))
  }

  write.csv(
    data.frame(order = seq_along(symbol_order), Symbol = symbol_order),
    file.path(out_subdir, paste0(cohort, "_", model_label, "_matched_gene_order_used.csv")),
    row.names = FALSE,
    quote = FALSE
  )

  write.csv(
    lm_df,
    file.path(out_subdir, paste0(cohort, "_", model_label, "_LM_results_long_used_for_redraw.csv")),
    row.names = FALSE,
    quote = FALSE
  )

  out_prefix <- file.path(out_subdir, paste0(cohort, "_", model_label))

  plot_continuous_beta_heatmap(
    lm_df = lm_df,
    continuous_vars = continuous_vars,
    symbol_order = symbol_order,
    out_prefix = out_prefix,
    title_prefix = paste0(cohort, " ", model_label)
  )

  plot_categorical_pvalue_heatmap(
    lm_df = lm_df,
    categorical_vars = categorical_vars,
    symbol_order = symbol_order,
    out_prefix = out_prefix,
    title_prefix = paste0(cohort, " ", model_label)
  )

  invisible(TRUE)
}

rosmap_main_file <- file.path(lm_outdir, "ROSMAP", "ROSMAP_LM_results_long.csv")
msbb_main_file <- file.path(lm_outdir, "MSBB", "MSBB_LM_results_long.csv")

redraw_for_cohort(
  cohort = "ROSMAP",
  continuous_vars = rosmap_continuous_vars,
  categorical_vars = rosmap_categorical_vars,
  lm_file = rosmap_main_file,
  out_subdir = file.path(redraw_outdir, "ROSMAP"),
  model_label = "main"
)

redraw_for_cohort(
  cohort = "MSBB",
  continuous_vars = msbb_continuous_vars,
  categorical_vars = msbb_categorical_vars,
  lm_file = msbb_main_file,
  out_subdir = file.path(redraw_outdir, "MSBB"),
  model_label = "main"
)
