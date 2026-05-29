#!/usr/bin/env Rscript
# Script cleaned for public release. Edit /path/to/... inputs before running.

rm(list = ls())
options(stringsAsFactors = FALSE)

project_dir <- "/path/to/project"
setwd(project_dir)

ucd_result_dir <- "out/ADNI/projected_rosmap_msbb_22gene_signature/imaging_UCD_WMH_covariate_adjusted_primary"
table_dir <- file.path(ucd_result_dir, "tables")

contrast_file <- file.path(
  table_dir,
  "UCD_WMH_primary_adjusted_LM_pairwise_contrasts_all_metrics_BH_FDR.csv"
)

forest_dir <- file.path(ucd_result_dir, "forest_plots")
dir.create(forest_dir, recursive = TRUE, showWarnings = FALSE)

main_metrics <- c(
  "TOTAL_CSF",
  "TOTAL_GRAY",
  "CEREBRUM_GRAY",
  "CEREBRUM_TCB",
  "TOTAL_HIPPO",
  "log10_TOTAL_WMH_plus1"
)

include_cerebrum_tcc_in_main <- FALSE

if (include_cerebrum_tcc_in_main) {
  main_metrics <- c(
    "TOTAL_CSF",
    "TOTAL_GRAY",
    "CEREBRUM_GRAY",
    "CEREBRUM_TCB",
    "CEREBRUM_TCC",
    "TOTAL_HIPPO",
    "log10_TOTAL_WMH_plus1"
  )
}

main_contrasts <- c(
  "cluster1_vs_non_resilience",
  "cluster2_vs_non_resilience"
)

supplementary_contrasts <- c(
  "cluster1_vs_non_resilience",
  "cluster2_vs_non_resilience",
  "cluster1_vs_cluster2"
)

q_threshold <- 0.05

group_colors <- c(
  "cluster1" = "#009E73",
  "cluster2" = "#E69F00",
  "non_resilience" = "gray60"
)

contrast_colors <- c(
  "cluster1_vs_non_resilience" = group_colors[["cluster1"]],
  "cluster2_vs_non_resilience" = group_colors[["cluster2"]],
  "cluster1_vs_cluster2" = "#4D4D4D"
)

main_width <- 8.4
main_height <- 5.3

main_label_width <- 9.4
main_label_height <- 5.3

supp_width <- 9.2
supp_height <- 8.6

supp_all_width <- 10.5
supp_all_height <- 8.6

base_size <- 16

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(forcats)
  library(readr)
  library(stringr)
})

clean_numeric <- function(x) {
  x <- as.character(x)
  x[x %in% c("", "NA", "NaN", "NULL", "null", ".", " ")] <- NA
  suppressWarnings(as.numeric(x))
}

p_to_stars <- function(p) {
  if (is.na(p)) return("")
  if (p > 0.05) return("ns")
  if (p > 0.01) return("*")
  if (p > 0.001) return("**")
  if (p > 0.0001) return("***")
  return("****")
}

q_to_stars <- function(q) {
  if (is.na(q)) return("")
  if (q > 0.05) return("ns")
  if (q > 0.01) return("*")
  if (q > 0.001) return("**")
  if (q > 0.0001) return("***")
  return("****")
}

metric_label <- function(x) {
  dplyr::case_when(
    x == "TOTAL_HIPPO" ~ "Total hippocampus",
    x == "TOTAL_CSF" ~ "Total CSF",
    x == "TOTAL_GRAY" ~ "Total gray matter",
    x == "TOTAL_WHITE" ~ "Total white matter",
    x == "TOTAL_BRAIN" ~ "Total brain",
    x == "CEREBRUM_TCV" ~ "Cerebrum TCV",
    x == "CEREBRUM_TCB" ~ "Cerebrum tissue volume",
    x == "CEREBRUM_TCC" ~ "Cerebrum CSF compartment",
    x == "CEREBRUM_GRAY" ~ "Cerebrum gray matter",
    x == "CEREBRUM_WHITE" ~ "Cerebrum white matter",
    x == "LEFT_HIPPO" ~ "Left hippocampus",
    x == "RIGHT_HIPPO" ~ "Right hippocampus",
    x == "log10_TOTAL_WMH_plus1" ~ "WMH volume, log10(x + 1)",
    TRUE ~ x
  )
}

contrast_label <- function(x) {
  dplyr::case_when(
    x == "cluster1_vs_non_resilience" ~ "Cluster 1 vs non-CR",
    x == "cluster2_vs_non_resilience" ~ "Cluster 2 vs non-CR",
    x == "cluster1_vs_cluster2" ~ "Cluster 1 vs Cluster 2",
    TRUE ~ x
  )
}

format_q_label <- function(q) {
  dplyr::case_when(
    is.na(q) ~ "q = NA",
    q < 0.001 ~ "q < 0.001",
    TRUE ~ paste0("q = ", sprintf("%.3f", q))
  )
}

add_standardized_ci <- function(df) {
  df %>%
    mutate(
      estimate = clean_numeric(estimate),
      lower.CL = clean_numeric(lower.CL),
      upper.CL = clean_numeric(upper.CL),
      standardized_difference = clean_numeric(standardized_difference),
      model_sigma_reconstructed = dplyr::case_when(
        !is.na(estimate) &
          !is.na(standardized_difference) &
          abs(standardized_difference) > 1e-12 ~ estimate / standardized_difference,
        TRUE ~ NA_real_
      ),
      std_lower.CL = dplyr::case_when(
        !is.na(model_sigma_reconstructed) &
          abs(model_sigma_reconstructed) > 1e-12 ~ lower.CL / model_sigma_reconstructed,
        TRUE ~ NA_real_
      ),
      std_upper.CL = dplyr::case_when(
        !is.na(model_sigma_reconstructed) &
          abs(model_sigma_reconstructed) > 1e-12 ~ upper.CL / model_sigma_reconstructed,
        TRUE ~ NA_real_
      )
    )
}

add_fallback_standardized_ci <- function(df) {
  df %>%
    mutate(
      SE = clean_numeric(SE),
      se_std = dplyr::case_when(
        !is.na(model_sigma_reconstructed) &
          abs(model_sigma_reconstructed) > 1e-12 ~ SE / model_sigma_reconstructed,
        TRUE ~ NA_real_
      ),
      std_lower.CL = ifelse(
        is.na(std_lower.CL) & !is.na(se_std),
        standardized_difference - 1.96 * se_std,
        std_lower.CL
      ),
      std_upper.CL = ifelse(
        is.na(std_upper.CL) & !is.na(se_std),
        standardized_difference + 1.96 * se_std,
        std_upper.CL
      )
    )
}

save_plot_multi <- function(plot_obj, prefix, width, height) {
  ggsave(paste0(prefix, ".svg"), plot_obj, width = width, height = height, units = "in")
  ggsave(paste0(prefix, ".pdf"), plot_obj, width = width, height = height, units = "in")
  ggsave(paste0(prefix, ".png"), plot_obj, width = width, height = height, units = "in", dpi = 300)
}

message("[1] Loading primary contrast table")

if (!file.exists(contrast_file)) {
  stop("Contrast file not found: ", contrast_file)
}

contrast_df <- read.csv(contrast_file, check.names = FALSE)

required_cols <- c(
  "metric",
  "contrast",
  "estimate",
  "SE",
  "p.value",
  "lower.CL",
  "upper.CL",
  "standardized_difference",
  "FDR_all_metrics_all_contrasts"
)

missing_cols <- setdiff(required_cols, colnames(contrast_df))
if (length(missing_cols) > 0) {
  stop("Contrast table is missing required columns: ", paste(missing_cols, collapse = ", "))
}

contrast_label_colors <- setNames(
  contrast_colors[names(contrast_colors)],
  contrast_label(names(contrast_colors))
)

contrast_df <- contrast_df %>%
  mutate(
    p.value = clean_numeric(p.value),
    FDR_all_metrics_all_contrasts = clean_numeric(FDR_all_metrics_all_contrasts),
    n = if ("n" %in% colnames(.)) clean_numeric(n) else NA_real_
  ) %>%
  add_standardized_ci() %>%
  add_fallback_standardized_ci() %>%
  mutate(
    metric_label = metric_label(metric),
    contrast_label = contrast_label(contrast),
    significant_q05 = !is.na(FDR_all_metrics_all_contrasts) &
      FDR_all_metrics_all_contrasts < q_threshold,
    q_label = format_q_label(FDR_all_metrics_all_contrasts),
    q_stars_plot = vapply(FDR_all_metrics_all_contrasts, q_to_stars, character(1)),
    p_stars_plot = vapply(p.value, p_to_stars, character(1))
  )

write.csv(
  contrast_df,
  file.path(forest_dir, "UCD_primary_contrasts_for_forest_plot_cleaned.csv"),
  row.names = FALSE,
  quote = FALSE
)

message("    Rows loaded: ", nrow(contrast_df))
message("    Metrics: ", paste(unique(contrast_df$metric), collapse = ", "))
message("    Contrasts: ", paste(unique(contrast_df$contrast), collapse = ", "))
message("    Color mapping:")
print(contrast_label_colors)

message("[2] Drawing main forest plot")

main_df <- contrast_df %>%
  filter(metric %in% main_metrics) %>%
  filter(contrast %in% main_contrasts)

main_metric_order <- rev(main_metrics[main_metrics %in% main_df$metric])

main_df <- main_df %>%
  mutate(
    metric = factor(metric, levels = main_metric_order),
    metric_label = factor(metric_label, levels = metric_label(main_metric_order)),
    contrast = factor(contrast, levels = main_contrasts),
    contrast_label = factor(contrast_label, levels = contrast_label(main_contrasts))
  )

if (nrow(main_df) == 0) {
  warning("No rows available for main forest plot. Check main_metrics and main_contrasts.")
} else {
  dodge_main <- position_dodge(width = 0.55)

  p_main <- ggplot(
    main_df,
    aes(
      x = standardized_difference,
      y = metric_label,
      color = contrast_label,
      shape = significant_q05
    )
  ) +
    geom_vline(
      xintercept = 0,
      linetype = "dashed",
      linewidth = 0.5,
      color = "gray40"
    ) +
    geom_errorbarh(
      aes(xmin = std_lower.CL, xmax = std_upper.CL),
      height = 0.18,
      linewidth = 0.7,
      position = dodge_main,
      na.rm = TRUE
    ) +
    geom_point(
      size = 3.2,
      stroke = 0.9,
      position = dodge_main,
      na.rm = TRUE
    ) +
    scale_color_manual(
      values = contrast_label_colors,
      name = "Projected group contrast"
    ) +
    scale_shape_manual(
      values = c("TRUE" = 16, "FALSE" = 1),
      labels = c("TRUE" = "BH-FDR q < 0.05", "FALSE" = "q >= 0.05"),
      name = "Statistical significance"
    ) +
    labs(
      title = "UCD WMH/global-volume metrics: covariate-adjusted group differences",
      subtitle = "Colors follow the projected CR subtype palette; points show adjusted contrasts relative to non-CR",
      x = "Standardized adjusted difference",
      y = NULL
    ) +
    theme_bw(base_size = base_size) +
    theme(
      plot.title = element_text(face = "bold", size = base_size + 2),
      plot.subtitle = element_text(size = base_size - 3),
      axis.text.y = element_text(size = base_size),
      axis.text.x = element_text(size = base_size - 1),
      axis.title.x = element_text(size = base_size),
      legend.position = "right",
      legend.title = element_text(size = base_size - 2),
      legend.text = element_text(size = base_size - 3),
      panel.grid.major.y = element_line(color = "gray90"),
      panel.grid.minor = element_blank()
    )

  save_plot_multi(
    p_main,
    file.path(forest_dir, "UCD_primary_forest_main_representative_metrics_colored"),
    width = main_width,
    height = main_height
  )
}

message("[3] Drawing main forest plot with q labels")

if (nrow(main_df) > 0) {
  label_df <- main_df %>%
    mutate(
      label_text = ifelse(
        significant_q05,
        paste0(q_stars_plot, " (", q_label, ")"),
        ""
      )
    )

  x_max <- max(label_df$std_upper.CL, label_df$standardized_difference, na.rm = TRUE)
  x_min <- min(label_df$std_lower.CL, label_df$standardized_difference, na.rm = TRUE)
  x_pad <- 0.10 * (x_max - x_min)
  if (!is.finite(x_pad) || x_pad == 0) x_pad <- 0.1

  dodge_main_label <- position_dodge(width = 0.55)

  p_main_label <- ggplot(
    label_df,
    aes(
      x = standardized_difference,
      y = metric_label,
      color = contrast_label,
      shape = significant_q05
    )
  ) +
    geom_vline(
      xintercept = 0,
      linetype = "dashed",
      linewidth = 0.5,
      color = "gray40"
    ) +
    geom_errorbarh(
      aes(xmin = std_lower.CL, xmax = std_upper.CL),
      height = 0.18,
      linewidth = 0.7,
      position = dodge_main_label,
      na.rm = TRUE
    ) +
    geom_point(
      size = 3.2,
      stroke = 0.9,
      position = dodge_main_label,
      na.rm = TRUE
    ) +
    geom_text(
      aes(label = label_text),
      position = dodge_main_label,
      hjust = -0.10,
      size = 4.0,
      show.legend = FALSE,
      na.rm = TRUE
    ) +
    coord_cartesian(xlim = c(x_min - x_pad, x_max + 2.4 * x_pad), clip = "off") +
    scale_color_manual(
      values = contrast_label_colors,
      name = "Projected group contrast"
    ) +
    scale_shape_manual(
      values = c("TRUE" = 16, "FALSE" = 1),
      labels = c("TRUE" = "BH-FDR q < 0.05", "FALSE" = "q >= 0.05"),
      name = "Statistical significance"
    ) +
    labs(
      title = "UCD WMH/global-volume metrics: covariate-adjusted group differences",
      subtitle = "Labels indicate BH-FDR q-values for significant contrasts",
      x = "Standardized adjusted difference",
      y = NULL
    ) +
    theme_bw(base_size = base_size) +
    theme(
      plot.title = element_text(face = "bold", size = base_size + 2),
      plot.subtitle = element_text(size = base_size - 3),
      axis.text.y = element_text(size = base_size),
      axis.text.x = element_text(size = base_size - 1),
      axis.title.x = element_text(size = base_size),
      legend.position = "right",
      legend.title = element_text(size = base_size - 2),
      legend.text = element_text(size = base_size - 3),
      panel.grid.major.y = element_line(color = "gray90"),
      panel.grid.minor = element_blank(),
      plot.margin = margin(t = 8, r = 34, b = 8, l = 8)
    )

  save_plot_multi(
    p_main_label,
    file.path(forest_dir, "UCD_primary_forest_main_representative_metrics_colored_with_q_labels"),
    width = main_label_width,
    height = main_label_height
  )
}

message("[4] Drawing supplementary forest plot: all metrics, cluster vs non-CR")

all_metric_order <- c(
  "TOTAL_CSF",
  "TOTAL_GRAY",
  "TOTAL_WHITE",
  "TOTAL_BRAIN",
  "TOTAL_HIPPO",
  "LEFT_HIPPO",
  "RIGHT_HIPPO",
  "CEREBRUM_TCV",
  "CEREBRUM_TCB",
  "CEREBRUM_TCC",
  "CEREBRUM_GRAY",
  "CEREBRUM_WHITE",
  "log10_TOTAL_WMH_plus1"
)

all_metric_order <- all_metric_order[all_metric_order %in% contrast_df$metric]

supp_df_2contrast <- contrast_df %>%
  filter(metric %in% all_metric_order) %>%
  filter(contrast %in% main_contrasts) %>%
  mutate(
    metric = factor(metric, levels = rev(all_metric_order)),
    metric_label = factor(metric_label, levels = metric_label(rev(all_metric_order))),
    contrast = factor(contrast, levels = main_contrasts),
    contrast_label = factor(contrast_label, levels = contrast_label(main_contrasts))
  )

if (nrow(supp_df_2contrast) > 0) {
  dodge_supp <- position_dodge(width = 0.55)

  p_supp_2contrast <- ggplot(
    supp_df_2contrast,
    aes(
      x = standardized_difference,
      y = metric_label,
      color = contrast_label,
      shape = significant_q05
    )
  ) +
    geom_vline(
      xintercept = 0,
      linetype = "dashed",
      linewidth = 0.5,
      color = "gray40"
    ) +
    geom_errorbarh(
      aes(xmin = std_lower.CL, xmax = std_upper.CL),
      height = 0.18,
      linewidth = 0.65,
      position = dodge_supp,
      na.rm = TRUE
    ) +
    geom_point(
      size = 2.9,
      stroke = 0.85,
      position = dodge_supp,
      na.rm = TRUE
    ) +
    scale_color_manual(
      values = contrast_label_colors,
      name = "Projected group contrast"
    ) +
    scale_shape_manual(
      values = c("TRUE" = 16, "FALSE" = 1),
      labels = c("TRUE" = "BH-FDR q < 0.05", "FALSE" = "q >= 0.05"),
      name = "Statistical significance"
    ) +
    labs(
      title = "All UCD WMH/global-volume metrics: covariate-adjusted contrasts",
      subtitle = "Cluster 1 and cluster 2 compared with non-CR",
      x = "Standardized adjusted difference",
      y = NULL
    ) +
    theme_bw(base_size = base_size) +
    theme(
      plot.title = element_text(face = "bold", size = base_size + 1),
      plot.subtitle = element_text(size = base_size - 3),
      axis.text.y = element_text(size = base_size - 1),
      axis.text.x = element_text(size = base_size - 1),
      axis.title.x = element_text(size = base_size),
      legend.position = "right",
      legend.title = element_text(size = base_size - 2),
      legend.text = element_text(size = base_size - 3),
      panel.grid.major.y = element_line(color = "gray90"),
      panel.grid.minor = element_blank()
    )

  save_plot_multi(
    p_supp_2contrast,
    file.path(forest_dir, "UCD_primary_forest_supplementary_all_metrics_cluster_vs_nonCR_colored"),
    width = supp_width,
    height = supp_height
  )
}

message("[5] Drawing supplementary forest plot: all metrics, all contrasts")

supp_df_all <- contrast_df %>%
  filter(metric %in% all_metric_order) %>%
  filter(contrast %in% supplementary_contrasts) %>%
  mutate(
    metric = factor(metric, levels = rev(all_metric_order)),
    metric_label = factor(metric_label, levels = metric_label(rev(all_metric_order))),
    contrast = factor(contrast, levels = supplementary_contrasts),
    contrast_label = factor(contrast_label, levels = contrast_label(supplementary_contrasts))
  )

if (nrow(supp_df_all) > 0) {
  dodge_supp_all <- position_dodge(width = 0.65)

  p_supp_all <- ggplot(
    supp_df_all,
    aes(
      x = standardized_difference,
      y = metric_label,
      color = contrast_label,
      shape = significant_q05
    )
  ) +
    geom_vline(
      xintercept = 0,
      linetype = "dashed",
      linewidth = 0.5,
      color = "gray40"
    ) +
    geom_errorbarh(
      aes(xmin = std_lower.CL, xmax = std_upper.CL),
      height = 0.18,
      linewidth = 0.6,
      position = dodge_supp_all,
      na.rm = TRUE
    ) +
    geom_point(
      size = 2.6,
      stroke = 0.8,
      position = dodge_supp_all,
      na.rm = TRUE
    ) +
    scale_color_manual(
      values = contrast_label_colors,
      name = "Projected group contrast"
    ) +
    scale_shape_manual(
      values = c("TRUE" = 16, "FALSE" = 1),
      labels = c("TRUE" = "BH-FDR q < 0.05", "FALSE" = "q >= 0.05"),
      name = "Statistical significance"
    ) +
    labs(
      title = "All UCD WMH/global-volume metrics: all prespecified contrasts",
      subtitle = "Covariate-adjusted linear models with BH-FDR correction across all metrics and contrasts",
      x = "Standardized adjusted difference",
      y = NULL
    ) +
    theme_bw(base_size = base_size) +
    theme(
      plot.title = element_text(face = "bold", size = base_size + 1),
      plot.subtitle = element_text(size = base_size - 3),
      axis.text.y = element_text(size = base_size - 1),
      axis.text.x = element_text(size = base_size - 1),
      axis.title.x = element_text(size = base_size),
      legend.position = "right",
      legend.title = element_text(size = base_size - 2),
      legend.text = element_text(size = base_size - 3),
      panel.grid.major.y = element_line(color = "gray90"),
      panel.grid.minor = element_blank()
    )

  save_plot_multi(
    p_supp_all,
    file.path(forest_dir, "UCD_primary_forest_supplementary_all_metrics_all_contrasts_colored"),
    width = supp_all_width,
    height = supp_all_height
  )
}

message("[6] Writing simplified supplementary tables")

supp_table <- contrast_df %>%
  transmute(
    metric,
    metric_label,
    contrast,
    contrast_label,
    estimate,
    lower_95CI = lower.CL,
    upper_95CI = upper.CL,
    standardized_difference,
    standardized_lower_95CI = std_lower.CL,
    standardized_upper_95CI = std_upper.CL,
    SE,
    df,
    t.ratio,
    p.value,
    FDR_BH = FDR_all_metrics_all_contrasts,
    n,
    significant_q05,
    covariates = if ("covariates" %in% colnames(contrast_df)) covariates else NA_character_
  )

write.csv(
  supp_table,
  file.path(forest_dir, "Supplementary_Table_UCD_primary_adjusted_pairwise_contrasts_simplified.csv"),
  row.names = FALSE,
  quote = FALSE
)

sig_table <- supp_table %>%
  filter(!is.na(FDR_BH), FDR_BH < q_threshold)

write.csv(
  sig_table,
  file.path(forest_dir, "Supplementary_Table_UCD_primary_adjusted_pairwise_contrasts_significant_q05.csv"),
  row.names = FALSE,
  quote = FALSE
)

main_table <- supp_table %>%
  filter(metric %in% main_metrics, contrast %in% main_contrasts)

write.csv(
  main_table,
  file.path(forest_dir, "Table_UCD_primary_forest_main_metrics.csv"),
  row.names = FALSE,
  quote = FALSE
)

settings <- data.frame(
  setting = c(
    "contrast_file",
    "main_metrics",
    "main_contrasts",
    "supplementary_contrasts",
    "q_threshold",
    "x_axis",
    "cluster1_color",
    "cluster2_color",
    "non_resilience_color",
    "primary_model_note",
    "legend_note"
  ),
  value = c(
    contrast_file,
    paste(main_metrics, collapse = ";"),
    paste(main_contrasts, collapse = ";"),
    paste(supplementary_contrasts, collapse = ";"),
    as.character(q_threshold),
    "standardized adjusted difference",
    group_colors[["cluster1"]],
    group_colors[["cluster2"]],
    group_colors[["non_resilience"]],
    "age at MRI, sex, education, APOE e4 carrier status, ADNI phase, manufacturer, and CEREBRUM_TCV as intracranial-size proxy for volumetric metrics",
    "Colors follow the projected CR subtype palette; contrast estimates are colored by the CR cluster compared with the non-CR reference group."
  ),
  stringsAsFactors = FALSE
)

write.csv(
  settings,
  file.path(forest_dir, "UCD_primary_forest_plot_settings.csv"),
  row.names = FALSE,
  quote = FALSE
)

sink(file.path(forest_dir, "sessionInfo_UCD_primary_forest_plot.txt"))
print(sessionInfo())
sink()

message("Done. Forest plots saved in: ", forest_dir)
