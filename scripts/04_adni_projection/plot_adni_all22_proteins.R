#!/usr/bin/env Rscript
# Script cleaned for public release. Edit /path/to/... inputs before running.

rm(list = ls())
options(stringsAsFactors = FALSE)

project_dir <- "/path/to/project"
setwd(project_dir)

adni_input_file <- "out/ADNI/projected_rosmap_msbb_22gene_signature/tables/ADNI_projected_cluster_scores_with_clinical_ADpatho_CR.csv"
marker_table_file <- "out/clustering/adni_projection_signature/tables/ROSMAP_MSBB_22gene_cluster_marker_table.csv"

outdir <- "out/ADNI/projected_rosmap_msbb_22gene_signature/boxplots_all22_genes"
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

cluster_col <- "predicted_cluster_like"
cluster_levels <- c("cluster1", "cluster2")

cluster_colors <- c(
  "cluster1" = "#009E73",
  "cluster2" = "#E69F00"
)

marker_direction_colors <- c(
  "cluster1-high" = "#009E73",
  "cluster2-high" = "#E69F00",
  "other" = "gray70"
)

test_method <- "wilcox"

reference_cluster_for_direction <- "cluster1"

facet_ncol <- 5
facet_plot_width <- 14
facet_plot_height <- 12

single_plot_width <- 4.5
single_plot_height <- 4.5

make_individual_plots <- TRUE

show_p_significance <- TRUE

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(ggpubr)
  library(readr)
})

safe_factor <- function(x, levels = NULL) {
  if (is.null(levels)) return(as.factor(as.character(x)))
  factor(as.character(x), levels = levels)
}

p_to_stars <- function(p) {
  if (is.na(p)) return("")
  if (p > 0.05) return("ns")
  if (p > 0.01) return("*")
  if (p > 0.001) return("**")
  if (p > 0.0001) return("***")
  return("****")
}

message("[1] Loading ADNI projected table and marker table")

if (!file.exists(adni_input_file)) {
  stop("ADNI input file not found: ", adni_input_file)
}
if (!file.exists(marker_table_file)) {
  stop("Marker table file not found: ", marker_table_file)
}

df <- read.csv(adni_input_file, check.names = FALSE)
marker <- read.csv(marker_table_file, check.names = FALSE)

if (!(cluster_col %in% colnames(df))) {
  stop("Cluster column not found in ADNI table: ", cluster_col)
}

required_marker_cols <- c("cluster", "Symbol", "delta_z_cluster_vs_others", "abs_delta_z")
missing_marker_cols <- setdiff(required_marker_cols, colnames(marker))
if (length(missing_marker_cols) > 0) {
  stop("Marker table is missing columns: ", paste(missing_marker_cols, collapse = ", "))
}

df[[cluster_col]] <- safe_factor(df[[cluster_col]], cluster_levels)

message("[2] Detecting all shared proteins and ordering by marker direction")

marker_ref <- marker %>%
  filter(cluster == reference_cluster_for_direction) %>%
  group_by(Symbol) %>%
  arrange(desc(abs_delta_z), .by_group = TRUE) %>%
  slice(1) %>%
  ungroup() %>%
  mutate(
    marker_direction = ifelse(delta_z_cluster_vs_others >= 0, "cluster1-high", "cluster2-high"),
    direction_rank = ifelse(marker_direction == "cluster1-high", 1, 2),
    sort_score = ifelse(
      marker_direction == "cluster1-high",
      -delta_z_cluster_vs_others,
      delta_z_cluster_vs_others
    )
  )

genes_all <- marker_ref %>%
  filter(Symbol %in% colnames(df)) %>%
  arrange(direction_rank, sort_score) %>%
  pull(Symbol) %>%
  unique()

genes_missing_in_adni <- setdiff(unique(marker_ref$Symbol), colnames(df))

if (length(genes_missing_in_adni) > 0) {
  warning("These marker-table genes were not found in the ADNI projected table and will be skipped: ",
          paste(genes_missing_in_adni, collapse = ", "))
}

if (length(genes_all) == 0) {
  stop("No marker-table genes were found as columns in the ADNI projected table.")
}

gene_annotation <- marker_ref %>%
  filter(Symbol %in% genes_all) %>%
  select(Symbol, marker_direction, delta_z_cluster_vs_others, abs_delta_z) %>%
  distinct(Symbol, .keep_all = TRUE) %>%
  arrange(match(Symbol, genes_all))

write.csv(
  gene_annotation,
  file.path(outdir, "ADNI_all22_genes_marker_direction_order.csv"),
  row.names = FALSE,
  quote = FALSE
)

message("    Genes plotted: ", length(genes_all))
message("    Gene order: ", paste(genes_all, collapse = ", "))
message("    Marker direction counts:")
print(table(gene_annotation$marker_direction, useNA = "ifany"))
message("    Cluster distribution:")
print(table(df[[cluster_col]], useNA = "ifany"))

message("[3] Creating long-format expression table")

df_long <- df %>%
  select(RID, all_of(cluster_col), all_of(genes_all)) %>%
  pivot_longer(
    cols = all_of(genes_all),
    names_to = "Gene",
    values_to = "z_expression"
  ) %>%
  left_join(gene_annotation, by = c("Gene" = "Symbol")) %>%
  filter(!is.na(.data[[cluster_col]]), !is.na(z_expression)) %>%
  mutate(
    Gene = factor(Gene, levels = genes_all),
    marker_direction = factor(marker_direction, levels = c("cluster1-high", "cluster2-high", "other")),
    projected_cluster_like = .data[[cluster_col]]
  )

write.csv(
  df_long,
  file.path(outdir, "ADNI_all22_genes_expression_long_ADpatho_CR.csv"),
  row.names = FALSE,
  quote = FALSE
)

message("[4] Testing cluster1-like vs cluster2-like differences for all genes")

stat_rows <- list()

for (g in genes_all) {
  d <- df_long %>% filter(Gene == g)

  x1 <- d$z_expression[d$projected_cluster_like == "cluster1"]
  x2 <- d$z_expression[d$projected_cluster_like == "cluster2"]

  p_wilcox <- tryCatch(wilcox.test(x1, x2)$p.value, error = function(e) NA_real_)
  p_ttest <- tryCatch(t.test(x1, x2)$p.value, error = function(e) NA_real_)

  md <- gene_annotation %>% filter(Symbol == g) %>% slice(1)

  stat_rows[[length(stat_rows) + 1]] <- data.frame(
    Gene = g,
    marker_direction = md$marker_direction,
    delta_z_ROSMAP_MSBB_cluster1_vs_others = md$delta_z_cluster_vs_others,
    abs_delta_z_ROSMAP_MSBB = md$abs_delta_z,
    n_cluster1 = sum(!is.na(x1)),
    n_cluster2 = sum(!is.na(x2)),
    mean_cluster1 = mean(x1, na.rm = TRUE),
    mean_cluster2 = mean(x2, na.rm = TRUE),
    median_cluster1 = median(x1, na.rm = TRUE),
    median_cluster2 = median(x2, na.rm = TRUE),
    mean_cluster1_minus_cluster2 = mean(x1, na.rm = TRUE) - mean(x2, na.rm = TRUE),
    median_cluster1_minus_cluster2 = median(x1, na.rm = TRUE) - median(x2, na.rm = TRUE),
    expected_direction = ifelse(md$marker_direction == "cluster1-high", 1,
                                ifelse(md$marker_direction == "cluster2-high", -1, NA_real_)),
    observed_direction = sign(mean(x1, na.rm = TRUE) - mean(x2, na.rm = TRUE)),
    p_wilcox = p_wilcox,
    p_ttest = p_ttest,
    stringsAsFactors = FALSE
  )
}

stat_df <- bind_rows(stat_rows) %>%
  mutate(
    direction_concordant = expected_direction == observed_direction,
    FDR_wilcox = p.adjust(p_wilcox, method = "BH"),
    FDR_ttest = p.adjust(p_ttest, method = "BH"),
    p_wilcox_stars = vapply(p_wilcox, p_to_stars, character(1)),
    p_ttest_stars = vapply(p_ttest, p_to_stars, character(1))
  ) %>%
  arrange(match(Gene, genes_all))

write.csv(
  stat_df,
  file.path(outdir, "ADNI_all22_genes_cluster1_vs_cluster2_stats_ADpatho_CR.csv"),
  row.names = FALSE,
  quote = FALSE
)

message("    Direction concordance:")
print(table(stat_df$direction_concordant, useNA = "ifany"))

print(stat_df)

sig_col <- ifelse(test_method == "t.test", "p_ttest_stars", "p_wilcox_stars")

sig_label_df <- df_long %>%
  group_by(Gene) %>%
  summarise(
    y_min = min(z_expression, na.rm = TRUE),
    y_max = max(z_expression, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    y_range = y_max - y_min,
    y_range = ifelse(is.na(y_range) | y_range == 0, 1, y_range),
    y_position = y_max + 0.10 * y_range
  ) %>%
  left_join(
    stat_df %>%
      mutate(
        Gene = factor(Gene, levels = genes_all),
        p_label = .data[[sig_col]]
      ) %>%
      select(Gene, p_label),
    by = "Gene"
  ) %>%
  mutate(
    Gene = factor(Gene, levels = genes_all),
    x_position = 1.5
  )

message("[5] Drawing faceted boxplot for all 22 genes")

compare_method <- ifelse(test_method == "t.test", "t.test", "wilcox.test")

p <- ggplot(df_long, aes(x = projected_cluster_like, y = z_expression, fill = projected_cluster_like)) +
  geom_boxplot(outlier.shape = NA, alpha = 0.85, width = 0.65) +
  facet_wrap(~ Gene, scales = "free_y", ncol = facet_ncol) +
  scale_fill_manual(values = cluster_colors, na.value = "gray80") +
  theme_bw(base_size = 24) +
  labs(
    title = "ADNI ADpatho+CR protein expression by projected cluster-like group",
    subtitle = paste0("All shared proteins; test: ", compare_method, "; values are ADNI protein z-scores"),
    x = "Projected cluster-like group",
    y = "Protein abundance z-score"
  ) +
  theme(
    legend.position = "right",
    axis.text.x = element_text(angle = 45, hjust = 1, size = 20),
    axis.text.y = element_text(size = 20),
    axis.title = element_text(size = 24),
    strip.text = element_text(face = "plain", size = 20),
    plot.title = element_text(face = "bold", size = 28),
    plot.subtitle = element_text(size = 22),
    legend.text = element_text(size = 20),
    legend.title = element_text(size = 22)
  )

if (show_p_significance) {
  p <- p +
    geom_text(
      data = sig_label_df,
      aes(x = x_position, y = y_position, label = p_label),
      inherit.aes = FALSE,
      size = 5,
      fontface = "plain",
      na.rm = TRUE
    ) +
    scale_y_continuous(expand = expansion(mult = c(0.05, 0.20))) +
    coord_cartesian(clip = "off")
}

ggsave(
  file.path(outdir, "ADNI_all22_genes_boxplot_by_projected_cluster_ADpatho_CR.svg"),
  p,
  width = facet_plot_width,
  height = facet_plot_height,
  units = "in"
)
ggsave(
  file.path(outdir, "ADNI_all22_genes_boxplot_by_projected_cluster_ADpatho_CR.pdf"),
  p,
  width = facet_plot_width,
  height = facet_plot_height,
  units = "in"
)
ggsave(
  file.path(outdir, "ADNI_all22_genes_boxplot_by_projected_cluster_ADpatho_CR.png"),
  p,
  width = facet_plot_width,
  height = facet_plot_height,
  units = "in",
  dpi = 300
)

message("[6] Drawing marker-direction split boxplot")

p_dir <- ggplot(df_long, aes(x = projected_cluster_like, y = z_expression, fill = projected_cluster_like)) +
  geom_boxplot(outlier.shape = NA, alpha = 0.85, width = 0.65) +
  facet_grid(marker_direction ~ Gene, scales = "free_y", space = "free_x") +
  scale_fill_manual(values = cluster_colors, na.value = "gray80") +
  theme_bw(base_size = 22) +
  labs(
    title = "ADNI ADpatho+CR protein expression by projected cluster-like group",
    subtitle = "Rows split by ROSMAP/MSBB marker direction",
    x = "Projected cluster-like group",
    y = "Protein abundance z-score"
  ) +
  theme(
    legend.position = "right",
    axis.text.x = element_text(angle = 90, hjust = 1, size = 18),
    axis.text.y = element_text(size = 18),
    axis.title = element_text(size = 22),
    strip.text.x = element_text(face = "plain", size = 16),
    strip.text.y = element_text(face = "plain", size = 18),
    plot.title = element_text(face = "bold", size = 26),
    plot.subtitle = element_text(size = 20),
    legend.text = element_text(size = 18),
    legend.title = element_text(size = 20)
  )

ggsave(
  file.path(outdir, "ADNI_all22_genes_boxplot_by_projected_cluster_split_by_marker_direction_ADpatho_CR.svg"),
  p_dir,
  width = 18,
  height = 7,
  units = "in"
)
ggsave(
  file.path(outdir, "ADNI_all22_genes_boxplot_by_projected_cluster_split_by_marker_direction_ADpatho_CR.png"),
  p_dir,
  width = 18,
  height = 7,
  units = "in",
  dpi = 300
)

if (make_individual_plots) {
  message("[7] Drawing individual per-gene boxplots")

  indiv_dir <- file.path(outdir, "individual_gene_boxplots")
  dir.create(indiv_dir, recursive = TRUE, showWarnings = FALSE)

  for (g in genes_all) {
    d <- df_long %>% filter(Gene == g)
    st <- stat_df %>% filter(Gene == g) %>% slice(1)

    subtitle_txt <- paste0(
      "ROSMAP/MSBB: ", st$marker_direction,
      "; ADNI mean difference cluster1-cluster2 = ",
      sprintf("%.3f", st$mean_cluster1_minus_cluster2),
      "; Wilcoxon p = ",
      signif(st$p_wilcox, 3)
    )

    y_min_g <- min(d$z_expression, na.rm = TRUE)
    y_max_g <- max(d$z_expression, na.rm = TRUE)
    y_range_g <- y_max_g - y_min_g
    if (is.na(y_range_g) || y_range_g == 0) y_range_g <- 1
    y_position_g <- y_max_g + 0.10 * y_range_g
    p_label_g <- st[[sig_col]]

    p_g <- ggplot(d, aes(x = projected_cluster_like, y = z_expression, fill = projected_cluster_like)) +
      geom_boxplot(outlier.shape = NA, alpha = 0.85, width = 0.65) +
      geom_text(
        data = data.frame(x_position = 1.5, y_position = y_position_g, p_label = p_label_g),
        aes(x = x_position, y = y_position, label = p_label),
        inherit.aes = FALSE,
        size = 5,
        fontface = "plain",
        na.rm = TRUE
      ) +
      scale_y_continuous(expand = expansion(mult = c(0.05, 0.20))) +
      coord_cartesian(clip = "off") +
      scale_fill_manual(values = cluster_colors, na.value = "gray80") +
      theme_bw(base_size = 14) +
      labs(
        title = paste0(g, " in ADNI ADpatho+CR samples"),
        subtitle = subtitle_txt,
        x = "Projected cluster-like group",
        y = "Protein abundance z-score"
      ) +
      theme(
        legend.position = "none",
        axis.text.x = element_text(angle = 90, hjust = 1),
        plot.title = element_text(face = "plain")
      )

    ggsave(
      file.path(indiv_dir, paste0("ADNI_", g, "_boxplot_by_projected_cluster_ADpatho_CR.svg")),
      p_g,
      width = single_plot_width,
      height = single_plot_height,
      units = "in"
    )
    ggsave(
      file.path(indiv_dir, paste0("ADNI_", g, "_boxplot_by_projected_cluster_ADpatho_CR.png")),
      p_g,
      width = single_plot_width,
      height = single_plot_height,
      units = "in",
      dpi = 300
    )
  }
}

sink(file.path(outdir, "sessionInfo_all22_genes_boxplot.txt"))
print(sessionInfo())
sink()

message("Done. Results saved in: ", outdir)
