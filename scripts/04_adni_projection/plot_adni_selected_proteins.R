#!/usr/bin/env Rscript
# Script cleaned for public release. Edit /path/to/... inputs before running.

rm(list = ls())
options(stringsAsFactors = FALSE)

project_dir <- "/path/to/project"
setwd(project_dir)

input_file <- "out/ADNI/projected_rosmap_msbb_22gene_signature/tables/ADNI_projected_cluster_scores_with_clinical_ADpatho_CR.csv"

outdir <- "out/ADNI/projected_rosmap_msbb_22gene_signature/boxplots_selected_genes"
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

genes_to_plot <- c(
  "TMPRSS5",
  "SULT1A1",
  "GMPR",
  "EHD3",
  "ZWINT",
  "PIH1D2",
  "GLT8D2"
)

cluster_col <- "predicted_cluster_like"

cluster_levels <- c("cluster1", "cluster2")
cluster_colors <- c(
  "cluster1" = "#009E73",
  "cluster2" = "#E69F00"
)

test_method <- "wilcox"

plot_width <- 10
plot_height <- 6

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(ggpubr)
  library(readr)
})

message("[1] Loading ADNI projected table")

if (!file.exists(input_file)) {
  stop("Input file not found: ", input_file)
}

df <- read.csv(input_file, check.names = FALSE)

if (!(cluster_col %in% colnames(df))) {
  stop("Cluster column not found: ", cluster_col)
}

df[[cluster_col]] <- factor(as.character(df[[cluster_col]]), levels = cluster_levels)

genes_present <- genes_to_plot[genes_to_plot %in% colnames(df)]
genes_missing <- setdiff(genes_to_plot, genes_present)

if (length(genes_missing) > 0) {
  warning("These genes were not found and will be skipped: ", paste(genes_missing, collapse = ", "))
}
if (length(genes_present) == 0) {
  stop("None of the selected genes were found in input file.")
}

message("    Genes plotted: ", paste(genes_present, collapse = ", "))
message("    Cluster distribution:")
print(table(df[[cluster_col]], useNA = "ifany"))

df_long <- df %>%
  select(RID, all_of(cluster_col), all_of(genes_present)) %>%
  pivot_longer(
    cols = all_of(genes_present),
    names_to = "Gene",
    values_to = "z_expression"
  ) %>%
  filter(!is.na(.data[[cluster_col]]), !is.na(z_expression)) %>%
  mutate(
    Gene = factor(Gene, levels = genes_present),
    projected_cluster_like = .data[[cluster_col]]
  )

write.csv(
  df_long,
  file.path(outdir, "ADNI_selected_genes_expression_long_ADpatho_CR.csv"),
  row.names = FALSE,
  quote = FALSE
)

message("[2] Testing cluster1-like vs cluster2-like differences")

stat_rows <- list()

for (g in genes_present) {
  d <- df_long %>% filter(Gene == g)

  x1 <- d$z_expression[d$projected_cluster_like == "cluster1"]
  x2 <- d$z_expression[d$projected_cluster_like == "cluster2"]

  p_wilcox <- tryCatch(wilcox.test(x1, x2)$p.value, error = function(e) NA_real_)
  p_ttest <- tryCatch(t.test(x1, x2)$p.value, error = function(e) NA_real_)

  stat_rows[[length(stat_rows) + 1]] <- data.frame(
    Gene = g,
    n_cluster1 = sum(!is.na(x1)),
    n_cluster2 = sum(!is.na(x2)),
    mean_cluster1 = mean(x1, na.rm = TRUE),
    mean_cluster2 = mean(x2, na.rm = TRUE),
    median_cluster1 = median(x1, na.rm = TRUE),
    median_cluster2 = median(x2, na.rm = TRUE),
    mean_cluster1_minus_cluster2 = mean(x1, na.rm = TRUE) - mean(x2, na.rm = TRUE),
    median_cluster1_minus_cluster2 = median(x1, na.rm = TRUE) - median(x2, na.rm = TRUE),
    p_wilcox = p_wilcox,
    p_ttest = p_ttest,
    stringsAsFactors = FALSE
  )
}

stat_df <- bind_rows(stat_rows) %>%
  mutate(
    FDR_wilcox = p.adjust(p_wilcox, method = "BH"),
    FDR_ttest = p.adjust(p_ttest, method = "BH")
  )

write.csv(
  stat_df,
  file.path(outdir, "ADNI_selected_genes_cluster1_vs_cluster2_stats_ADpatho_CR.csv"),
  row.names = FALSE,
  quote = FALSE
)

print(stat_df)

message("[3] Drawing faceted boxplot")

compare_method <- ifelse(test_method == "t.test", "t.test", "wilcox.test")

p <- ggplot(df_long, aes(x = projected_cluster_like, y = z_expression, fill = projected_cluster_like)) +
  geom_boxplot(outlier.shape = NA, alpha = 0.85, width = 0.65) +
  geom_jitter(width = 0.16, alpha = 0.55, size = 1.2, color = "black") +
  facet_wrap(~ Gene, scales = "free_y", ncol = min(4, length(genes_present))) +
  stat_compare_means(
    comparisons = list(c("cluster1", "cluster2")),
    method = compare_method,
    label = "p.signif",
    hide.ns = FALSE,
    size = 5
  ) +
  scale_fill_manual(values = cluster_colors, na.value = "gray80") +
  theme_bw(base_size = 13) +
  labs(
    title = "ADNI ADpatho+CR protein expression by projected cluster-like group",
    subtitle = paste0("Test: ", compare_method, "; expression values are ADNI protein z-scores"),
    x = "Projected cluster-like group",
    y = "Protein abundance z-score"
  ) +
  theme(
    legend.position = "right",
    axis.text.x = element_text(angle = 45, hjust = 1),
    strip.text = element_text(face = "bold", size = 12),
    plot.title = element_text(face = "bold")
  )

ggsave(
  file.path(outdir, "ADNI_selected_genes_boxplot_by_projected_cluster_ADpatho_CR.svg"),
  p,
  width = plot_width,
  height = plot_height,
  units = "in"
)
ggsave(
  file.path(outdir, "ADNI_selected_genes_boxplot_by_projected_cluster_ADpatho_CR.pdf"),
  p,
  width = plot_width,
  height = plot_height,
  units = "in"
)
ggsave(
  file.path(outdir, "ADNI_selected_genes_boxplot_by_projected_cluster_ADpatho_CR.png"),
  p,
  width = plot_width,
  height = plot_height,
  units = "in",
  dpi = 300
)

message("[4] Drawing per-gene boxplots")

for (g in genes_present) {
  d <- df_long %>% filter(Gene == g)

  p_g <- ggplot(d, aes(x = projected_cluster_like, y = z_expression, fill = projected_cluster_like)) +
    geom_boxplot(outlier.shape = NA, alpha = 0.85, width = 0.65) +
    geom_jitter(width = 0.16, alpha = 0.6, size = 1.4, color = "black") +
    stat_compare_means(
      comparisons = list(c("cluster1", "cluster2")),
      method = compare_method,
      label = "p.format",
      hide.ns = FALSE,
      size = 5
    ) +
    scale_fill_manual(values = cluster_colors, na.value = "gray80") +
    theme_bw(base_size = 14) +
    labs(
      title = paste0(g, " in ADNI ADpatho+CR samples"),
      subtitle = "Grouped by projected ROSMAP/MSBB cluster-like label",
      x = "Projected cluster-like group",
      y = "Protein abundance z-score"
    ) +
    theme(
      legend.position = "none",
      axis.text.x = element_text(angle = 45, hjust = 1),
      plot.title = element_text(face = "bold")
    )

  ggsave(
    file.path(outdir, paste0("ADNI_", g, "_boxplot_by_projected_cluster_ADpatho_CR.svg")),
    p_g,
    width = 4.5,
    height = 4.5,
    units = "in"
  )
  ggsave(
    file.path(outdir, paste0("ADNI_", g, "_boxplot_by_projected_cluster_ADpatho_CR.png")),
    p_g,
    width = 4.5,
    height = 4.5,
    units = "in",
    dpi = 300
  )
}

sink(file.path(outdir, "sessionInfo_selected_genes_boxplot.txt"))
print(sessionInfo())
sink()

message("Done. Results saved in: ", outdir)
