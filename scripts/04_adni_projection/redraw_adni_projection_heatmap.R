#!/usr/bin/env Rscript
# Script cleaned for public release. Edit /path/to/... inputs before running.

rm(list = ls())
options(stringsAsFactors = FALSE)

project_dir <- "/path/to/project"
setwd(project_dir)

adni_projected_file <- "out/ADNI/projected_rosmap_msbb_22gene_signature/tables/ADNI_projected_cluster_scores_with_clinical_ADpatho_CR.csv"
marker_table_file <- "out/clustering/adni_projection_signature/tables/ROSMAP_MSBB_22gene_cluster_marker_table.csv"

outdir <- "out/ADNI/projected_rosmap_msbb_22gene_signature/heatmaps_marker_ordered"
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

primary_projection_method <- "centroid_correlation"
cluster_label_col <- "predicted_cluster_like"
reference_cluster_for_direction <- "cluster1"

sample_order_method <- "predicted_cluster_then_own_score"

show_column_names <- FALSE
cluster_rows_heatmap <- FALSE
cluster_cols_heatmap <- FALSE

heatmap_width <- 12
heatmap_height <- 8

group_mean_heatmap_width <- 4.8
group_mean_heatmap_height <- 7.2

zlim <- 2.5

cluster_palette <- c(
  "cluster1" = "#009E73",
  "cluster2" = "#E69F00",
  "cluster3" = "#7570b3",
  "cluster4" = "#e7298a",
  "cluster5" = "#66a61e"
)

marker_direction_colors <- c(
  "cluster1-high" = "#009E73",
  "cluster2-high" = "#E69F00",
  "other" = "gray70"
)

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(tibble)
  library(pheatmap)
  library(RColorBrewer)
  library(grid)
})

clip_matrix <- function(mat, lim) {
  mat <- as.matrix(mat)
  mat[mat > lim] <- lim
  mat[mat < -lim] <- -lim
  mat
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

safe_factor <- function(x) {
  as.factor(as.character(x))
}

message("[1] Loading projected ADNI table and marker table")

if (!file.exists(adni_projected_file)) {
  stop("ADNI projected file not found: ", adni_projected_file)
}
if (!file.exists(marker_table_file)) {
  stop("Marker table file not found: ", marker_table_file)
}

adni <- read.csv(adni_projected_file, check.names = FALSE)
marker <- read.csv(marker_table_file, check.names = FALSE)

required_marker_cols <- c("cluster", "Symbol", "delta_z_cluster_vs_others", "abs_delta_z")
missing_marker_cols <- setdiff(required_marker_cols, colnames(marker))
if (length(missing_marker_cols) > 0) {
  stop("Marker table is missing columns: ", paste(missing_marker_cols, collapse = ", "))
}

if (!(cluster_label_col %in% colnames(adni))) {
  stop("ADNI table does not contain cluster_label_col: ", cluster_label_col)
}
if (!("RID" %in% colnames(adni))) {
  stop("ADNI projected table must contain RID.")
}

adni[[cluster_label_col]] <- safe_factor(adni[[cluster_label_col]])

message("[2] Defining row order by ROSMAP/MSBB marker direction")

available_symbols <- intersect(unique(marker$Symbol), colnames(adni))
if (length(available_symbols) < 2) {
  stop("Fewer than 2 marker symbols are present in ADNI projected table.")
}

marker_ref <- marker %>%
  filter(cluster == reference_cluster_for_direction, Symbol %in% available_symbols) %>%
  group_by(Symbol) %>%
  arrange(desc(abs_delta_z), .by_group = TRUE) %>%
  slice(1) %>%
  ungroup()

if (nrow(marker_ref) == 0) {
  stop("No marker rows found for reference_cluster_for_direction: ", reference_cluster_for_direction)
}

marker_ref <- marker_ref %>%
  mutate(
    marker_direction = ifelse(delta_z_cluster_vs_others >= 0, "cluster1-high", "cluster2-high"),
    direction_rank = ifelse(marker_direction == "cluster1-high", 1, 2),
    sort_score = ifelse(marker_direction == "cluster1-high",
                        -delta_z_cluster_vs_others,
                        delta_z_cluster_vs_others)
  )

row_order <- marker_ref %>%
  arrange(direction_rank, sort_score) %>%
  pull(Symbol) %>%
  unique()

row_order <- row_order[row_order %in% colnames(adni)]

row_anno <- marker_ref %>%
  select(Symbol, marker_direction, delta_z_cluster_vs_others, abs_delta_z) %>%
  distinct(Symbol, .keep_all = TRUE) %>%
  filter(Symbol %in% row_order) %>%
  arrange(match(Symbol, row_order))

write.csv(row_anno,
          file.path(outdir, "ADNI_22protein_marker_direction_row_order.csv"),
          row.names = FALSE, quote = FALSE)

message("    Proteins in heatmap: ", length(row_order))
message("    Marker direction counts:")
print(table(row_anno$marker_direction, useNA = "ifany"))

message("[3] Defining sample order")

clusters <- levels(droplevels(adni[[cluster_label_col]]))

if (sample_order_method == "predicted_cluster_then_own_score") {
  adni$order_score <- NA_real_

  for (cl in clusters) {
    score_col <- paste0("score_", cl, "_", primary_projection_method)
    if (score_col %in% colnames(adni)) {
      idx <- adni[[cluster_label_col]] == cl
      adni$order_score[idx] <- adni[[score_col]][idx]
    }
  }

  if ("max_primary_score" %in% colnames(adni)) {
    adni$order_score[is.na(adni$order_score)] <- adni$max_primary_score[is.na(adni$order_score)]
  }

  sample_order <- adni %>%
    mutate(predicted_cluster_order = factor(.data[[cluster_label_col]], levels = clusters)) %>%
    arrange(predicted_cluster_order, desc(order_score)) %>%
    pull(RID)

} else if (sample_order_method == "predicted_cluster_then_max_score") {
  if (!("max_primary_score" %in% colnames(adni))) {
    stop("max_primary_score is required for sample_order_method = predicted_cluster_then_max_score.")
  }

  sample_order <- adni %>%
    mutate(predicted_cluster_order = factor(.data[[cluster_label_col]], levels = clusters)) %>%
    arrange(predicted_cluster_order, desc(max_primary_score)) %>%
    pull(RID)

} else if (sample_order_method == "current_order") {
  sample_order <- adni$RID

} else {
  stop("Unknown sample_order_method: ", sample_order_method)
}

sample_order <- as.character(sample_order)

message("[4] Building heatmap matrix")

expr_mat <- adni %>%
  select(RID, all_of(row_order)) %>%
  mutate(RID = as.character(RID)) %>%
  column_to_rownames("RID") %>%
  as.matrix()

storage.mode(expr_mat) <- "numeric"
expr_mat <- expr_mat[sample_order, row_order, drop = FALSE]

heat_mat <- t(expr_mat)
heat_mat_clipped <- clip_matrix(heat_mat, zlim)

annotation_row <- row_anno %>%
  select(Symbol, marker_direction) %>%
  column_to_rownames("Symbol")
annotation_row <- annotation_row[rownames(heat_mat_clipped), , drop = FALSE]

ann_cols_wanted <- c(
  cluster_label_col,
  "max_primary_score",
  "score_margin",
  "MMSCORE",
  "resilience_score",
  "ABETA42",
  "TAU",
  "PTAU",
  "DIAGNOSIS",
  "ADpatho",
  "resilience",
  "sex",
  "age_at_visit",
  "PTEDUCAT"
)

ann_cols_present <- ann_cols_wanted[ann_cols_wanted %in% colnames(adni)]

annotation_col <- adni %>%
  mutate(RID = as.character(RID)) %>%
  filter(RID %in% colnames(heat_mat_clipped)) %>%
  arrange(match(RID, colnames(heat_mat_clipped))) %>%
  select(RID, all_of(ann_cols_present)) %>%
  column_to_rownames("RID")

for (cc in intersect(c(cluster_label_col, "DIAGNOSIS", "ADpatho", "resilience", "sex"), colnames(annotation_col))) {
  annotation_col[[cc]] <- safe_factor(annotation_col[[cc]])
}

ann_colors <- list(
  marker_direction = marker_direction_colors,
  predicted_cluster_like = cluster_palette[intersect(names(cluster_palette), levels(safe_factor(adni[[cluster_label_col]])))],
  DIAGNOSIS = c(CN = "#4daf4a", MCI = "#ff7f00", Dementia = "#e41a1c"),
  ADpatho = c("No AD" = "#999999", "AD" = "#377eb8"),
  resilience = c(Low = "#999999", High = "#377eb8"),
  sex = c(Male = "#377eb8", Female = "#e78ac3")
)

if (cluster_label_col != "predicted_cluster_like") {
  ann_colors[[cluster_label_col]] <- cluster_palette[intersect(names(cluster_palette), levels(safe_factor(adni[[cluster_label_col]])))]
}

message("[5] Drawing marker-direction ordered heatmap")

cols <- colorRampPalette(rev(brewer.pal(n = 11, name = "RdBu")))(100)
breaks <- seq(-zlim, zlim, length.out = 101)

ph <- pheatmap(
  heat_mat_clipped,
  annotation_col = annotation_col,
  annotation_row = annotation_row,
  annotation_colors = ann_colors,
  cluster_rows = cluster_rows_heatmap,
  cluster_cols = cluster_cols_heatmap,
  show_colnames = show_column_names,
  main = paste0(
    "ADNI projection of ROSMAP/MSBB 22-gene CR-cluster signatures\n",
    "rows ordered by ROSMAP/MSBB marker direction"
  ),
  color = cols,
  breaks = breaks,
  border_color = NA,
  fontsize = 9,
  fontsize_row = 8,
  fontsize_col = 5,
  silent = TRUE
)

save_pheatmap_multi(
  ph,
  file.path(outdir, "ADNI_22protein_projection_heatmap_ADpatho_CR_marker_direction_ordered"),
  width = heatmap_width,
  height = heatmap_height
)

message("[6] Drawing group-mean heatmap")

group_col <- cluster_label_col

group_mean <- adni %>%
  select(all_of(group_col), all_of(row_order)) %>%
  pivot_longer(cols = all_of(row_order), names_to = "Symbol", values_to = "z") %>%
  group_by(.data[[group_col]], Symbol) %>%
  summarise(mean_z = mean(z, na.rm = TRUE), .groups = "drop") %>%
  rename(cluster_like = all_of(group_col)) %>%
  pivot_wider(names_from = cluster_like, values_from = mean_z)

group_mean_mat <- group_mean %>%
  filter(Symbol %in% row_order) %>%
  arrange(match(Symbol, row_order)) %>%
  column_to_rownames("Symbol") %>%
  as.matrix()

storage.mode(group_mean_mat) <- "numeric"
group_mean_mat <- group_mean_mat[row_order, , drop = FALSE]

if (all(c("cluster1", "cluster2") %in% colnames(group_mean_mat))) {
  diff_col <- group_mean_mat[, "cluster1"] - group_mean_mat[, "cluster2"]
  group_mean_mat2 <- cbind(
    group_mean_mat,
    "cluster1_minus_cluster2" = diff_col
  )
  group_mean_mat2 <- group_mean_mat2[row_order, , drop = FALSE]
} else {
  group_mean_mat2 <- group_mean_mat
}

group_mean_mat2_clipped <- clip_matrix(group_mean_mat2, zlim)
annotation_row2 <- annotation_row[rownames(group_mean_mat2_clipped), , drop = FALSE]

ph_mean <- pheatmap(
  group_mean_mat2_clipped,
  annotation_row = annotation_row2,
  annotation_colors = ann_colors,
  cluster_rows = FALSE,
  cluster_cols = FALSE,
  main = paste0(
    "ADNI mean z-score by projected cluster-like group\n",
    "rows ordered by ROSMAP/MSBB marker direction"
  ),
  color = cols,
  breaks = breaks,
  border_color = NA,
  fontsize = 9,
  fontsize_row = 8,
  fontsize_col = 9,
  angle_col = 45,
  silent = TRUE
)

save_pheatmap_multi(
  ph_mean,
  file.path(outdir, "ADNI_22protein_projection_group_mean_heatmap_ADpatho_CR_marker_direction_ordered"),
  width = group_mean_heatmap_width,
  height = group_mean_heatmap_height
)

write.csv(group_mean_mat2,
          file.path(outdir, "ADNI_22protein_projection_group_mean_matrix_marker_direction_ordered.csv"),
          quote = FALSE)

message("[7] Computing ADNI direction concordance with ROSMAP/MSBB marker direction")

if (all(c("cluster1", "cluster2") %in% colnames(group_mean_mat))) {
  concordance <- row_anno %>%
    filter(Symbol %in% rownames(group_mean_mat)) %>%
    mutate(
      adni_cluster1_mean_z = group_mean_mat[Symbol, "cluster1"],
      adni_cluster2_mean_z = group_mean_mat[Symbol, "cluster2"],
      adni_cluster1_minus_cluster2 = adni_cluster1_mean_z - adni_cluster2_mean_z,
      expected_direction = ifelse(marker_direction == "cluster1-high", 1,
                                  ifelse(marker_direction == "cluster2-high", -1, NA_real_)),
      observed_direction = sign(adni_cluster1_minus_cluster2),
      direction_concordant = expected_direction == observed_direction
    ) %>%
    arrange(match(Symbol, row_order))

  write.csv(concordance,
            file.path(outdir, "ADNI_22protein_direction_concordance_with_ROSMAP_MSBB_marker_direction.csv"),
            row.names = FALSE, quote = FALSE)

  message("    Direction concordance:")
  print(table(concordance$direction_concordant, useNA = "ifany"))
}

sink(file.path(outdir, "sessionInfo_redraw_marker_direction_ordered_heatmap.txt"))
print(sessionInfo())
sink()

message("Done. Marker-direction ordered heatmaps saved in: ", outdir)
