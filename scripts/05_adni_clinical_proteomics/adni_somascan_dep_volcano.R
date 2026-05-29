#!/usr/bin/env Rscript
# Script cleaned for public release. Edit /path/to/... inputs before running.

rm(list = ls())
options(stringsAsFactors = FALSE)

project_dir <- "/path/to/project"
setwd(project_dir)

adni_projection_file <- "out/ADNI/projected_rosmap_msbb_22gene_signature/tables/ADNI_projected_cluster_scores_with_clinical_ADpatho.csv"
adni_projection_all_file <- "out/ADNI/projected_rosmap_msbb_22gene_signature/tables/ADNI_projected_cluster_scores_with_clinical_all.csv"

somascan_matrix_file <- "/path/to/ADNI/CruchagaLab_CSF_SOMAscan7k_Protein_matrix_postQC_20230620.csv"
somascan_info_file <- "/path/to/ADNI/ADNI_Cruchaga_lab_CSF_SOMAscan7k_analyte_information_20_06_2023.csv"
protein_visit <- "bl"

outdir <- "out/ADNI/projected_rosmap_msbb_22gene_signature/DE_volcano_current_groups"
table_dir <- file.path(outdir, "tables")
plot_dir <- file.path(outdir, "plots")
dir.create(table_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(plot_dir, recursive = TRUE, showWarnings = FALSE)

group_levels <- c("cluster1", "cluster2", "non_resilience")

collapse_analytes_by_gene <- TRUE

impute_missing_by_min <- TRUE

offset <- 1e-6
fdr_cutoff <- 0.05
lfc_cutoff <- 0.1
top_n_labels <- 10

volcano_base_size <- 14
label_size <- 3.5
point_size <- 1.2
single_volcano_width <- 5
single_volcano_height <- 4.5
combined_volcano_width <- 9
combined_volcano_height <- 5
venn_width <- 7
venn_height <- 5

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(data.table)
  library(tidyr)
  library(patchwork)
  library(ggrepel)
  library(ggVennDiagram)
  library(stringr)
})

clean_numeric <- function(x) {
  x <- as.character(x)
  x[x %in% c("", "NA", "NaN", "NULL", "null")] <- NA
  suppressWarnings(as.numeric(x))
}

run_ttest <- function(mat, grp, g1, g2, offset = 1e-6) {
  idx1 <- which(grp == g1)
  idx2 <- which(grp == g2)
  n1 <- length(idx1)
  n2 <- length(idx2)

  if (n1 < 2 || n2 < 2) {
    stop(sprintf("Group size too small: %s=%d, %s=%d", g1, n1, g2, n2))
  }

  mean1 <- rowMeans(mat[, idx1, drop = FALSE], na.rm = TRUE)
  mean2 <- rowMeans(mat[, idx2, drop = FALSE], na.rm = TRUE)
  log2FC <- log2(mean1 + offset) - log2(mean2 + offset)

  pval <- apply(mat, 1, function(x) {
    x1 <- as.numeric(x[idx1])
    x2 <- as.numeric(x[idx2])
    if (sum(!is.na(x1)) < 2 || sum(!is.na(x2)) < 2) return(NA_real_)
    if (sd(x1, na.rm = TRUE) == 0 || sd(x2, na.rm = TRUE) == 0) return(NA_real_)
    tryCatch(t.test(x1, x2, var.equal = FALSE)$p.value, error = function(e) NA_real_)
  })

  padj <- p.adjust(pval, method = "BH")

  out <- data.frame(
    gene = rownames(mat),
    mean_g1 = mean1,
    mean_g2 = mean2,
    log2FC = log2FC,
    pval = pval,
    padj = padj,
    n_g1 = n1,
    n_g2 = n2,
    comp = paste0(g1, "_vs_", g2),
    stringsAsFactors = FALSE
  )
  rownames(out) <- out$gene
  out
}

volcano_plot <- function(df, title = NULL,
                         fdr = 0.05,
                         lfc = 0,
                         top_n = 10,
                         genes_highlight = NULL) {
  df_plot <- df[!is.na(df$padj), ]
  df_plot$neglog10FDR <- -log10(df_plot$padj)

  df_plot$signif <- "NS"
  df_plot$signif[df_plot$padj < fdr & df_plot$log2FC >  lfc] <- "Up"
  df_plot$signif[df_plot$padj < fdr & df_plot$log2FC < -lfc] <- "Down"

  sig_idx <- which(df_plot$signif != "NS")
  label_idx <- integer(0)

  if (length(sig_idx) > 0) {
    ord <- order(df_plot$padj[sig_idx], decreasing = FALSE)
    label_idx <- sig_idx[ord][seq_len(min(top_n, length(sig_idx)))]
  }

  if (!is.null(genes_highlight)) {
    force_idx <- which(df_plot$gene %in% genes_highlight)
    label_idx <- union(label_idx, force_idx)
  }

  df_plot$label <- NA_character_
  if (length(label_idx) > 0) df_plot$label[label_idx] <- df_plot$gene[label_idx]

  ggplot(df_plot, aes(x = log2FC, y = neglog10FDR)) +
    geom_point(aes(color = signif), alpha = 0.7, size = point_size) +
    geom_vline(xintercept = c(-lfc, lfc), linetype = "dashed") +
    geom_hline(yintercept = -log10(fdr), linetype = "dashed") +
    geom_text_repel(
      data = subset(df_plot, !is.na(label)),
      aes(label = label),
      size = label_size,
      max.overlaps = Inf,
      box.padding = 0.3,
      point.padding = 0.2,
      min.segment.length = 0
    ) +
    scale_color_manual(values = c("Down" = "#2C7BB6", "NS" = "grey70", "Up" = "#D7191C")) +
    labs(title = title, x = "log2 Fold Change", y = "-log10(FDR)") +
    theme_classic(base_size = volcano_base_size) +
    theme(legend.position = "right")
}

sig_set <- function(df, direction = c("up", "down"), fdr = 0.05, lfc = 0.1) {
  direction <- match.arg(direction)
  df <- df[!is.na(df$padj) & !is.na(df$log2FC), ]
  if (direction == "up") {
    unique(df$gene[df$padj < fdr & df$log2FC > lfc])
  } else {
    unique(df$gene[df$padj < fdr & df$log2FC < -lfc])
  }
}

message("[1] Loading current ADNI projection table")

if (file.exists(adni_projection_file)) {
  clinical <- read.csv(adni_projection_file, check.names = FALSE)
  projection_source <- adni_projection_file
} else if (file.exists(adni_projection_all_file)) {
  warning("ADpatho projection table not found. Falling back to all table and filtering ADpatho == 'AD'.")
  clinical <- read.csv(adni_projection_all_file, check.names = FALSE)
  projection_source <- adni_projection_all_file
  if (!("ADpatho" %in% colnames(clinical))) stop("Fallback all table does not contain ADpatho.")
  clinical <- clinical %>% filter(ADpatho == "AD")
} else {
  stop("Neither ADpatho nor all ADNI projection table was found.")
}

required_cols <- c("RID", "resilience", "predicted_cluster_like")
missing_cols <- setdiff(required_cols, colnames(clinical))
if (length(missing_cols) > 0) {
  stop("ADNI projection table is missing required columns: ", paste(missing_cols, collapse = ", "))
}

clinical <- clinical %>%
  mutate(
    RID = as.integer(RID),
    resilience = as.character(resilience),
    predicted_cluster_like = as.character(predicted_cluster_like),
    dep_group = case_when(
      resilience == "Low" ~ "non_resilience",
      resilience == "High" & predicted_cluster_like %in% c("cluster1", "cluster2") ~ predicted_cluster_like,
      TRUE ~ NA_character_
    ),
    dep_group = factor(dep_group, levels = group_levels)
  ) %>%
  filter(!is.na(dep_group), dep_group %in% group_levels)

message("    Projection source: ", projection_source)
message("    Group distribution:")
print(table(clinical$dep_group, useNA = "ifany"))

write.csv(
  clinical,
  file.path(table_dir, "ADNI_current_projected_groups_for_DEP.csv"),
  row.names = FALSE,
  quote = FALSE
)

message("[2] Loading ADNI SOMAscan matrix")

array <- fread(somascan_matrix_file, header = TRUE) %>% as.data.frame()
somascan <- read.csv(somascan_info_file, check.names = FALSE)

if (!all(c("RID", "VISCODE2") %in% colnames(array))) {
  stop("SOMAscan matrix must contain RID and VISCODE2.")
}
if (!all(c("Analytes", "EntrezGeneSymbol") %in% colnames(somascan))) {
  stop("SOMAscan annotation must contain Analytes and EntrezGeneSymbol.")
}

array <- array %>%
  dplyr::mutate(RID = as.integer(RID)) %>%
  dplyr::filter(VISCODE2 == protein_visit, RID %in% clinical$RID) %>%
  dplyr::arrange(RID) %>%
  dplyr::distinct(RID, .keep_all = TRUE)

somascan <- somascan %>%
  dplyr::select(Analytes, EntrezGeneSymbol) %>%
  dplyr::filter(!is.na(EntrezGeneSymbol), EntrezGeneSymbol != "")

available_analytes <- intersect(somascan$Analytes, colnames(array))
somascan <- somascan %>% filter(Analytes %in% available_analytes)

if (collapse_analytes_by_gene) {
  somascan_filtered <- somascan %>%
    dplyr::mutate(Xnum = as.numeric(stringr::str_remove(Analytes, "^X"))) %>%
    dplyr::group_by(EntrezGeneSymbol) %>%
    dplyr::slice_max(Xnum, n = 1, with_ties = FALSE) %>%
    dplyr::ungroup()
} else {
  somascan_filtered <- somascan %>%
    distinct(Analytes, .keep_all = TRUE)
}

array_expr <- array %>%
  dplyr::select(RID, dplyr::all_of(somascan_filtered$Analytes))

rownames(array_expr) <- array_expr$RID
array_expr <- array_expr[, setdiff(colnames(array_expr), "RID"), drop = FALSE]

for (cc in colnames(array_expr)) {
  array_expr[[cc]] <- clean_numeric(array_expr[[cc]])
}

if (impute_missing_by_min) {
  array_expr <- as.data.frame(
    apply(array_expr, 2, function(x) {
      min_val <- min(x, na.rm = TRUE)
      x[is.na(x)] <- min_val
      x
    })
  )
}

colnames(array_expr) <- somascan_filtered$EntrezGeneSymbol[match(colnames(array_expr), somascan_filtered$Analytes)]

array_expr$RID <- rownames(array_expr)
expr_long <- array_expr %>%
  pivot_longer(cols = -RID, names_to = "gene", values_to = "expr") %>%
  mutate(RID = as.integer(RID), expr = clean_numeric(expr)) %>%
  group_by(RID, gene) %>%
  summarise(expr = mean(expr, na.rm = TRUE), .groups = "drop") %>%
  mutate(expr = ifelse(is.nan(expr), NA_real_, expr))

array_symbol <- expr_long %>%
  pivot_wider(names_from = gene, values_from = expr) %>%
  arrange(RID)

common_ids <- intersect(array_symbol$RID, clinical$RID)
array_symbol <- array_symbol %>% filter(RID %in% common_ids)
clinical_sub <- clinical[match(array_symbol$RID, clinical$RID), c("RID", "dep_group")]

expr <- array_symbol %>%
  dplyr::select(-RID) %>%
  as.data.frame()

rownames(expr) <- array_symbol$RID
expr <- t(as.matrix(expr))
storage.mode(expr) <- "numeric"

keep_genes <- apply(expr, 1, function(z) {
  all(is.finite(z)) && !any(is.na(z)) && sd(z) > 0
})
expr <- expr[keep_genes, , drop = FALSE]

groups <- factor(clinical_sub$dep_group, levels = group_levels)

message("    Samples used for DEP: ", ncol(expr))
message("    Proteins/genes used for DEP: ", nrow(expr))
message("    Group distribution after expression merge:")
print(table(groups, useNA = "ifany"))

write.csv(
  data.frame(RID = colnames(expr), dep_group = as.character(groups)),
  file.path(table_dir, "ADNI_DEP_samples_used.csv"),
  row.names = FALSE,
  quote = FALSE
)

write.csv(
  somascan_filtered,
  file.path(table_dir, "ADNI_DEP_somascan_analytes_used.csv"),
  row.names = FALSE,
  quote = FALSE
)

message("[3] Running Welch t-test DEP contrasts")

res_c1_vs_non <- run_ttest(expr, groups, "cluster1", "non_resilience", offset = offset)
res_c2_vs_non <- run_ttest(expr, groups, "cluster2", "non_resilience", offset = offset)
res_c1_vs_c2  <- run_ttest(expr, groups, "cluster1", "cluster2", offset = offset)

res_all <- bind_rows(res_c1_vs_non, res_c2_vs_non, res_c1_vs_c2)

write.csv(
  res_all,
  file.path(table_dir, "DE_all_contrasts_current_groups.csv"),
  row.names = FALSE,
  quote = FALSE
)

write.csv(
  res_all,
  file.path(outdir, "DE_all_contrasts.csv"),
  row.names = FALSE,
  quote = FALSE
)

message("[4] Drawing volcano plots")

p1 <- volcano_plot(res_c1_vs_non, title = "cluster1 vs non_resilience", fdr = fdr_cutoff, lfc = lfc_cutoff, top_n = top_n_labels)
p2 <- volcano_plot(res_c2_vs_non, title = "cluster2 vs non_resilience", fdr = fdr_cutoff, lfc = lfc_cutoff, top_n = top_n_labels)
p3 <- volcano_plot(res_c1_vs_c2,  title = "cluster1 vs cluster2", fdr = fdr_cutoff, lfc = lfc_cutoff, top_n = top_n_labels)

p_all <- (p1 | p2) + plot_layout(guides = "collect") & theme(legend.position = "bottom")
p_three <- (p1 | p2 | p3) + plot_layout(guides = "collect") & theme(legend.position = "bottom")

ggsave(file.path(plot_dir, "volcano_cluster1_vs_non_resilience.png"), p1, width = single_volcano_width, height = single_volcano_height, dpi = 600)
ggsave(file.path(plot_dir, "volcano_cluster2_vs_non_resilience.png"), p2, width = single_volcano_width, height = single_volcano_height, dpi = 600)
ggsave(file.path(plot_dir, "volcano_cluster1_vs_cluster2.png"), p3, width = single_volcano_width, height = single_volcano_height, dpi = 600)

ggsave(file.path(plot_dir, "volcano_cluster_vs_non_resilience.png"), p_all, width = combined_volcano_width, height = combined_volcano_height, dpi = 600)
ggsave(file.path(plot_dir, "volcano_all_three.png"), p_three, width = combined_volcano_width + 3, height = combined_volcano_height, dpi = 600)

ggsave(file.path(plot_dir, "volcano_cluster_vs_non_resilience.svg"), p_all, width = combined_volcano_width, height = combined_volcano_height, device = "svg")
ggsave(file.path(plot_dir, "volcano_all_three.svg"), p_three, width = combined_volcano_width + 3, height = combined_volcano_height, device = "svg")

ggsave(file.path(outdir, "volcano_all_three.png"), p_all, width = 6, height = 4, dpi = 600)

message("[5] Drawing Venn diagrams")

up_c1 <- sig_set(res_c1_vs_non, "up", fdr = fdr_cutoff, lfc = lfc_cutoff)
up_c2 <- sig_set(res_c2_vs_non, "up", fdr = fdr_cutoff, lfc = lfc_cutoff)
down_c1 <- sig_set(res_c1_vs_non, "down", fdr = fdr_cutoff, lfc = lfc_cutoff)
down_c2 <- sig_set(res_c2_vs_non, "down", fdr = fdr_cutoff, lfc = lfc_cutoff)

venn_up <- ggVennDiagram(
  list(`c1_non` = up_c1, `c2_non` = up_c2),
  label_size = 8
) +
  labs(title = paste0("Upregulated (FDR < ", fdr_cutoff, ", log2FC > 0)")) +
  theme(legend.position = "none")

venn_down <- ggVennDiagram(
  list(`c1_non` = down_c1, `c2_non` = down_c2),
  label_size = 8
) +
  labs(title = paste0("Downregulated (FDR < ", fdr_cutoff, ", log2FC < 0)")) +
  theme(legend.position = "none")

ggsave(file.path(plot_dir, "Venn_up_cluster_vs_non.svg"),   venn_up,   width = venn_width, height = venn_height)
ggsave(file.path(plot_dir, "Venn_down_cluster_vs_non.svg"), venn_down, width = venn_width, height = venn_height)
ggsave(file.path(plot_dir, "Venn_up_cluster_vs_non.png"),   venn_up,   width = venn_width, height = venn_height, dpi = 600)
ggsave(file.path(plot_dir, "Venn_down_cluster_vs_non.png"), venn_down, width = venn_width, height = venn_height, dpi = 600)

ggsave(file.path(outdir, "Venn_up_cluster_vs_non.svg"),   venn_up,   width = venn_width, height = venn_height)
ggsave(file.path(outdir, "Venn_down_cluster_vs_non.svg"), venn_down, width = venn_width, height = venn_height)

dep_summary <- res_all %>%
  group_by(comp) %>%
  summarise(
    n_total = n(),
    n_FDR_lt_cutoff = sum(padj < fdr_cutoff, na.rm = TRUE),
    n_up = sum(padj < fdr_cutoff & log2FC > 0, na.rm = TRUE),
    n_down = sum(padj < fdr_cutoff & log2FC < 0, na.rm = TRUE),
    .groups = "drop"
  )
write.csv(dep_summary, file.path(table_dir, "DE_summary_current_groups.csv"), row.names = FALSE, quote = FALSE)

settings <- data.frame(
  setting = c(
    "projection_source",
    "somascan_matrix_file",
    "somascan_info_file",
    "protein_visit",
    "group_definition",
    "collapse_analytes_by_gene",
    "impute_missing_by_min",
    "fdr_cutoff",
    "lfc_cutoff"
  ),
  value = c(
    projection_source,
    somascan_matrix_file,
    somascan_info_file,
    protein_visit,
    "High resilience: predicted_cluster_like; Low resilience: non_resilience",
    collapse_analytes_by_gene,
    impute_missing_by_min,
    fdr_cutoff,
    lfc_cutoff
  )
)
write.csv(settings, file.path(table_dir, "DE_current_groups_settings.csv"), row.names = FALSE, quote = FALSE)

sink(file.path(outdir, "sessionInfo_DE_current_groups.txt"))
print(sessionInfo())
sink()

message("Done. Results saved in: ", outdir)
