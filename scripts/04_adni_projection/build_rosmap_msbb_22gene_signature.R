#!/usr/bin/env Rscript
# Script cleaned for public release. Edit /path/to/... inputs before running.

rm(list = ls())
options(stringsAsFactors = FALSE)

project_dir <- "/path/to/project"
setwd(project_dir)

outdir <- "out/clustering/adni_projection_signature"
table_dir <- file.path(outdir, "tables")
dir.create(table_dir, recursive = TRUE, showWarnings = FALSE)

expr_merged_combat_file <- "/path/to/project/out/combined/exp_merged_combat_exclude89.csv"
meta_merged_file <- "/path/to/project/out/combined/meta_merged_exclude89.csv"
selected_gene_file <- "out/clustering/positice_gene_annotation.csv"
gene_anno_file <- "/path/to/annotation/genelist_human_2101gene.txt"
somascan_info_file <- "/path/to/ADNI/ADNI_Cruchaga_lab_CSF_SOMAscan7k_analyte_information_20_06_2023.csv"

cluster_label_file <- "out/clustering/subtypes.csv"

cohorts_to_use <- c("rosmap", "msbb")
excluded_analytes <- c("X9750.7", "X9341.1")
min_abs_delta_for_marker_flag <- 0

sample_col_candidates <- c("specimenID", "sampleID", "sample_id", "SampleID", "IID", "id", "sample")
cluster_col_candidates <- c("cluster", "Cluster", "subtype", "Subtype", "cluster_label", "predicted_cluster", "group")
cohort_col_candidates <- c("cohort", "Cohort", "dataset", "Dataset")

suppressPackageStartupMessages({
  library(dplyr)
  library(data.table)
  library(tidyr)
  library(tibble)
})

sanitize_sample_ids <- function(x) {
  x <- as.character(x)
  gsub("-", ".", x)
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

zscore_rows <- function(mat) {
  out <- t(scale(t(as.matrix(mat))))
  out[!is.finite(out)] <- NA_real_
  out
}

message("[1] Loading selected genes and ADNI SOMAscan overlap")
selected_genes_raw <- read.csv(selected_gene_file, check.names = FALSE)
if (!("gene" %in% colnames(selected_genes_raw))) stop("selected_gene_file must contain column: gene")

gene_anno <- fread(gene_anno_file)
gene_anno <- gene_anno[, c(2, 5)]
names(gene_anno) <- c("Symbol", "gene")
gene_anno <- as.data.frame(gene_anno)

somascan_info <- read.csv(somascan_info_file, check.names = FALSE)
if (!all(c("Analytes", "EntrezGeneSymbol") %in% colnames(somascan_info))) {
  stop("SOMAscan info file must contain Analytes and EntrezGeneSymbol")
}
somascan_info <- somascan_info %>% filter(!(Analytes %in% excluded_analytes))

selected_genes <- merge(selected_genes_raw, gene_anno, by = "gene", all.x = TRUE)
selected_genes$Symbol[is.na(selected_genes$Symbol) | selected_genes$Symbol == ""] <-
  selected_genes$gene[is.na(selected_genes$Symbol) | selected_genes$Symbol == ""]
selected_genes <- selected_genes %>% select(gene, Symbol) %>% distinct()

shared_genes <- selected_genes %>%
  filter(Symbol %in% somascan_info$EntrezGeneSymbol) %>%
  left_join(
    somascan_info %>% select(Analytes, EntrezGeneSymbol) %>% distinct() %>% rename(Symbol = EntrezGeneSymbol),
    by = "Symbol"
  ) %>%
  distinct(gene, Symbol, Analytes)

if (nrow(shared_genes) == 0) stop("No selected genes overlap with ADNI SOMAscan analytes.")
write.csv(shared_genes, file.path(table_dir, "shared_22_genes_with_adni_somascan.csv"), row.names = FALSE, quote = FALSE)
message("    Unique shared genes: ", length(unique(shared_genes$gene)))
message("    Unique shared symbols: ", length(unique(shared_genes$Symbol)))

message("[2] Loading expression and metadata")
expr <- read.csv(expr_merged_combat_file, row.names = 1, check.names = FALSE)
expr <- as.matrix(expr)
storage.mode(expr) <- "numeric"
colnames(expr) <- sanitize_sample_ids(colnames(expr))

meta <- fread(meta_merged_file) %>% as.data.frame()
sample_col <- first_existing_col(meta, sample_col_candidates)
if (is.na(sample_col)) stop("Could not find sample ID column in merged metadata")
meta$specimenID <- sanitize_sample_ids(meta[[sample_col]])
if (!("cohort" %in% colnames(meta))) stop("Merged metadata must contain cohort")
meta$cohort <- tolower(as.character(meta$cohort))

expr22 <- expr[rownames(expr) %in% unique(shared_genes$gene), , drop = FALSE]
expr22 <- remove_bad_genes(expr22)
if (nrow(expr22) == 0) stop("No shared genes are present in expression matrix after filtering")

message("[3] Loading existing cluster labels")
if (!file.exists(cluster_label_file)) {
  stop("Cluster label file not found: ", cluster_label_file, "\nSet cluster_label_file to a CSV with specimenID and cluster columns.")
}
cluster_df <- read.csv(cluster_label_file, check.names = FALSE)
sample_col_cluster <- first_existing_col(cluster_df, sample_col_candidates)
cluster_col <- first_existing_col(cluster_df, cluster_col_candidates)
if (is.na(sample_col_cluster)) stop("Could not find sample ID column in cluster label file")
if (is.na(cluster_col)) stop("Could not find cluster label column in cluster label file")
cluster_df <- cluster_df %>%
  mutate(
    specimenID = sanitize_sample_ids(.data[[sample_col_cluster]]),
    cluster = as.factor(.data[[cluster_col]])
  ) %>%
  filter(cluster != "non-resilience") %>%
  droplevels()

cohort_col_cluster <- first_existing_col(cluster_df, cohort_col_candidates)
if (!is.na(cohort_col_cluster)) cluster_df$cohort_cluster_file <- tolower(as.character(cluster_df[[cohort_col_cluster]]))

meta_use <- meta %>% filter(cohort %in% cohorts_to_use) %>% select(specimenID, cohort, everything())
sample_table <- meta_use %>% inner_join(cluster_df %>% select(specimenID, cluster, everything()), by = "specimenID")
common_samples <- intersect(colnames(expr22), sample_table$specimenID)
if (length(common_samples) == 0) stop("No common samples among expression, metadata, and cluster labels")

expr22 <- expr22[, common_samples, drop = FALSE]
sample_table <- sample_table[match(common_samples, sample_table$specimenID), , drop = FALSE]
stopifnot(identical(colnames(expr22), sample_table$specimenID))
write.csv(sample_table, file.path(table_dir, "ROSMAP_MSBB_samples_used_for_22gene_signature.csv"), row.names = FALSE, quote = FALSE)
message("    Samples used: ", ncol(expr22))
message("    Genes used: ", nrow(expr22))
print(table(sample_table$cluster, useNA = "ifany"))

message("[4] Building cluster centroids")
expr22_z <- zscore_rows(expr22)

gene_symbol_map <- shared_genes %>% select(gene, Symbol) %>% distinct() %>% filter(gene %in% rownames(expr22_z))
symbol_vec <- gene_symbol_map$Symbol[match(rownames(expr22_z), gene_symbol_map$gene)]
symbol_vec[is.na(symbol_vec)] <- rownames(expr22_z)[is.na(symbol_vec)]

expr_z_by_sample <- as.data.frame(t(expr22_z))
expr_z_by_sample$specimenID <- rownames(expr_z_by_sample)
expr_z_by_sample <- expr_z_by_sample %>% relocate(specimenID) %>% left_join(sample_table %>% select(specimenID, cohort, cluster), by = "specimenID")
write.csv(expr_z_by_sample, file.path(table_dir, "ROSMAP_MSBB_22gene_expression_z_by_sample.csv"), row.names = FALSE, quote = FALSE)

clusters <- levels(droplevels(sample_table$cluster))
centroid_mat <- matrix(NA_real_, nrow = nrow(expr22_z), ncol = length(clusters), dimnames = list(rownames(expr22_z), clusters))
for (cl in clusters) {
  samples_cl <- sample_table$specimenID[sample_table$cluster == cl]
  centroid_mat[, cl] <- rowMeans(expr22_z[, samples_cl, drop = FALSE], na.rm = TRUE)
}

centroid_df <- as.data.frame(centroid_mat) %>% rownames_to_column("gene") %>%
  mutate(Symbol = symbol_vec[match(gene, rownames(expr22_z))]) %>% relocate(gene, Symbol)
write.csv(centroid_df, file.path(table_dir, "ROSMAP_MSBB_22gene_cluster_centroids.csv"), row.names = FALSE, quote = FALSE)

message("[5] Building cluster marker table")
marker_rows <- list()
for (cl in clusters) {
  samples_in <- sample_table$specimenID[sample_table$cluster == cl]
  samples_out <- sample_table$specimenID[sample_table$cluster != cl]
  for (g in rownames(expr22_z)) {
    xin <- as.numeric(expr22_z[g, samples_in])
    xout <- as.numeric(expr22_z[g, samples_out])
    p <- tryCatch(t.test(xin, xout)$p.value, error = function(e) NA_real_)
    delta <- mean(xin, na.rm = TRUE) - mean(xout, na.rm = TRUE)
    marker_rows[[length(marker_rows) + 1]] <- data.frame(
      cluster = cl,
      gene = g,
      Symbol = symbol_vec[match(g, rownames(expr22_z))],
      mean_z_in_cluster = mean(xin, na.rm = TRUE),
      mean_z_outside_cluster = mean(xout, na.rm = TRUE),
      delta_z_cluster_vs_others = delta,
      direction = ifelse(delta >= 0, "high_in_cluster", "low_in_cluster"),
      abs_delta_z = abs(delta),
      p_value = p,
      n_in_cluster = length(xin),
      n_outside_cluster = length(xout),
      stringsAsFactors = FALSE
    )
  }
}

marker_table <- bind_rows(marker_rows) %>%
  group_by(cluster) %>% mutate(FDR = p.adjust(p_value, method = "BH")) %>% ungroup() %>%
  mutate(use_as_marker = abs_delta_z >= min_abs_delta_for_marker_flag)
write.csv(marker_table, file.path(table_dir, "ROSMAP_MSBB_22gene_cluster_marker_table.csv"), row.names = FALSE, quote = FALSE)

signature_long <- marker_table %>% filter(use_as_marker) %>% arrange(cluster, desc(abs_delta_z))
write.csv(signature_long, file.path(table_dir, "ROSMAP_MSBB_22gene_cluster_signature_long.csv"), row.names = FALSE, quote = FALSE)

projection_reference <- list(
  shared_genes = shared_genes,
  sample_table = sample_table,
  expression_z_by_sample = expr_z_by_sample,
  centroid_df = centroid_df,
  centroid_mat = centroid_mat,
  marker_table = marker_table,
  signature_long = signature_long,
  clusters = clusters,
  settings = list(
    expr_merged_combat_file = expr_merged_combat_file,
    meta_merged_file = meta_merged_file,
    selected_gene_file = selected_gene_file,
    somascan_info_file = somascan_info_file,
    cluster_label_file = cluster_label_file,
    cohorts_to_use = cohorts_to_use
  )
)
saveRDS(projection_reference, file.path(table_dir, "ROSMAP_MSBB_22gene_projection_reference.rds"))

sink(file.path(outdir, "sessionInfo_build_projection_signature.txt")); print(sessionInfo()); sink()
message("Done. Signature files saved in: ", table_dir)
