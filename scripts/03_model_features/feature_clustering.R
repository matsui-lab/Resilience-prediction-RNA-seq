# Script cleaned for public release. Edit /path/to/... inputs before running.
rm(list = ls())
options(stringsAsFactors=FALSE)

path <- "/path/to/project"
setwd(path)

library(dplyr)
library(pheatmap)
library(RColorBrewer)
library(ggplot2)
library(data.table)

deg.table_posi <- read.csv("/path/to/project/out/feature_overlap_gene_visualization/features/union_positive_features.csv")
deg.table_posi$direction <- "positive"
names(deg.table_posi)[1] <- "Ens.ID"

deg.table_nega <- read.csv("/path/to/project/out/feature_overlap_gene_visualization/features/union_negative_features.csv")
deg.table_nega$direction <- "negative"
names(deg.table_nega)[1] <- "Ens.ID"

genes_nega <- deg.table_nega$Ens.ID
deg.table_posi <- deg.table_posi %>% filter(!(Ens.ID %in% genes_nega))

exp <- read.csv("/path/to/project/out/combined/exp_merged_combat_exclude89.csv", row.names = 1)
meta <- read.csv("/path/to/project/out/combined/meta_merged_exclude89.csv")

meta$specimenID_mod <- ifelse(grepl("^[0-9]", meta$specimenID),
                              paste0("X", meta$specimenID),
                              meta$specimenID)

meta_sub <- meta %>% filter(resilience == 1)

genes_keep <- deg.table_posi$Ens.ID
samples_keep <- meta_sub$specimenID_mod

exp_sub_posi <- exp[rownames(exp) %in% genes_keep, samples_keep, drop=FALSE]
exp_sub_posi <- exp_sub_posi[match(genes_keep, rownames(exp_sub_posi)), , drop=FALSE]

sample_anno_sub <- meta_sub %>%
  dplyr::select(specimenID_mod, cohort) %>%
  tibble::column_to_rownames("specimenID_mod")

ann_colors = list(
  cohort = setNames(brewer.pal(length(unique(sample_anno_sub$cohort)), "Set1"), unique(sample_anno_sub$cohort))
)
my_colors <- colorRampPalette(c("green", "black", "red"))(100)

my_breaks <- seq(-3, 3, length.out = 101)
svg("out/clustering/heatmap_resilience1_positive_features_filtered.svg", width = 10, height = 4)
pheatmap(
  as.matrix(exp_sub_posi),
  annotation_col = sample_anno_sub,
  annotation_colors = ann_colors,
  color = my_colors,
  breaks = my_breaks,
  show_colnames = FALSE,
  show_rownames = FALSE,
  cluster_cols = TRUE,
  cluster_rows = TRUE,
  scale = "row",
  clustering_distance_rows = "euclidean", #correlation
  clustering_distance_cols = "euclidean",
  clustering_method = "ward.D2", #"ward.D2""average"
)
dev.off()

res <- pheatmap(
  as.matrix(exp_sub_posi),
  annotation_col = sample_anno_sub,
  annotation_colors = ann_colors,
  color = my_colors,
  breaks = my_breaks,
  show_colnames = FALSE,
  show_rownames = FALSE,
  cluster_cols = TRUE,
  cluster_rows = TRUE,
  scale = "row",
  clustering_distance_rows = "euclidean",
  clustering_distance_cols = "euclidean",
  clustering_method = "ward.D2"
)

col_clusters <- cutree(res$tree_col, k = 2)
row_clusters <- cutree(res$tree_row, k = 2)

sample_anno_sub$col_cluster <- factor(col_clusters[rownames(sample_anno_sub)])

gene_order <- rownames(exp_sub_posi)
gene_anno_df <- data.frame(
  gene = gene_order,
  row_cluster = factor(row_clusters[gene_order])
)
rownames(gene_anno_df) <- gene_anno_df$gene

ann_colors$col_cluster <- c("1" = "#009E73", "2" = "#E69F00")
ann_colors$row_cluster <- c("1" = "#984EA3", "2" = "#4DAF4A", "3" = "#FF7F00")

png(
  filename = "out/clustering/heatmap_resilience1_positive_features_filtered_with_clusters.png",
  width = 10,
  height = 5,
  units = "in",
  res = 600
)
pheatmap(
  as.matrix(exp_sub_posi),
  annotation_col = sample_anno_sub,
  annotation_row = gene_anno_df["row_cluster"],
  annotation_colors = ann_colors,
  color = my_colors,
  breaks = my_breaks,
  show_colnames = FALSE,
  show_rownames = FALSE,
  cluster_cols = TRUE,
  cluster_rows = TRUE,
  scale = "row",
  clustering_distance_rows = "euclidean",
  clustering_distance_cols = "euclidean",
  clustering_method = "ward.D2"
)
dev.off()

df_col <- data.frame(
  sample = names(col_clusters),
  col_cluster = col_clusters,
  stringsAsFactors = FALSE
)

df_row <- data.frame(
  gene = names(row_clusters),
  row_cluster = row_clusters,
  stringsAsFactors = FALSE
)

somascan <- read.csv("/path/to/ADNI/ADNI_Cruchaga_lab_CSF_SOMAscan7k_analyte_information_20_06_2023.csv")
somascan <- somascan[,c("Analytes","EntrezGeneSymbol")]

exp_long <- as.data.frame(as.matrix(exp_sub_posi))
exp_long$gene <- rownames(exp_long)
exp_long <- tidyr::pivot_longer(exp_long, -gene, names_to = "sample", values_to = "expr")

exp_long <- exp_long %>%
  left_join(df_row, by = "gene") %>%
  left_join(df_col, by = "sample")

gene_anno <- fread("/path/to/annotation/genelist_human_2101gene.txt")
gene_anno <- gene_anno[,c(2,5)]
names(gene_anno) <- c("Gene_Symbol", "Ensembl")
gene_anno <- gene_anno %>% distinct(Gene_Symbol, Ensembl)

exp_long <- merge(exp_long, gene_anno, by.x = "gene", by.y = "Ensembl", all.x = T)
exp_long <- exp_long %>%
  mutate(Gene_Symbol = ifelse(is.na(Gene_Symbol) | Gene_Symbol == "", gene, Gene_Symbol))

samples_nonres <- meta %>% filter(resilience == 0) %>% pull(specimenID_mod)
exp_nonres <- exp[rownames(exp) %in% genes_keep, samples_nonres, drop=FALSE]
exp_nonres <- exp_nonres[match(genes_keep, rownames(exp_nonres)), , drop=FALSE]
exp_long_res1 <- as.data.frame(as.matrix(exp_sub_posi))
exp_long_res1$gene <- rownames(exp_long_res1)
exp_long_res1 <- tidyr::pivot_longer(exp_long_res1, -gene, names_to = "sample", values_to = "expr")
exp_long_res1 <- exp_long_res1 %>%
  left_join(df_row, by = "gene") %>%
  left_join(df_col, by = "sample") %>%
  mutate(group = paste0("cluster", col_cluster))
exp_long_nonres <- as.data.frame(as.matrix(exp_nonres))
exp_long_nonres$gene <- rownames(exp_long_nonres)
exp_long_nonres <- tidyr::pivot_longer(exp_long_nonres, -gene, names_to = "sample", values_to = "expr")
exp_long_nonres <- exp_long_nonres %>%
  left_join(df_row, by = "gene") %>%
  mutate(group = "non-resilience")
exp_long_all <- bind_rows(exp_long_res1, exp_long_nonres)
exp_long_all <- merge(exp_long_all, gene_anno, by.x = "gene", by.y = "Ensembl", all.x = T)
exp_long_all <- exp_long_all %>%
  mutate(Gene_Symbol = ifelse(is.na(Gene_Symbol) | Gene_Symbol == "", gene, Gene_Symbol)) %>%
  mutate(gene_facet = paste0(Gene_Symbol, " (C", row_cluster, ")"))

exp_long_all <- exp_long_all %>%
  mutate(
    row_cluster = as.integer(row_cluster),
    gene_facet = paste0(Gene_Symbol, " (C", row_cluster, ")")
  )

exp_long_all_2 <- exp_long_all[exp_long_all$Gene_Symbol %in% somascan$EntrezGeneSymbol,]
length(unique(exp_long_all_2$Gene_Symbol))

gene_facet_levels <- exp_long_all_2 %>%
  distinct(row_cluster, gene_facet) %>%
  arrange(row_cluster, gene_facet) %>%  
  pull(gene_facet)

exp_long_all_2$gene_facet <- factor(exp_long_all_2$gene_facet, levels = gene_facet_levels)

comparisons <- list(
  c("cluster1", "cluster2")
)

p <- ggplot(exp_long_all_2, aes(x = group, y = expr, fill = group)) +
  geom_boxplot(outlier.shape = NA, alpha = 0.85) +
  ggpubr::stat_compare_means(
    comparisons = comparisons,
    method = "t.test",
    label = "p.signif",
    hide.ns = FALSE,
    size = 3
  ) +
  xlab("Group") +
  ylab("Expression") +
  scale_fill_manual(
    values = c(
      "cluster1" = "#009E73",
      "cluster2" = "#E69F00",
      "non-resilience" = "gray"
    )
  ) +
  theme_bw(base_size = 12) +
  theme(
    strip.text = element_text(size = 9),
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "right"
  ) +
  scale_y_continuous(expand = expansion(mult = c(0.05, 0.25))) +
  facet_wrap(~ gene_facet, scales = "free_y", ncol = 6)
p
ggsave("out/clustering/boxplot_clusters_vs_nonresilience.svg",
       plot = p,
       width = 14, height = 8, units = "in")
ggsave("out/clustering/boxplot_clusters_vs_nonresilience.png",
       plot = p,
       width = 14, height = 6, units = "in", dpi=600)

t_test_results <- exp_long_all %>%
  filter(group %in% c("cluster1", "cluster2", "non-resilience")) %>%
  group_by(Gene_Symbol) %>%
  summarise(
    n_cluster1 = sum(group == "cluster1" & !is.na(expr)),
    n_cluster2 = sum(group == "cluster2" & !is.na(expr)),
    n_nonresilience = sum(group == "non-resilience" & !is.na(expr)),

    mean_cluster1 = mean(expr[group == "cluster1"], na.rm = TRUE),
    mean_cluster2 = mean(expr[group == "cluster2"], na.rm = TRUE),
    mean_nonresilience = mean(expr[group == "non-resilience"], na.rm = TRUE),

    median_cluster1 = median(expr[group == "cluster1"], na.rm = TRUE),
    median_cluster2 = median(expr[group == "cluster2"], na.rm = TRUE),
    median_nonresilience = median(expr[group == "non-resilience"], na.rm = TRUE),

    delta_cluster1_minus_cluster2 =
      mean_cluster1 - mean_cluster2,
    delta_cluster1_minus_nonresilience =
      mean_cluster1 - mean_nonresilience,
    delta_cluster2_minus_nonresilience =
      mean_cluster2 - mean_nonresilience,

    pval_cluster1_vs_cluster2 = tryCatch(
      t.test(expr[group == "cluster1"], expr[group == "cluster2"])$p.value,
      error = function(e) NA_real_
    ),
    pval_cluster1_vs_nonresilience = tryCatch(
      t.test(expr[group == "cluster1"], expr[group == "non-resilience"])$p.value,
      error = function(e) NA_real_
    ),
    pval_cluster2_vs_nonresilience = tryCatch(
      t.test(expr[group == "cluster2"], expr[group == "non-resilience"])$p.value,
      error = function(e) NA_real_
    ),
    .groups = "drop"
  ) %>%
  mutate(
    FDR_cluster1_vs_cluster2 =
      p.adjust(pval_cluster1_vs_cluster2, method = "fdr"),
    FDR_cluster1_vs_nonresilience =
      p.adjust(pval_cluster1_vs_nonresilience, method = "fdr"),
    FDR_cluster2_vs_nonresilience =
      p.adjust(pval_cluster2_vs_nonresilience, method = "fdr")
  )

write.csv(
  t_test_results,
  "out/clustering/t_test_clusters_vs_nonresilience.csv",
  row.names = FALSE,
  quote = FALSE
)

write.csv(gene_anno_df, "out/clustering/positice_gene_annotation.csv", row.names = FALSE, quote = F)

names(exp_long_all)
subtype <- exp_long_all[,c("sample", "group")] %>% unique()
write.csv(subtype, "out/clustering/subtypes.csv", row.names = F, quote = F)
