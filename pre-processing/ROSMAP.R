rm(list=ls())
options(stringsAsFactors=FALSE)

path <- "/path/to"
setwd(path)

library(data.table)
library(ggplot2)
library(dplyr)
library(sva)

out.dir <- "/out"
script.dir <- "/script"

dir.create(paste0(path, out.dir), recursive = TRUE)
dir.create(paste0(path, script.dir), recursive = TRUE)

### ROSMAP metadata
df_rosmap <- fread("data/resilience/ROSMAP_Normalized_counts_(CQN).tsv") %>% as.data.frame()
row.names(df_rosmap) <- df_rosmap$feature
df_rosmap <- df_rosmap[, -1]
meta_rosmap <- read.csv("data/resilience/RNAseq_Harmonization_ROSMAP_combined_metadata.csv")
meta_rosmap_merge <- meta_rosmap[meta_rosmap$specimenID %in% names(df_rosmap), ]

###################
## Resilience Score Calculation
###################

clinical.rosmap <- meta_rosmap_merge
clinical.rosmap$ceradsc <- 5 - clinical.rosmap$ceradsc
model <- lm(cts_mmse30_lv ~ ceradsc + braaksc, data = clinical.rosmap)
clinical.rosmap$mmse_pred <- predict(model, newdata = clinical.rosmap)
clinical.rosmap$resilience_score <- clinical.rosmap$cts_mmse30_lv - clinical.rosmap$mmse_pred
clinical.rosmap$norm_resilience_score <- scale(clinical.rosmap$resilience_score)
clinical.rosmap$norm_resilience_score <- as.numeric(clinical.rosmap$norm_resilience_score)
clinical.rosmap$cohort <- "rosmap"

write.csv(clinical.rosmap, "out/rosmap_metadata_withresilience.csv", quote = FALSE)

#####################
# PCA and Gene Correlation
#####################

clinical.rosmap_filtered <- clinical.rosmap[clinical.rosmap$tissue != "Head of caudate nucleus", ]
clinical.rosmap_filtered <- clinical.rosmap_filtered[clinical.rosmap_filtered$sequencingBatch != 9, ]
clinical.rosmap_filtered <- clinical.rosmap_filtered[!is.na(clinical.rosmap_filtered$norm_resilience_score), ]
clinical.rosmap_filtered <- clinical.rosmap_filtered[clinical.rosmap_filtered$assay == "rnaSeq", ]
df_rosmap_filtered <- df_rosmap[, clinical.rosmap_filtered$specimenID]

df_rosmap_sorted <- df_rosmap_filtered[, order(names(df_rosmap_filtered))]
clinical.rosmap_sorted <- clinical.rosmap_filtered[order(clinical.rosmap_filtered$specimenID), ]
exprs_adjusted <- ComBat(dat = df_rosmap_sorted, batch = clinical.rosmap_sorted$notes, mod = NULL, par.prior = TRUE, prior.plots = FALSE)

df_expr_adjusted <- as.data.frame(exprs_adjusted)
write.csv(df_expr_adjusted, "out/expr_rosmap_combat.csv", quote = FALSE)

pca_result <- prcomp(t(exprs_adjusted), scale = TRUE)
pca_data <- as.data.frame(pca_result$x)
pca_data <- pca_data[, c(1, 2)]
pca_data$specimenID <- rownames(pca_data)
pca_data <- merge(pca_data, clinical.rosmap_sorted, by = "specimenID")

# Calculate correlations between gene expression and resilience score
resilience_scores <- clinical.rosmap_sorted$norm_resilience_score[match(colnames(exprs_adjusted), clinical.rosmap_sorted$specimenID)]
correlations_with_genes <- sapply(rownames(exprs_adjusted), function(gene) {
  gene_expression <- as.numeric(exprs_adjusted[gene, ])
  cor(gene_expression, resilience_scores, use="complete.obs")
}, simplify = "array")

# Save datasets with varying numbers of top correlated genes
numbers <- c(1000, 2000, 3000, 4000, 5000)

for (n in numbers) {
  correlations_df <- data.frame(gene=names(correlations_with_genes), correlation=correlations_with_genes)
  correlations_df <- correlations_df[order(-correlations_df$correlation), ] # Sort in descending order
  top_positive_genes <- head(correlations_df$gene, n / 2)
  top_negative_genes <- tail(correlations_df$gene, n / 2)
  selected_genes <- c(top_positive_genes, top_negative_genes)
  selected_genes_df <- df_rosmap_sorted[selected_genes, , drop = FALSE]
  top_genes_df_sorted <- selected_genes_df
  
  exprs_adjusted_2 <- as.data.frame(t(top_genes_df_sorted))
  exprs_adjusted_2$specimenID <- row.names(exprs_adjusted_2)
  resilience <- pca_data[, c("specimenID", "norm_resilience_score")]
  
  merged_df <- merge(exprs_adjusted_2, resilience, by="specimenID")
  row.names(merged_df) <- merged_df$specimenID
  merged_df <- merged_df[, -1]
  
  merged_df <- na.omit(merged_df)
  write.csv(merged_df, file=paste0(path, out.dir, "/exp_rosmap_with.resilience_cor_seqbatch_", n, ".csv"), quote = FALSE)
}
