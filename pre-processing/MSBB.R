rm(list=ls())
options(stringsAsFactors=FALSE)

path <- "/share1/kitani/data_from_first/resilience_study"
setwd(path)

library(data.table)
library(ggplot2)
library(dplyr)
library(sva)

out.dir <- "/out.dir.kitani"
script.dir <- "/script"

dir.create(paste0(path, out.dir), recursive = TRUE)
dir.create(paste0(path, script.dir), recursive = TRUE)

### MSBB metadata
df_msbb <- fread("data/resilience/MSBB_Normalized_counts_(CQN).tsv") %>% as.data.frame()
row.names(df_msbb) <- df_msbb$feature
df_msbb <- df_msbb[, -1]
meta_msbb <- read.csv("data/resilience/RNAseq_Harmonization_MSBB_combined_metadata.csv")
meta_msbb_merge <- meta_msbb[meta_msbb$specimenID %in% names(df_msbb), ]

###################
## Resilience Score Calculation
###################

# Calculate resilience score using a linear model
clinical.msbb <- meta_msbb_merge
model <- lm(CDR ~ CERAD + Braak, data = clinical.msbb)
clinical.msbb$cdr_pred <- predict(model, newdata = clinical.msbb)
clinical.msbb$resilience_score <- clinical.msbb$cdr_pred - clinical.msbb$CDR
clinical.msbb$norm_resilience_score <- scale(clinical.msbb$resilience_score)
clinical.msbb$norm_resilience_score <- as.numeric(clinical.msbb$norm_resilience_score)
clinical.msbb$cohort <- "MSBB"

write.csv(clinical.msbb, "out.dir.kitani/msbb_metadata_withresilience.csv", quote = FALSE)

#####################
# PCA and Gene Correlation
#####################

df_msbb_sorted <- df_msbb[, order(names(df_msbb))]
clinical.msbb_sorted <- clinical.msbb[order(clinical.msbb$specimenID), ]
exprs_adjusted <- ComBat(dat = df_msbb_sorted, batch = clinical.msbb_sorted$sequencingBatch, mod = NULL, par.prior = TRUE, prior.plots = FALSE)

# Calculate correlations between gene expression and resilience score
resilience_scores <- clinical.msbb_sorted$norm_resilience_score[match(colnames(exprs_adjusted), clinical.msbb_sorted$specimenID)]
correlations_with_genes <- sapply(rownames(exprs_adjusted), function(gene) {
  gene_expression <- as.numeric(exprs_adjusted[gene, ])
  cor(gene_expression, resilience_scores, use="complete.obs")
}, simplify = "array")

# Save datasets with varying numbers of top correlated genes
numbers <- c(1000, 2000, 3000, 4000, 5000)

for (n in numbers) {
  correlations_df <- data.frame(gene=names(correlations_with_genes), correlation=correlations_with_genes)
  correlations_df <- correlations_df[order(-correlations_df$correlation),] # Sort in descending order
  top_positive_genes <- head(correlations_df$gene, n / 2)
  top_negative_genes <- tail(correlations_df$gene, n / 2)
  selected_genes <- c(top_positive_genes, top_negative_genes)
  selected_genes_df <- df_msbb_sorted[selected_genes, , drop = FALSE]
  top_genes_df_sorted <- selected_genes_df
  
  exprs_adjusted_2 <- as.data.frame(t(top_genes_df_sorted))
  exprs_adjusted_2$specimenID <- row.names(exprs_adjusted_2)
  resilience <- clinical.msbb_sorted[, c("specimenID", "norm_resilience_score")]
  
  merged_df <- merge(exprs_adjusted_2, resilience, by="specimenID")
  row.names(merged_df) <- merged_df$specimenID
  merged_df <- merged_df[, -1]
  
  merged_df <- na.omit(merged_df)
  write.csv(merged_df, file=paste0(path, out.dir, "/exp_msbb_with.resilience_cor_seqbatch_", n, ".csv"), quote = FALSE)
}
