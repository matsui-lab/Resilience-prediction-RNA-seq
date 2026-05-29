# Script cleaned for public release. Edit /path/to/... inputs before running.
rm(list=ls())
options(stringsAsFactors=FALSE)

path <- "/path/to/project"
setwd(path)

library(data.table)
library(ggplot2)
library(dplyr)
library(sva)
library(stringr)
library(patchwork)

rosmap_df <- read.csv("out/rosmap/rosmap_gene_expression_ADpatho.csv", row.names = 1)
msbb_df <- read.csv("out/msbb/msbb_gene_expression_ADpatho.csv", row.names = 1)

rosmap_meta <- read.csv("out/rosmap/rosmap_ADpatho_metadata_withresilience.csv", fill=T, row.names = 1)
msbb_meta <- read.csv("out/msbb/msbb_ADpatho_metadata_withresilience.csv", fill=T, row.names = 1)

common_genes <- intersect(row.names(rosmap_df), row.names(msbb_df))

rosmap_df_filtered <- rosmap_df[row.names(rosmap_df) %in% common_genes,]
msbb_df_filtered <- msbb_df[row.names(msbb_df) %in% common_genes,]

names(rosmap_df_filtered) <- gsub("X","",names(rosmap_df_filtered))
rosmap_meta$specimenID <- gsub("-",".", rosmap_meta$specimenID)

parse_base_id <- function(ids) {
  sub("(_resequenced)?$", "", ids)
}

msbb_ids <- colnames(msbb_df_filtered)
msbb_meta$orig_id <- msbb_meta$specimenID

msbb_meta$base_id <- parse_base_id(msbb_meta$specimenID)
msbb_meta$has_reseq  <- str_detect(msbb_meta$specimenID, "_resequenced")

msbb_meta <- msbb_meta %>%
  group_by(base_id) %>%
  mutate(priority = case_when(
    has_reseq ~ 2,
    TRUE ~ 1
  )) %>%
  filter(priority == max(priority)) %>%
  ungroup()

msbb_df_filtered <- msbb_df_filtered[, msbb_meta$orig_id, drop=FALSE]

msbb_meta <- msbb_meta[msbb_meta$tissue=="frontal pole",]
msbb_df_filtered <- msbb_df_filtered[, msbb_meta$specimenID, drop=FALSE]

df_merged <- cbind(rosmap_df_filtered, msbb_df_filtered)

names(rosmap_meta)
rosmap_meta_filtered <- rosmap_meta[,c("specimenID","tissue","cohort","sequencingBatch","resilience_score","resilience",
                                       "apoe_genotype","age_death","msex")]

names(msbb_meta)
msbb_meta_filtered <- msbb_meta[,c("specimenID","tissue","cohort","sequencingBatch","resilience_score","resilience"
                                   ,"apoeGenotype","ageDeath","sex")]
names(rosmap_meta_filtered)[7:9] <- c("apoeGenotype","ageDeath","sex")

rosmap_meta_filtered$sex <- gsub(1,"male",rosmap_meta_filtered$sex)
rosmap_meta_filtered$sex <- gsub(0,"female",rosmap_meta_filtered$sex)

meta_merged <- rbind(rosmap_meta_filtered, msbb_meta_filtered)
meta_merged$ageDeath <- gsub("90+","99",meta_merged$ageDeath)
meta_merged$apoeGenotype <- as.character(meta_merged$apoeGenotype)
meta_merged$ageDeath <- as.numeric(as.character(meta_merged$ageDeath))
meta_merged$ageDeath <- as.numeric(meta_merged$ageDeath)

expr_t <- t(df_merged)
pca <- prcomp(expr_t, center=TRUE, scale.=TRUE)

pca_df <- as.data.frame(pca$x[,1:2])
pca_df$specimenID <- rownames(pca_df)
pca_merged <- pca_df %>%
  left_join(meta_merged, by = "specimenID")
pca_merged$cohort <- gsub("rosmap","ROSMAP",pca_merged$cohort)

p1 <- ggplot(pca_merged, aes(x = PC1, y = PC2, color = sequencingBatch)) +
  geom_point(size = 2, alpha = 0.8) +
  labs(title = "PCA colored by sequencingBatch", color = "sequencingBatch") +
  theme_bw() +
  theme(text = element_text(size = 16))
p1
p2 <- ggplot(pca_merged, aes(x = PC1, y = PC2, color = tissue)) +
  geom_point(size = 2, alpha = 0.8) +
  labs(title = "PCA colored by tissue", color = "tissue") +
  theme_bw() +
  theme(text = element_text(size = 16))
p2
p3 <- ggplot(pca_merged, aes(x = PC1, y = PC2, color = cohort)) +
  geom_point(size = 2, alpha = 0.8) +
  labs(title = "PCA colored by cohort", color = "cohort") +
  theme_bw() +
  theme(text = element_text(size = 16))
p4 <- ggplot(pca_merged, aes(x = PC1, y = PC2, color = apoeGenotype)) +
  geom_point(size = 2, alpha = 0.8) +
  labs(title = "PCA colored by apoeGenotype", color = "apoeGenotype") +
  theme_bw() +
  theme(text = element_text(size = 16))
p4
p5 <- ggplot(pca_merged, aes(x = PC1, y = PC2, color = ageDeath)) +
  geom_point(size = 2, alpha = 0.8) +
  labs(title = "PCA colored by ageDeath", color = "ageDeath") +
  theme_bw() +
  theme(text = element_text(size = 16))
p5
p6 <- ggplot(pca_merged, aes(x = PC1, y = PC2, color = sex)) +
  geom_point(size = 2, alpha = 0.8) +
  labs(title = "PCA colored by sex", color = "sex") +
  theme_bw() +
  theme(text = element_text(size = 16))
p6

p_all <- p1 + p2 + p3 + p4 + p5 + p6 + plot_layout(ncol = 3)
ggsave("out/combined/PCA_colored_by_batch_tissue_cohort.svg", plot = p_all, width = 18, height = 12)

batch <- meta_merged$sequencingBatch
names(batch) <- meta_merged$specimenID

combat_expr <- ComBat(dat = as.matrix(df_merged), batch = batch, par.prior = TRUE, prior.plots = FALSE)
combat_expr_t <- t(combat_expr)
pca_combat <- prcomp(combat_expr_t, center=TRUE, scale.=TRUE)

pca_combat_df <- as.data.frame(pca_combat$x[,1:2])
pca_combat_df$specimenID <- rownames(pca_combat_df)
pca_combat_merged <- pca_combat_df %>%
  left_join(meta_merged, by = "specimenID")

p1_c <- ggplot(pca_combat_merged, aes(x = PC1, y = PC2, color = sequencingBatch)) +
  geom_point(size = 2, alpha = 0.8) +
  labs(title = "Batch-corrected PCA: sequencingBatch", color = "sequencingBatch") +
  theme_bw() +
  theme(text = element_text(size = 16))

p2_c <- ggplot(pca_combat_merged, aes(x = PC1, y = PC2, color = tissue)) +
  geom_point(size = 2, alpha = 0.8) +
  labs(title = "Batch-corrected PCA: tissue", color = "tissue") +
  theme_bw() +
  theme(text = element_text(size = 16))

p3_c <- ggplot(pca_combat_merged, aes(x = PC1, y = PC2, color = cohort)) +
  geom_point(size = 2, alpha = 0.8) +
  labs(title = "Batch-corrected PCA: cohort", color = "cohort") +
  theme_bw() +
  theme(text = element_text(size = 16))

p4_c <- ggplot(pca_combat_merged, aes(x = PC1, y = PC2, color = apoeGenotype)) +
  geom_point(size = 2, alpha = 0.8) +
  labs(title = "Batch-corrected PCA: apoeGenotype", color = "apoeGenotype") +
  theme_bw() +
  theme(text = element_text(size = 16))

p5_c <- ggplot(pca_combat_merged, aes(x = PC1, y = PC2, color = ageDeath)) +
  geom_point(size = 2, alpha = 0.8) +
  labs(title = "Batch-corrected PCA: ageDeath", color = "ageDeath") +
  theme_bw() +
  theme(text = element_text(size = 16))

p6_c <- ggplot(pca_combat_merged, aes(x = PC1, y = PC2, color = sex)) +
  geom_point(size = 2, alpha = 0.8) +
  labs(title = "Batch-corrected PCA: sex", color = "sex") +
  theme_bw() +
  theme(text = element_text(size = 16))

p_all_c <- p1_c + p2_c + p3_c + p4_c + p5_c + p6_c + plot_layout(ncol = 3)
ggsave("out/combined/PCA_batchcorrected_by_batch_tissue_cohort.svg", plot = p_all_c, width = 18, height = 6)

write.csv(meta_merged, "out/combined/meta_marged.csv", quote = F)
write.csv(df_merged, "out/combined/exp_marged.csv", quote = F)
write.csv(combat_expr, "out/combined/exp_marged_combat.csv", quote = F)

meta_sub <- meta_merged[!meta_merged$sequencingBatch %in% c(8,9), ]
df_sub <- df_merged[, meta_sub$specimenID]

batch_sub <- meta_sub$sequencingBatch
names(batch_sub) <- meta_sub$specimenID

combat_expr_sub <- ComBat(dat = as.matrix(df_sub), batch = batch_sub, par.prior = TRUE, prior.plots = FALSE)
combat_expr_sub_t <- t(combat_expr_sub)
pca_combat_sub <- prcomp(combat_expr_sub_t, center=TRUE, scale.=TRUE)

pca_combat_sub_df <- as.data.frame(pca_combat_sub$x[,1:2])
pca_combat_sub_df$specimenID <- rownames(pca_combat_sub_df)
pca_combat_sub_merged <- pca_combat_sub_df %>%
  left_join(meta_sub, by = "specimenID")
pca_combat_sub_merged$cohort <- gsub("rosmap","ROSMAP",pca_combat_sub_merged$cohort)

p1_sub <- ggplot(pca_combat_sub_merged, aes(x = PC1, y = PC2, color = sequencingBatch)) +
  geom_point(size = 2, alpha = 0.8) +
  labs(title = "Batch-corrected PCA: sequencingBatch", color = "sequencingBatch") +
  theme_bw() +
  theme(text = element_text(size = 16))

p2_sub <- ggplot(pca_combat_sub_merged, aes(x = PC1, y = PC2, color = tissue)) +
  geom_point(size = 2, alpha = 0.8) +
  labs(title = "Batch-corrected PCA: tissue", color = "tissue") +
  theme_bw() +
  theme(text = element_text(size = 16))

p3_sub <- ggplot(pca_combat_sub_merged, aes(x = PC1, y = PC2, color = cohort)) +
  geom_point(size = 2, alpha = 0.8) +
  labs(title = "Batch-corrected PCA: cohort", color = "cohort") +
  theme_bw() +
  theme(text = element_text(size = 16))

p4_sub <- ggplot(pca_combat_sub_merged, aes(x = PC1, y = PC2, color = apoeGenotype)) +
  geom_point(size = 2, alpha = 0.8) +
  labs(title = "Batch-corrected PCA: apoeGenotype", color = "apoeGenotype") +
  theme_bw() +
  theme(text = element_text(size = 16))

p5_sub <- ggplot(pca_combat_sub_merged, aes(x = PC1, y = PC2, color = ageDeath)) +
  geom_point(size = 2, alpha = 0.8) +
  labs(title = "Batch-corrected PCA: ageDeath", color = "apoeDeath") +
  theme_bw() +
  theme(text = element_text(size = 16))

p6_sub <- ggplot(pca_combat_sub_merged, aes(x = PC1, y = PC2, color = sex)) +
  geom_point(size = 2, alpha = 0.8) +
  labs(title = "Batch-corrected PCA: sex", color = "sex") +
  theme_bw() +
  theme(text = element_text(size = 16))

p_all_sub <- p1_sub + p2_sub + p3_sub + p4_sub + p5_sub + p6_sub + plot_layout(ncol = 3)
p_all_sub

ggsave("out/combined/PCA_batchcorrected_by_batch_tissue_cohort_rm_outlier.svg", plot = p_all_sub, width = 18, height = 12)

na_resilience_specimenIDs <- meta_sub$specimenID[is.na(meta_sub$resilience)]
meta_sub <- meta_sub[!is.na(meta_sub$resilience), ]
combat_expr_sub <- combat_expr_sub[, meta_sub$specimenID, drop = FALSE]

write.csv(meta_sub, "out/combined/meta_merged_exclude89.csv", quote = FALSE, row.names = FALSE)
write.csv(combat_expr_sub, "out/combined/exp_merged_combat_exclude89.csv", quote = FALSE)
