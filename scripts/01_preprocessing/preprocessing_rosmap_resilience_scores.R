# Script cleaned for public release. Edit /path/to/... inputs before running.
rm(list=ls())
options(stringsAsFactors=FALSE)

path <- "/path/to/project/"
setwd(path)

library(data.table)
library(ggplot2)
library(dplyr)
library(sva)
library(plotly)
library(rgl)

out.dir <- "/out"
script.dir <- "/script"

dir.create(paste0(path,out.dir), recursive = T)
dir.create(paste0(path,script.dir), recursive = T)
dir.create(paste0(path,out.dir,"/rosmap"), recursive = T)

df_rosmap <- fread("/path/to/input_data/resilience/ROSMAP_Normalized_counts_(CQN).tsv") %>% as.data.frame()
row.names(df_rosmap) <- df_rosmap$feature
df_rosmap <- df_rosmap[,-1]

meta_rosmap <- read.csv("/path/to/input_data/resilience/RNAseq_Harmonization_ROSMAP_combined_metadata.csv")
meta_rosmap_merge <- meta_rosmap[meta_rosmap$specimenID %in% names(df_rosmap),]

table(meta_rosmap$tissue)

meta_rosmap_merge <- meta_rosmap_merge[meta_rosmap_merge$tissue %in% c("dorsolateral prefrontal cortex","frontal cortex"),]
meta_rosmap_merge <- meta_rosmap_merge[meta_rosmap_merge$dcfdx_lv %in% c(1,2,4),]
meta_rosmap_merge <- meta_rosmap_merge[meta_rosmap_merge$assay=="rnaSeq",]
length(unique(meta_rosmap_merge$specimenID))

clinical.rosmap <- meta_rosmap_merge

clinical.rosmap$ceradsc <- 5 - clinical.rosmap$ceradsc

clinical.rosmap$age_death_numeric <- clinical.rosmap$age_death
clinical.rosmap$age_death_numeric[clinical.rosmap$age_death_numeric == "90+"] <- "99"
clinical.rosmap$age_death_numeric <- as.numeric(clinical.rosmap$age_death_numeric)

clinical.rosmap$msex <- factor(
  clinical.rosmap$msex,
  levels = c(0, 1),
  labels = c("female", "male")
)

clinical.rosmap$educ <- as.numeric(clinical.rosmap$educ)

model <- lm(
  cts_mmse30_lv ~ ceradsc + braaksc,
  data = clinical.rosmap
)
summary(model)

clinical.rosmap$mmse_pred <- predict(model, newdata = clinical.rosmap)
clinical.rosmap$resilience_score <- clinical.rosmap$cts_mmse30_lv - clinical.rosmap$mmse_pred
clinical.rosmap$norm_resilience_score <- as.numeric(scale(clinical.rosmap$resilience_score))

model_cov <- lm(
  cts_mmse30_lv ~ ceradsc + braaksc + age_death_numeric + msex + educ,
  data = clinical.rosmap
)
summary(model_cov)

clinical.rosmap$mmse_pred_cov <- predict(model_cov, newdata = clinical.rosmap)
clinical.rosmap$resilience_score_cov <- clinical.rosmap$cts_mmse30_lv - clinical.rosmap$mmse_pred_cov
clinical.rosmap$norm_resilience_score_cov <- as.numeric(scale(clinical.rosmap$resilience_score_cov))

p_cor <- ggplot(clinical.rosmap, 
                aes(x = resilience_score, y = resilience_score_cov)) +
  geom_point(alpha = 0.3, size = 3) +
  geom_smooth(method = "lm", color = "red") +
  labs(
    x = "Resilience Score",
    y = "CV-adjusted Resilience Score"
  ) +
  theme_classic() +
  theme(
    text = element_text(size = 18),
    axis.title = element_text(size = 28),
    axis.text = element_text(size = 16),
  )

cor_val <- cor(
  clinical.rosmap$resilience_score,
  clinical.rosmap$resilience_score_cov,
  use = "complete.obs"
)

print(p_cor)
ggsave("out/rosmap/resilience_correlation.svg",
       plot = p_cor, width = 6, height = 6)

clinical.rosmap$sign_resilience_score <- ifelse(
  clinical.rosmap$resilience_score >= 0, "positive", "negative"
)

clinical.rosmap$sign_resilience_score_cov <- ifelse(
  clinical.rosmap$resilience_score_cov >= 0, "positive", "negative"
)

sign_table <- table(
  original = clinical.rosmap$sign_resilience_score,
  covariate_adjusted = clinical.rosmap$sign_resilience_score_cov
)

print(sign_table)

n_sign_changed <- sum(
  clinical.rosmap$sign_resilience_score != clinical.rosmap$sign_resilience_score_cov,
  na.rm = TRUE
)

n_total_sign <- sum(
  complete.cases(
    clinical.rosmap$resilience_score,
    clinical.rosmap$resilience_score_cov
  )
)

prop_sign_changed <- n_sign_changed / n_total_sign

print(paste0("Number of sign-changed samples: ", n_sign_changed))
print(paste0("Proportion of sign-changed samples: ", round(prop_sign_changed * 100, 2), "%"))
