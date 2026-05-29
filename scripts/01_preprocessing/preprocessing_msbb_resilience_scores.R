# Script cleaned for public release. Edit /path/to/... inputs before running.
rm(list=ls())
options(stringsAsFactors=FALSE)

path <- "/path/to/project"
setwd(path)

library(data.table)
library(ggplot2)
library(dplyr)
library(sva)

out.dir <- "/out"
script.dir <- "/script"

dir.create(paste0(path,out.dir), recursive = T)
dir.create(paste0(path,script.dir), recursive = T)
dir.create(paste0(path,out.dir,"/msbb"), recursive = T)

df_msbb <- fread("/path/to/input_data/resilience/MSBB_Normalized_counts_(CQN).tsv") %>% as.data.frame()
row.names(df_msbb) <- df_msbb$feature
df_msbb <- df_msbb[,-1]

meta_msbb <- read.csv("/path/to/input_data/resilience/RNAseq_Harmonization_MSBB_combined_metadata_251031.csv")
meta_msbb_merge <- meta_msbb[meta_msbb$specimenID %in% names(df_msbb),]

table(meta_msbb$tissue)

meta_msbb_merge <- meta_msbb_merge[
  meta_msbb_merge$tissue %in% c("frontal pole", "inferior frontal gyrus", "prefrontal cortex"),
]

length(unique(meta_msbb_merge$specimenID))

clinical.msbb <- meta_msbb_merge

clinical.msbb$ageDeath_numeric <- clinical.msbb$ageDeath
clinical.msbb$ageDeath_numeric[clinical.msbb$ageDeath_numeric == "90+"] <- "99"
clinical.msbb$ageDeath_numeric <- as.numeric(clinical.msbb$ageDeath_numeric)

clinical.msbb$sex <- factor(
  clinical.msbb$sex,
  levels = c("female", "male")
)

clinical.msbb$CDR <- as.numeric(clinical.msbb$CDR)
clinical.msbb$CERAD <- as.numeric(clinical.msbb$CERAD)
clinical.msbb$Braak <- as.numeric(clinical.msbb$Braak)

model <- lm(
  CDR ~ CERAD + Braak,
  data = clinical.msbb
)
summary(model)

clinical.msbb$cdr_pred <- predict(model, newdata = clinical.msbb)

clinical.msbb$resilience_score <- clinical.msbb$cdr_pred - clinical.msbb$CDR
clinical.msbb$norm_resilience_score <- as.numeric(scale(clinical.msbb$resilience_score))

model_cov <- lm(
  CDR ~ CERAD + Braak + ageDeath_numeric + sex,
  data = clinical.msbb
)
summary(model_cov)

clinical.msbb$cdr_pred_cov <- predict(model_cov, newdata = clinical.msbb)

clinical.msbb$resilience_score_cov <- clinical.msbb$cdr_pred_cov - clinical.msbb$CDR
clinical.msbb$norm_resilience_score_cov <- as.numeric(scale(clinical.msbb$resilience_score_cov))

clinical.msbb$cohort <- "MSBB"

p_cor <- ggplot(clinical.msbb,
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
    axis.text = element_text(size = 16)
  )

cor_val <- cor(
  clinical.msbb$resilience_score,
  clinical.msbb$resilience_score_cov,
  use = "complete.obs"
)

print(p_cor)

ggsave("out/msbb/resilience_correlation.svg",
       plot = p_cor, width = 6, height = 6)

clinical.msbb$sign_resilience_score <- ifelse(
  clinical.msbb$resilience_score >= 0, "positive", "negative"
)

clinical.msbb$sign_resilience_score_cov <- ifelse(
  clinical.msbb$resilience_score_cov >= 0, "positive", "negative"
)

sign_table <- table(
  original = clinical.msbb$sign_resilience_score,
  covariate_adjusted = clinical.msbb$sign_resilience_score_cov
)

print(sign_table)

n_sign_changed <- sum(
  clinical.msbb$sign_resilience_score != clinical.msbb$sign_resilience_score_cov,
  na.rm = TRUE
)

n_total_sign <- sum(
  complete.cases(
    clinical.msbb$resilience_score,
    clinical.msbb$resilience_score_cov
  )
)

prop_sign_changed <- n_sign_changed / n_total_sign

print(paste0("Number of sign-changed samples: ", n_sign_changed))
print(paste0("Proportion of sign-changed samples: ", round(prop_sign_changed * 100, 2), "%"))
