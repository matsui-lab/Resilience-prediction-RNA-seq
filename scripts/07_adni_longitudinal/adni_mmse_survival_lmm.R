#!/usr/bin/env Rscript
# Script cleaned for public release. Edit /path/to/... inputs before running.

rm(list = ls())
options(stringsAsFactors = FALSE)

project_dir <- "/path/to/project"
setwd(project_dir)

adni_projection_file <- "out/ADNI/projected_rosmap_msbb_22gene_signature/tables/ADNI_projected_cluster_scores_with_clinical_ADpatho.csv"
adni_projection_all_file <- "out/ADNI/projected_rosmap_msbb_22gene_signature/tables/ADNI_projected_cluster_scores_with_clinical_all.csv"

mmse_file <- "/path/to/ADNI/All_Subjects_MMSE_19Jul2025.csv"
ptdemog_file <- "/path/to/ADNI/PTDEMOG_30Jul2025.csv"
apoe_file <- "/path/to/ADNI/APOERES_07Nov2025.csv"

outdir <- "out/ADNI/projected_rosmap_msbb_22gene_signature/mmse_current_projected_groups_survival_lmm"
table_dir <- file.path(outdir, "tables")
plot_dir <- file.path(outdir, "plots")
dir.create(table_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(plot_dir, recursive = TRUE, showWarnings = FALSE)

group_levels <- c("cluster1", "cluster2", "non_resilience")
reference_group <- "non_resilience"

group_colors <- c(
  "cluster1" = "#009E73",
  "cluster2" = "#E69F00",
  "non_resilience" = "gray60"
)

group_labels <- c(
  "cluster1" = "Cluster 1",
  "cluster2" = "Cluster 2",
  "non_resilience" = "Non-CR"
)

mmse_event_threshold <- 24

time_map <- c(
  "sc" = 0, "bl" = 0, "init" = 0,
  "m03" = 3, "m06" = 6, "m6" = 6,
  "m12" = 12, "m18" = 18, "m24" = 24,
  "m30" = 30, "m36" = 36, "m42" = 42,
  "m48" = 48, "m54" = 54, "m60" = 60,
  "m66" = 66, "m72" = 72, "m78" = 78,
  "m84" = 84, "m90" = 90, "m96" = 96,
  "m102" = 102, "1m02" = 102,
  "m108" = 108, "m114" = 114, "m120" = 120,
  "m126" = 126, "m132" = 132, "m138" = 138,
  "m144" = 144, "m150" = 150, "m156" = 156,
  "m162" = 162, "m168" = 168, "m174" = 174,
  "m180" = 180
)

cox_covariates_primary <- c(
  "AGE_AT_BASELINE",
  "SEX",
  "PTEDUCAT_clean",
  "APOE_e4_carrier"
)

lmm_covariates_primary <- c(
  "AGE_AT_BASELINE",
  "SEX",
  "PTEDUCAT_clean",
  "APOE_e4_carrier"
)

run_optional_cognitive_diagnosis_sensitivity <- FALSE
sensitivity_covariates_extra <- c("BASELINE_MMSE", "DIAGNOSIS")

km_width <- 8.5
km_height <- 5.2
risk_table_height <- 2.0
lmm_width <- 8.5
lmm_height <- 5.5
forest_width <- 7.2
forest_height <- 4.8
base_font_size <- 18

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(data.table)
  library(lubridate)
  library(survival)
  library(survminer)
  library(broom)
  library(broom.mixed)
  library(lme4)
  library(lmerTest)
  library(emmeans)
  library(readr)
  library(stringr)
})

clean_numeric <- function(x) {
  x <- as.character(x)
  x[x %in% c("", "NA", "NaN", "NULL", "null", "-4", ".", " ")] <- NA
  x <- ifelse(x == "90+", "99", x)
  suppressWarnings(as.numeric(x))
}

safe_factor <- function(x) {
  x <- as.character(x)
  x[x %in% c("", "NA", "NaN", "NULL", "null", "-4", ".", " ")] <- NA
  as.factor(x)
}

harmonize_sex <- function(x) {
  y <- as.character(x)
  dplyr::case_when(
    y %in% c("1", "M", "Male", "male", "MALE", "m") ~ "Male",
    y %in% c("2", "F", "Female", "female", "FEMALE", "f") ~ "Female",
    TRUE ~ NA_character_
  )
}

make_apoe4 <- function(genotype) {
  g <- as.character(genotype)
  dplyr::case_when(
    is.na(g) | g == "" ~ NA_character_,
    grepl("4", g) ~ "Carrier",
    TRUE ~ "Non-carrier"
  )
}

p_to_stars <- function(p) {
  if (is.na(p)) return("")
  if (p > 0.05) return("ns")
  if (p > 0.01) return("*")
  if (p > 0.001) return("**")
  if (p > 0.0001) return("***")
  return("****")
}

first_present <- function(candidates, cols) {
  z <- candidates[candidates %in% cols]
  if (length(z) == 0) return(NA_character_)
  z[1]
}

keep_usable_covariates <- function(df, covars) {
  covars <- covars[covars %in% colnames(df)]
  covars[sapply(covars, function(v) {
    z <- df[[v]]
    z <- z[!is.na(z)]
    length(unique(z)) >= 2
  })]
}

parse_visit_month <- function(viscode2) {
  v <- as.character(viscode2)
  out <- rep(NA_real_, length(v))
  hit <- v %in% names(time_map)
  out[hit] <- as.numeric(time_map[v[hit]])

  miss <- is.na(out) & grepl("^m[0-9]+$", v)
  out[miss] <- suppressWarnings(as.numeric(sub("^m", "", v[miss])))

  miss2 <- is.na(out) & grepl("^[0-9]+m[0-9]+$", v)
  if (any(miss2)) {
    tmp <- v[miss2]
    left <- suppressWarnings(as.numeric(sub("m.*$", "", tmp)))
    right <- suppressWarnings(as.numeric(sub("^.*m", "", tmp)))
    out[miss2] <- left * 100 + right
  }
  out
}

contrast_label <- function(x) {
  dplyr::case_when(
    x == "cluster1" ~ "Cluster 1 vs Non-CR",
    x == "cluster2" ~ "Cluster 2 vs Non-CR",
    TRUE ~ x
  )
}

message("[1] Loading current ADNI projected group table")

if (file.exists(adni_projection_file)) {
  adni <- read.csv(adni_projection_file, check.names = FALSE)
  projection_source <- adni_projection_file
} else if (file.exists(adni_projection_all_file)) {
  warning("ADpatho projection table not found. Falling back to all table and filtering ADpatho == 'AD'.")
  adni <- read.csv(adni_projection_all_file, check.names = FALSE)
  projection_source <- adni_projection_all_file
  if ("ADpatho" %in% colnames(adni)) {
    adni <- adni %>% filter(ADpatho == "AD")
  } else {
    stop("Fallback all table does not contain ADpatho column.")
  }
} else {
  stop("Neither ADpatho nor all ADNI projection table was found.")
}

required_cols <- c("RID", "resilience", "predicted_cluster_like")
missing_cols <- setdiff(required_cols, colnames(adni))
if (length(missing_cols) > 0) {
  stop("ADNI projection table is missing required columns: ", paste(missing_cols, collapse = ", "))
}

adni <- adni %>%
  mutate(
    RID = as.integer(RID),
    resilience = as.character(resilience),
    predicted_cluster_like = as.character(predicted_cluster_like),
    group = case_when(
      resilience == "Low" ~ "non_resilience",
      resilience == "High" & predicted_cluster_like %in% c("cluster1", "cluster2") ~ predicted_cluster_like,
      TRUE ~ NA_character_
    ),
    group = factor(group, levels = group_levels)
  ) %>%
  filter(!is.na(group), group %in% group_levels) %>%
  select(RID, group, everything())

message("    Projection source: ", projection_source)
message("    Samples with current groups: ", nrow(adni))
print(table(adni$group, useNA = "ifany"))

if ("DIAGNOSIS" %in% colnames(adni)) {
  adni$DIAGNOSIS <- safe_factor(adni$DIAGNOSIS)
} else if ("DX" %in% colnames(adni)) {
  adni$DIAGNOSIS <- safe_factor(adni$DX)
} else {
  adni$DIAGNOSIS <- factor(NA_character_)
}

if ("MMSCORE" %in% colnames(adni)) {
  adni$BASELINE_MMSE <- clean_numeric(adni$MMSCORE)
} else if ("MMSE" %in% colnames(adni)) {
  adni$BASELINE_MMSE <- clean_numeric(adni$MMSE)
} else {
  adni$BASELINE_MMSE <- NA_real_
}

message("[2] Adding demographics and APOE")

if (file.exists(ptdemog_file)) {
  ptdemog <- read.csv(ptdemog_file, check.names = FALSE)
  ptdemog_sub <- ptdemog %>%
    mutate(
      RID = as.integer(RID),
      PTEDUCAT = if ("PTEDUCAT" %in% colnames(.)) clean_numeric(PTEDUCAT) else NA_real_,
      PTDOBYY = if ("PTDOBYY" %in% colnames(.)) clean_numeric(PTDOBYY) else NA_real_,
      PTGENDER = if ("PTGENDER" %in% colnames(.)) PTGENDER else NA
    ) %>%
    mutate(
      visit_rank = match(VISCODE2, c("sc", "bl", "init", "f")),
      visit_rank = ifelse(is.na(visit_rank), 999, visit_rank)
    ) %>%
    arrange(RID, visit_rank) %>%
    group_by(RID) %>%
    slice(1) %>%
    ungroup() %>%
    select(any_of(c("RID", "PTID", "PTGENDER", "PTEDUCAT", "PTDOB", "PTDOBYY", "VISDATE", "VISCODE", "VISCODE2")))

  adni <- adni %>% left_join(ptdemog_sub, by = "RID", suffix = c("", ".ptdemog"))
} else {
  warning("PTDEMOG file not found: ", ptdemog_file)
}

if ("sex" %in% colnames(adni)) {
  sex_proj <- harmonize_sex(adni$sex)
} else {
  sex_proj <- rep(NA_character_, nrow(adni))
}
if ("PTGENDER" %in% colnames(adni)) {
  sex_pt <- harmonize_sex(adni$PTGENDER)
} else {
  sex_pt <- rep(NA_character_, nrow(adni))
}
adni$SEX <- ifelse(!is.na(sex_proj), sex_proj, sex_pt)
adni$SEX <- factor(adni$SEX, levels = c("Male", "Female"))

if ("PTEDUCAT" %in% colnames(adni)) {
  adni$PTEDUCAT_clean <- clean_numeric(adni$PTEDUCAT)
} else if ("education" %in% colnames(adni)) {
  adni$PTEDUCAT_clean <- clean_numeric(adni$education)
} else {
  adni$PTEDUCAT_clean <- NA_real_
}

if ("age_at_visit" %in% colnames(adni)) {
  adni$AGE_AT_PROJECTION <- clean_numeric(adni$age_at_visit)
} else if ("AGE" %in% colnames(adni)) {
  adni$AGE_AT_PROJECTION <- clean_numeric(adni$AGE)
} else {
  adni$AGE_AT_PROJECTION <- NA_real_
}

if ("PTDOBYY" %in% colnames(adni) && "VISDATE" %in% colnames(adni)) {
  adni$VISDATE_parsed <- suppressWarnings(lubridate::parse_date_time(adni$VISDATE, orders = c("Y-m-d", "Y/m/d", "m/d/Y", "d/m/Y")))
  adni$AGE_FROM_PTDOBYY <- lubridate::year(adni$VISDATE_parsed) - clean_numeric(adni$PTDOBYY)
} else {
  adni$AGE_FROM_PTDOBYY <- NA_real_
}

adni$AGE_AT_BASELINE <- dplyr::case_when(
  !is.na(adni$AGE_AT_PROJECTION) ~ adni$AGE_AT_PROJECTION,
  !is.na(adni$AGE_FROM_PTDOBYY) ~ adni$AGE_FROM_PTDOBYY,
  TRUE ~ NA_real_
)

if ("APOE_e4_carrier" %in% colnames(adni)) {
  x <- as.character(adni$APOE_e4_carrier)
  adni$APOE_e4_carrier <- dplyr::case_when(
    x %in% c("1", "Carrier", "carrier", "E4", "e4") ~ "Carrier",
    x %in% c("0", "Non-carrier", "Noncarrier", "non-carrier", "noncarrier") ~ "Non-carrier",
    TRUE ~ NA_character_
  )
} else {
  adni$APOE_e4_carrier <- NA_character_
}

if (file.exists(apoe_file)) {
  apoe <- read.csv(apoe_file, check.names = FALSE)
  if (all(c("RID", "GENOTYPE") %in% colnames(apoe))) {
    apoe_sub <- apoe %>%
      mutate(RID = as.integer(RID)) %>%
      select(RID, GENOTYPE) %>%
      distinct(RID, .keep_all = TRUE)
    adni <- adni %>% left_join(apoe_sub, by = "RID")
    apoe_file_carrier <- make_apoe4(adni$GENOTYPE)
    adni$APOE_e4_carrier <- ifelse(is.na(adni$APOE_e4_carrier), apoe_file_carrier, adni$APOE_e4_carrier)
  } else {
    warning("APOE file does not contain RID and GENOTYPE.")
  }
} else {
  warning("APOE file not found: ", apoe_file)
}

adni$APOE_e4_carrier <- factor(adni$APOE_e4_carrier, levels = c("Non-carrier", "Carrier"))
adni$DIAGNOSIS <- droplevels(safe_factor(adni$DIAGNOSIS))

write.csv(adni, file.path(table_dir, "ADNI_current_projected_groups_with_demographics_APOE.csv"), row.names = FALSE, quote = FALSE)

message("[3] Loading longitudinal MMSE")

if (!file.exists(mmse_file)) stop("MMSE file not found: ", mmse_file)
mmse_raw <- read.csv(mmse_file, check.names = FALSE)
if (!all(c("RID", "VISCODE2", "MMSCORE") %in% colnames(mmse_raw))) {
  stop("MMSE file must contain RID, VISCODE2, and MMSCORE.")
}

mmse_long <- mmse_raw %>%
  mutate(
    RID = as.integer(RID),
    VISCODE2 = as.character(VISCODE2),
    MMSCORE = clean_numeric(MMSCORE),
    month = parse_visit_month(VISCODE2)
  ) %>%
  filter(VISCODE2 != "f") %>%
  filter(!is.na(RID), !is.na(MMSCORE), !is.na(month)) %>%
  group_by(RID, month) %>%
  arrange(RID, month) %>%
  summarise(
    MMSCORE = mean(MMSCORE, na.rm = TRUE),
    VISCODE2 = first(VISCODE2),
    .groups = "drop"
  ) %>%
  left_join(
    adni %>% select(RID, group, AGE_AT_BASELINE, SEX, PTEDUCAT_clean, APOE_e4_carrier, BASELINE_MMSE, DIAGNOSIS),
    by = "RID"
  ) %>%
  filter(!is.na(group)) %>%
  mutate(
    group = factor(group, levels = group_levels),
    group_label = factor(group_labels[as.character(group)], levels = group_labels[group_levels]),
    time_years = month / 12
  )

write.csv(mmse_long, file.path(table_dir, "ADNI_current_groups_MMSE_long.csv"), row.names = FALSE, quote = FALSE)

message("    Longitudinal MMSE rows: ", nrow(mmse_long))
message("    Subjects with MMSE: ", length(unique(mmse_long$RID)))
print(table(mmse_long$group, useNA = "ifany"))
print(table(mmse_long$VISCODE2, useNA = "ifany"))

message("[4] Creating time-to-MMSE-decline event data")

event_data <- mmse_long %>%
  arrange(RID, month) %>%
  group_by(RID) %>%
  mutate(event_at_visit = MMSCORE < mmse_event_threshold) %>%
  summarise(
    time = ifelse(any(event_at_visit, na.rm = TRUE), month[event_at_visit][1], max(month, na.rm = TRUE)),
    event = as.numeric(any(event_at_visit, na.rm = TRUE)),
    first_month = min(month, na.rm = TRUE),
    last_month = max(month, na.rm = TRUE),
    n_mmse_visits = dplyr::n(),
    group = first(group),
    AGE_AT_BASELINE = first(AGE_AT_BASELINE),
    SEX = first(SEX),
    PTEDUCAT_clean = first(PTEDUCAT_clean),
    APOE_e4_carrier = first(APOE_e4_carrier),
    BASELINE_MMSE = first(BASELINE_MMSE),
    DIAGNOSIS = first(DIAGNOSIS),
    .groups = "drop"
  ) %>%
  filter(!is.na(time), time >= 0, !is.na(group)) %>%
  mutate(
    group = factor(group, levels = group_levels),
    group_label = factor(group_labels[as.character(group)], levels = group_labels[group_levels])
  )

write.csv(event_data, file.path(table_dir, "ADNI_current_groups_MMSE_decline_event_data.csv"), row.names = FALSE, quote = FALSE)

event_data_anon <- event_data %>%
  mutate(anon_id = paste0("ID_", seq_len(n()))) %>%
  select(anon_id, time, event, group, n_mmse_visits)
write.csv(event_data_anon, file.path(table_dir, "ADNI_current_groups_MMSE_decline_event_data_anonymized.csv"), row.names = FALSE, quote = FALSE)

message("    Event data subjects: ", nrow(event_data))
print(table(event_data$group, useNA = "ifany"))
print(table(event_data$event, useNA = "ifany"))

message("[5] Running KM visualization and log-rank tests")

surv_obj <- survival::Surv(time = event_data$time, event = event_data$event)
km_fit <- survival::survfit(surv_obj ~ group, data = event_data)

logrank_overall <- survival::survdiff(survival::Surv(time, event) ~ group, data = event_data)
logrank_overall_df <- data.frame(
  test = "overall_logrank",
  chisq = logrank_overall$chisq,
  df = length(logrank_overall$n) - 1,
  p.value = pchisq(logrank_overall$chisq, df = length(logrank_overall$n) - 1, lower.tail = FALSE),
  stringsAsFactors = FALSE
) %>%
  mutate(p_stars = vapply(p.value, p_to_stars, character(1)))
write.csv(logrank_overall_df, file.path(table_dir, "ADNI_current_groups_MMSE_logrank_overall.csv"), row.names = FALSE, quote = FALSE)

pair_list <- combn(group_levels, 2, simplify = FALSE)
pairwise_logrank <- lapply(pair_list, function(pair) {
  d <- event_data %>% filter(group %in% pair) %>% droplevels()
  test <- survival::survdiff(survival::Surv(time, event) ~ group, data = d)
  chisq <- test$chisq
  df_test <- length(test$n) - 1
  p_val <- pchisq(chisq, df = df_test, lower.tail = FALSE)
  data.frame(
    group1 = pair[1],
    group2 = pair[2],
    chisq = chisq,
    df = df_test,
    p.value = p_val,
    stringsAsFactors = FALSE
  )
}) %>%
  bind_rows() %>%
  mutate(
    p_adj_bonferroni = p.adjust(p.value, method = "bonferroni"),
    p_adj_fdr = p.adjust(p.value, method = "BH"),
    p_stars = vapply(p.value, p_to_stars, character(1)),
    q_stars = vapply(p_adj_fdr, p_to_stars, character(1))
  )
write.csv(pairwise_logrank, file.path(table_dir, "ADNI_current_groups_MMSE_pairwise_logrank_tests_secondary.csv"), row.names = FALSE, quote = FALSE)

km_palette <- group_colors[group_levels]
names(km_palette) <- group_levels

surv_plot <- survminer::ggsurvplot(
  km_fit,
  data = event_data,
  risk.table = TRUE,
  conf.int = TRUE,
  pval = TRUE,
  xlab = "Months",
  ylab = paste0("Probability of maintaining MMSE >= ", mmse_event_threshold),
  palette = km_palette,
  legend.title = "Projected group",
  legend.labs = group_labels[group_levels],
  title = paste0("Time to MMSE < ", mmse_event_threshold, " by current projected CR-subtype group"),
  risk.table.height = 0.28,
  ggtheme = theme_bw(base_size = base_font_size)
)

svg(file.path(plot_dir, "ADNI_current_groups_KM_MMSE_decline_supplementary.svg"), width = km_width, height = km_height + risk_table_height)
print(surv_plot)
dev.off()

pdf(file.path(plot_dir, "ADNI_current_groups_KM_MMSE_decline_supplementary.pdf"), width = km_width, height = km_height + risk_table_height)
print(surv_plot)
dev.off()

png(file.path(plot_dir, "ADNI_current_groups_KM_MMSE_decline_supplementary.png"), width = km_width, height = km_height + risk_table_height, units = "in", res = 300)
print(surv_plot)
dev.off()

ggsave(file.path(plot_dir, "ADNI_current_groups_KM_MMSE_decline_supplementary_curve_only.svg"), surv_plot$plot, width = km_width, height = km_height, units = "in")
ggsave(file.path(plot_dir, "ADNI_current_groups_KM_MMSE_decline_supplementary_curve_only.pdf"), surv_plot$plot, width = km_width, height = km_height, units = "in")
ggsave(file.path(plot_dir, "ADNI_current_groups_KM_MMSE_decline_supplementary_curve_only.png"), surv_plot$plot, width = km_width, height = km_height, units = "in", dpi = 300)

ggsave(file.path(plot_dir, "ADNI_current_groups_KM_MMSE_decline_supplementary_risk_table.svg"), surv_plot$table, width = km_width, height = risk_table_height, units = "in")
ggsave(file.path(plot_dir, "ADNI_current_groups_KM_MMSE_decline_supplementary_risk_table.pdf"), surv_plot$table, width = km_width, height = risk_table_height, units = "in")
ggsave(file.path(plot_dir, "ADNI_current_groups_KM_MMSE_decline_supplementary_risk_table.png"), surv_plot$table, width = km_width, height = risk_table_height, units = "in", dpi = 300)

message("[6] Running primary Cox proportional hazards models")

run_cox_model <- function(data, covars, model_name) {
  d <- data
  d$group <- relevel(factor(d$group, levels = group_levels), ref = reference_group)
  covars <- keep_usable_covariates(d, covars)

  use_cols <- unique(c("time", "event", "group", covars))
  d <- d[, use_cols, drop = FALSE]

  for (cv in covars) {
    if (cv %in% c("SEX", "APOE_e4_carrier", "DIAGNOSIS")) {
      d[[cv]] <- droplevels(safe_factor(d[[cv]]))
    } else {
      d[[cv]] <- clean_numeric(d[[cv]])
    }
  }
  d <- d[complete.cases(d), , drop = FALSE]
  d$group <- relevel(droplevels(d$group), ref = reference_group)

  if (nrow(d) < 10 || length(unique(d$event)) < 2 || nlevels(d$group) < 2) {
    return(list(
      fit = NULL,
      data = d,
      result = data.frame(
        model = model_name,
        term = NA_character_,
        HR = NA_real_,
        lower_95CI = NA_real_,
        upper_95CI = NA_real_,
        statistic = NA_real_,
        p.value = NA_real_,
        n = nrow(d),
        events = sum(d$event, na.rm = TRUE),
        formula = NA_character_,
        covariates = paste(covars, collapse = ";"),
        note = "insufficient_data",
        stringsAsFactors = FALSE
      )
    ))
  }

  rhs <- c("group", covars)
  form_txt <- paste("Surv(time, event) ~", paste(rhs, collapse = " + "))
  fit <- tryCatch(coxph(as.formula(form_txt), data = d), error = function(e) NULL)

  if (is.null(fit)) {
    return(list(
      fit = NULL,
      data = d,
      result = data.frame(
        model = model_name,
        term = NA_character_,
        HR = NA_real_,
        lower_95CI = NA_real_,
        upper_95CI = NA_real_,
        statistic = NA_real_,
        p.value = NA_real_,
        n = nrow(d),
        events = sum(d$event, na.rm = TRUE),
        formula = form_txt,
        covariates = paste(covars, collapse = ";"),
        note = "model_failed",
        stringsAsFactors = FALSE
      )
    ))
  }

  td <- broom::tidy(fit, exponentiate = TRUE, conf.int = TRUE)
  out <- td %>%
    filter(grepl("^group", term)) %>%
    transmute(
      model = model_name,
      term,
      contrast = case_when(
        term == "groupcluster1" ~ "cluster1_vs_non_resilience",
        term == "groupcluster2" ~ "cluster2_vs_non_resilience",
        TRUE ~ term
      ),
      HR = estimate,
      lower_95CI = conf.low,
      upper_95CI = conf.high,
      statistic = statistic,
      p.value = p.value,
      n = nrow(d),
      events = sum(d$event, na.rm = TRUE),
      formula = form_txt,
      covariates = paste(covars, collapse = ";"),
      note = ""
    )

  list(fit = fit, data = d, result = out)
}

cox_primary <- run_cox_model(event_data, cox_covariates_primary, "primary_no_MMSE_no_CDR_no_diagnosis")
cox_results <- cox_primary$result %>%
  mutate(
    p_adj_BH_within_model = p.adjust(p.value, method = "BH"),
    p_stars = vapply(p.value, p_to_stars, character(1)),
    q_stars = vapply(p_adj_BH_within_model, p_to_stars, character(1))
  )

write.csv(cox_results, file.path(table_dir, "ADNI_current_groups_MMSE_Cox_primary_HR.csv"), row.names = FALSE, quote = FALSE)

if (!is.null(cox_primary$fit)) {
  sink(file.path(table_dir, "ADNI_current_groups_MMSE_Cox_primary_summary.txt"))
  print(summary(cox_primary$fit))
  sink()

  zph <- tryCatch(cox.zph(cox_primary$fit), error = function(e) NULL)
  if (!is.null(zph)) {
    zph_df <- as.data.frame(zph$table) %>%
      tibble::rownames_to_column("term")
    write.csv(zph_df, file.path(table_dir, "ADNI_current_groups_MMSE_Cox_primary_PH_test_cox_zph.csv"), row.names = FALSE, quote = FALSE)
  }
}

if (run_optional_cognitive_diagnosis_sensitivity) {
  cox_sens_covars <- unique(c(cox_covariates_primary, sensitivity_covariates_extra))
  cox_sens <- run_cox_model(event_data, cox_sens_covars, "sensitivity_plus_baseline_MMSE_diagnosis")
  cox_sens_results <- cox_sens$result %>%
    mutate(
      p_adj_BH_within_model = p.adjust(p.value, method = "BH"),
      p_stars = vapply(p.value, p_to_stars, character(1)),
      q_stars = vapply(p_adj_BH_within_model, p_to_stars, character(1))
    )
  write.csv(cox_sens_results, file.path(table_dir, "ADNI_current_groups_MMSE_Cox_sensitivity_plus_MMSE_diagnosis_HR.csv"), row.names = FALSE, quote = FALSE)
}

if (nrow(cox_results) > 0 && any(!is.na(cox_results$HR))) {
  cox_plot_df <- cox_results %>%
    mutate(
      contrast_label = case_when(
        contrast == "cluster1_vs_non_resilience" ~ "Cluster 1 vs Non-CR",
        contrast == "cluster2_vs_non_resilience" ~ "Cluster 2 vs Non-CR",
        TRUE ~ contrast
      ),
      contrast_label = factor(contrast_label, levels = c("Cluster 2 vs Non-CR", "Cluster 1 vs Non-CR")),
      significant = !is.na(p_adj_BH_within_model) & p_adj_BH_within_model < 0.05
    )

  cox_colors <- c(
    "Cluster 1 vs Non-CR" = group_colors[["cluster1"]],
    "Cluster 2 vs Non-CR" = group_colors[["cluster2"]]
  )

  p_cox <- ggplot(cox_plot_df, aes(x = HR, y = contrast_label, color = contrast_label, shape = significant)) +
    geom_vline(xintercept = 1, linetype = "dashed", color = "gray40", linewidth = 0.5) +
    geom_errorbarh(aes(xmin = lower_95CI, xmax = upper_95CI), height = 0.18, linewidth = 0.8, na.rm = TRUE) +
    geom_point(size = 3.4, stroke = 0.9, na.rm = TRUE) +
    scale_x_log10() +
    scale_color_manual(values = cox_colors, name = "Projected group contrast") +
    scale_shape_manual(values = c("TRUE" = 16, "FALSE" = 1), labels = c("TRUE" = "BH-FDR q < 0.05", "FALSE" = "q >= 0.05"), name = "Statistical significance") +
    labs(
      title = paste0("Cox model: time to MMSE < ", mmse_event_threshold),
      subtitle = "Primary model adjusted for age, sex, education, and APOE e4 carrier status",
      x = "Hazard ratio for cognitive decline event",
      y = NULL
    ) +
    theme_bw(base_size = base_font_size) +
    theme(
      plot.title = element_text(face = "bold"),
      legend.position = "right",
      panel.grid.minor = element_blank()
    )

  ggsave(file.path(plot_dir, "ADNI_current_groups_MMSE_Cox_primary_HR_forest.svg"), p_cox, width = forest_width, height = forest_height, units = "in")
  ggsave(file.path(plot_dir, "ADNI_current_groups_MMSE_Cox_primary_HR_forest.pdf"), p_cox, width = forest_width, height = forest_height, units = "in")
  ggsave(file.path(plot_dir, "ADNI_current_groups_MMSE_Cox_primary_HR_forest.png"), p_cox, width = forest_width, height = forest_height, units = "in", dpi = 300)
}

message("[7] Running longitudinal MMSE LMM")

run_lmm_model <- function(data, covars, model_name) {
  d <- data %>%
    mutate(
      group = relevel(factor(group, levels = group_levels), ref = reference_group),
      time_years = clean_numeric(time_years),
      MMSCORE = clean_numeric(MMSCORE)
    )

  covars <- keep_usable_covariates(d, covars)
  use_cols <- unique(c("RID", "MMSCORE", "time_years", "group", covars))
  d <- d[, use_cols, drop = FALSE]

  for (cv in covars) {
    if (cv %in% c("SEX", "APOE_e4_carrier", "DIAGNOSIS")) {
      d[[cv]] <- droplevels(safe_factor(d[[cv]]))
    } else {
      d[[cv]] <- clean_numeric(d[[cv]])
    }
  }

  d <- d[complete.cases(d), , drop = FALSE]
  d$group <- relevel(droplevels(d$group), ref = reference_group)

  n_subjects <- length(unique(d$RID))
  if (nrow(d) < 20 || n_subjects < 10 || nlevels(d$group) < 2 || length(unique(d$time_years)) < 2) {
    return(list(
      fit = NULL,
      data = d,
      result = data.frame(
        model = model_name,
        term = NA_character_,
        estimate = NA_real_,
        SE = NA_real_,
        df = NA_real_,
        statistic = NA_real_,
        p.value = NA_real_,
        n_obs = nrow(d),
        n_subjects = n_subjects,
        formula = NA_character_,
        covariates = paste(covars, collapse = ";"),
        note = "insufficient_data",
        stringsAsFactors = FALSE
      )
    ))
  }

  rhs_fixed <- c("group * time_years", covars)
  form_txt_rs <- paste("MMSCORE ~", paste(rhs_fixed, collapse = " + "), "+ (time_years | RID)")
  form_txt_ri <- paste("MMSCORE ~", paste(rhs_fixed, collapse = " + "), "+ (1 | RID)")

  fit <- tryCatch(lmer(as.formula(form_txt_rs), data = d, REML = FALSE), error = function(e) NULL)
  random_structure <- "random_intercept_slope"

  if (is.null(fit) || isSingular(fit, tol = 1e-4)) {
    fit2 <- tryCatch(lmer(as.formula(form_txt_ri), data = d, REML = FALSE), error = function(e) NULL)
    if (!is.null(fit2)) {
      fit <- fit2
      random_structure <- "random_intercept_only"
    }
  }

  if (is.null(fit)) {
    return(list(
      fit = NULL,
      data = d,
      result = data.frame(
        model = model_name,
        term = NA_character_,
        estimate = NA_real_,
        SE = NA_real_,
        df = NA_real_,
        statistic = NA_real_,
        p.value = NA_real_,
        n_obs = nrow(d),
        n_subjects = n_subjects,
        formula = form_txt_rs,
        covariates = paste(covars, collapse = ";"),
        note = "model_failed",
        stringsAsFactors = FALSE
      )
    ))
  }

  td <- broom.mixed::tidy(fit, effects = "fixed", conf.int = TRUE)
  out <- td %>%
    transmute(
      model = model_name,
      term,
      estimate,
      SE = std.error,
      df = df,
      statistic = statistic,
      p.value = p.value,
      lower_95CI = conf.low,
      upper_95CI = conf.high,
      n_obs = nrow(d),
      n_subjects = n_subjects,
      formula = ifelse(random_structure == "random_intercept_slope", form_txt_rs, form_txt_ri),
      random_structure = random_structure,
      covariates = paste(covars, collapse = ";"),
      note = ""
    )

  list(fit = fit, data = d, result = out)
}

lmm_primary <- run_lmm_model(mmse_long, lmm_covariates_primary, "primary_no_MMSE_no_CDR_no_diagnosis")
lmm_results <- lmm_primary$result %>%
  mutate(
    p_adj_BH_fixed_terms = p.adjust(p.value, method = "BH"),
    p_stars = vapply(p.value, p_to_stars, character(1)),
    q_stars = vapply(p_adj_BH_fixed_terms, p_to_stars, character(1))
  )
write.csv(lmm_results, file.path(table_dir, "ADNI_current_groups_MMSE_LMM_primary_fixed_effects.csv"), row.names = FALSE, quote = FALSE)

if (!is.null(lmm_primary$fit)) {
  sink(file.path(table_dir, "ADNI_current_groups_MMSE_LMM_primary_summary.txt"))
  print(summary(lmm_primary$fit))
  print(anova(lmm_primary$fit))
  sink()

  anova_lmm <- as.data.frame(anova(lmm_primary$fit)) %>% tibble::rownames_to_column("term")
  write.csv(anova_lmm, file.path(table_dir, "ADNI_current_groups_MMSE_LMM_primary_anova_terms.csv"), row.names = FALSE, quote = FALSE)

  emm_trends <- tryCatch(emmeans::emtrends(lmm_primary$fit, ~ group, var = "time_years"), error = function(e) NULL)
  if (!is.null(emm_trends)) {
    slopes <- summary(emm_trends, infer = c(TRUE, TRUE)) %>%
      as.data.frame() %>%
      mutate(
        group = as.character(group),
        group_label = group_labels[group],
        p_stars = vapply(p.value, p_to_stars, character(1))
      )
    write.csv(slopes, file.path(table_dir, "ADNI_current_groups_MMSE_LMM_estimated_annual_slopes_by_group.csv"), row.names = FALSE, quote = FALSE)

    slope_contrasts <- summary(contrast(emm_trends, method = "pairwise", adjust = "none"), infer = c(TRUE, TRUE)) %>%
      as.data.frame() %>%
      mutate(
        FDR_BH = p.adjust(p.value, method = "BH"),
        p_stars = vapply(p.value, p_to_stars, character(1)),
        q_stars = vapply(FDR_BH, p_to_stars, character(1))
      )
    write.csv(slope_contrasts, file.path(table_dir, "ADNI_current_groups_MMSE_LMM_pairwise_annual_slope_differences.csv"), row.names = FALSE, quote = FALSE)
  }
}

plot_long <- mmse_long %>%
  filter(!is.na(MMSCORE), !is.na(time_years), !is.na(group)) %>%
  mutate(group_label = factor(group_labels[as.character(group)], levels = group_labels[group_levels]))

p_lmm_raw <- ggplot(plot_long, aes(x = time_years, y = MMSCORE, color = group_label, fill = group_label)) +
  geom_point(alpha = 0.18, size = 0.8, show.legend = FALSE) +
  geom_smooth(method = "loess", se = TRUE, linewidth = 1.1, span = 0.75) +
  scale_color_manual(values = setNames(group_colors[group_levels], group_labels[group_levels]), name = "Projected group") +
  scale_fill_manual(values = setNames(group_colors[group_levels], group_labels[group_levels]), name = "Projected group") +
  labs(
    title = "Longitudinal MMSE trajectories by current projected CR-subtype group",
    subtitle = "Visualization only; primary inference is from LMM group x time interaction",
    x = "Years from baseline/screening visit",
    y = "MMSE"
  ) +
  theme_bw(base_size = base_font_size) +
  theme(
    plot.title = element_text(face = "bold"),
    legend.position = "right",
    panel.grid.minor = element_blank()
  )

ggsave(file.path(plot_dir, "ADNI_current_groups_MMSE_longitudinal_LOESS_visualization.svg"), p_lmm_raw, width = lmm_width, height = lmm_height, units = "in")
ggsave(file.path(plot_dir, "ADNI_current_groups_MMSE_longitudinal_LOESS_visualization.pdf"), p_lmm_raw, width = lmm_width, height = lmm_height, units = "in")
ggsave(file.path(plot_dir, "ADNI_current_groups_MMSE_longitudinal_LOESS_visualization.png"), p_lmm_raw, width = lmm_width, height = lmm_height, units = "in", dpi = 300)

slope_file <- file.path(table_dir, "ADNI_current_groups_MMSE_LMM_estimated_annual_slopes_by_group.csv")
if (file.exists(slope_file)) {
  slopes <- read.csv(slope_file, check.names = FALSE) %>%
    mutate(
      group = factor(group, levels = group_levels),
      group_label = factor(group_labels[as.character(group)], levels = group_labels[group_levels])
    )
  slope_color_map <- setNames(group_colors[group_levels], group_labels[group_levels])

  slope_col <- first_present(c("time_years.trend", "trend"), colnames(slopes))
  if (!is.na(slope_col)) {
    slopes$annual_slope <- slopes[[slope_col]]
    p_slope <- ggplot(slopes, aes(x = annual_slope, y = group_label, color = group_label)) +
      geom_vline(xintercept = 0, linetype = "dashed", color = "gray40", linewidth = 0.5) +
      geom_errorbarh(aes(xmin = lower.CL, xmax = upper.CL), height = 0.18, linewidth = 0.8) +
      geom_point(size = 3.5) +
      scale_color_manual(values = slope_color_map, name = "Projected group") +
      labs(
        title = "Estimated annual MMSE slope by projected group",
        subtitle = "From covariate-adjusted LMM",
        x = "Estimated annual change in MMSE",
        y = NULL
      ) +
      theme_bw(base_size = base_font_size) +
      theme(
        plot.title = element_text(face = "bold"),
        legend.position = "right",
        panel.grid.minor = element_blank()
      )
    ggsave(file.path(plot_dir, "ADNI_current_groups_MMSE_LMM_estimated_annual_slopes.svg"), p_slope, width = forest_width, height = forest_height, units = "in")
    ggsave(file.path(plot_dir, "ADNI_current_groups_MMSE_LMM_estimated_annual_slopes.pdf"), p_slope, width = forest_width, height = forest_height, units = "in")
    ggsave(file.path(plot_dir, "ADNI_current_groups_MMSE_LMM_estimated_annual_slopes.png"), p_slope, width = forest_width, height = forest_height, units = "in", dpi = 300)
  }
}

message("[8] Writing summary tables")

baseline_summary <- event_data %>%
  group_by(group) %>%
  summarise(
    n = n(),
    events = sum(event, na.rm = TRUE),
    event_rate = events / n,
    median_followup_months = median(time, na.rm = TRUE),
    mean_age = mean(AGE_AT_BASELINE, na.rm = TRUE),
    sd_age = sd(AGE_AT_BASELINE, na.rm = TRUE),
    mean_education = mean(PTEDUCAT_clean, na.rm = TRUE),
    sd_education = sd(PTEDUCAT_clean, na.rm = TRUE),
    apoe4_carrier_n = sum(APOE_e4_carrier == "Carrier", na.rm = TRUE),
    apoe4_carrier_prop = apoe4_carrier_n / sum(!is.na(APOE_e4_carrier)),
    .groups = "drop"
  )
write.csv(baseline_summary, file.path(table_dir, "ADNI_current_groups_MMSE_event_baseline_summary_by_group.csv"), row.names = FALSE, quote = FALSE)

mmse_visit_summary <- mmse_long %>%
  group_by(group, month, VISCODE2) %>%
  summarise(
    n = n(),
    mean_MMSE = mean(MMSCORE, na.rm = TRUE),
    sd_MMSE = sd(MMSCORE, na.rm = TRUE),
    median_MMSE = median(MMSCORE, na.rm = TRUE),
    q1_MMSE = quantile(MMSCORE, 0.25, na.rm = TRUE),
    q3_MMSE = quantile(MMSCORE, 0.75, na.rm = TRUE),
    .groups = "drop"
  )
write.csv(mmse_visit_summary, file.path(table_dir, "ADNI_current_groups_MMSE_visit_summary_by_group.csv"), row.names = FALSE, quote = FALSE)

settings <- data.frame(
  setting = c(
    "projection_source",
    "mmse_file",
    "ptdemog_file",
    "apoe_file",
    "group_definition",
    "event_definition",
    "primary_cox_model",
    "primary_lmm_model",
    "primary_covariates",
    "cognition_diagnosis_covariates_excluded_from_primary",
    "km_logrank_role"
  ),
  value = c(
    projection_source,
    mmse_file,
    ptdemog_file,
    apoe_file,
    "High resilience: predicted_cluster_like; Low resilience: non_resilience",
    paste0("First observed MMSE < ", mmse_event_threshold, "; censored at last available MMSE visit"),
    "coxph(Surv(time, event) ~ group + AGE_AT_BASELINE + SEX + PTEDUCAT_clean + APOE_e4_carrier)",
    "lmer(MMSCORE ~ group * time_years + AGE_AT_BASELINE + SEX + PTEDUCAT_clean + APOE_e4_carrier + (time_years | RID)); falls back to random intercept if singular",
    paste(cox_covariates_primary, collapse = ";"),
    "BASELINE_MMSE;baseline_CDR;DIAGNOSIS",
    "Kaplan-Meier and log-rank are secondary visualization/supplementary analyses"
  ),
  stringsAsFactors = FALSE
)
write.csv(settings, file.path(table_dir, "ADNI_current_groups_MMSE_survival_LMM_settings.csv"), row.names = FALSE, quote = FALSE)

sink(file.path(outdir, "sessionInfo_ADNI_current_groups_MMSE_survival_LMM.txt"))
print(sessionInfo())
sink()

message("Done. Results saved in: ", outdir)
