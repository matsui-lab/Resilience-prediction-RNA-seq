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

outdir <- "out/ADNI/projected_rosmap_msbb_22gene_signature/mmse_current_projected_groups_cox_no_baselineMMSE_pairwise"
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

cox_covariates_primary <- c(
  "AGE_AT_BASELINE",
  "SEX",
  "PTEDUCAT_clean",
  "APOE_e4_carrier"
)

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

forest_width <- 7.5
forest_height <- 4.5
km_width <- 8.5
km_height <- 5.0
risk_table_height <- 2.0
base_font_size <- 18

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(data.table)
  library(lubridate)
  library(survival)
  library(survminer)
  library(tibble)
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
  if (is.na(p)) return("ns")
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
    x == "cluster1_vs_non_resilience" ~ "Cluster 1 vs Non-CR",
    x == "cluster2_vs_non_resilience" ~ "Cluster 2 vs Non-CR",
    x == "cluster1_vs_cluster2" ~ "Cluster 1 vs Cluster 2",
    TRUE ~ x
  )
}

make_pairwise_cox_contrasts <- function(fit, data_n, event_n, model_name, formula_text, covariates_text) {
  beta <- stats::coef(fit)
  vc <- stats::vcov(fit)
  out <- list()

  get_contrast <- function(name, L) {
    est_log <- sum(L * beta[names(L)], na.rm = TRUE)
    vc_sub <- vc[names(L), names(L), drop = FALSE]
    se_log <- sqrt(as.numeric(t(L) %*% vc_sub %*% L))
    z <- est_log / se_log
    p <- 2 * stats::pnorm(abs(z), lower.tail = FALSE)
    data.frame(
      model = model_name,
      contrast = name,
      log_HR = est_log,
      SE_log_HR = se_log,
      HR = exp(est_log),
      lower_95CI = exp(est_log - 1.96 * se_log),
      upper_95CI = exp(est_log + 1.96 * se_log),
      statistic = z,
      p.value = p,
      n = data_n,
      events = event_n,
      formula = formula_text,
      covariates = covariates_text,
      note = "",
      stringsAsFactors = FALSE
    )
  }

  terms_needed <- c("groupcluster1", "groupcluster2")
  has_terms <- terms_needed %in% names(beta)

  if (all(has_terms)) {
    L1 <- c("groupcluster1" = 1, "groupcluster2" = 0)
    L2 <- c("groupcluster1" = 0, "groupcluster2" = 1)
    L12 <- c("groupcluster1" = 1, "groupcluster2" = -1)
    out[[1]] <- get_contrast("cluster1_vs_non_resilience", L1)
    out[[2]] <- get_contrast("cluster2_vs_non_resilience", L2)
    out[[3]] <- get_contrast("cluster1_vs_cluster2", L12)
  } else {
    for (nm in c("cluster1_vs_non_resilience", "cluster2_vs_non_resilience", "cluster1_vs_cluster2")) {
      out[[length(out) + 1]] <- data.frame(
        model = model_name,
        contrast = nm,
        log_HR = NA_real_,
        SE_log_HR = NA_real_,
        HR = NA_real_,
        lower_95CI = NA_real_,
        upper_95CI = NA_real_,
        statistic = NA_real_,
        p.value = NA_real_,
        n = data_n,
        events = event_n,
        formula = formula_text,
        covariates = covariates_text,
        note = "required_group_terms_absent",
        stringsAsFactors = FALSE
      )
    }
  }

  dplyr::bind_rows(out)
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

if ("MMSCORE" %in% colnames(adni)) {
  adni$BASELINE_MMSE_PROJECTION <- clean_numeric(adni$MMSCORE)
} else if ("MMSE" %in% colnames(adni)) {
  adni$BASELINE_MMSE_PROJECTION <- clean_numeric(adni$MMSE)
} else {
  adni$BASELINE_MMSE_PROJECTION <- NA_real_
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

write.csv(adni, file.path(table_dir, "ADNI_current_projected_groups_with_demographics_APOE_for_Cox.csv"), row.names = FALSE, quote = FALSE)

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
    adni %>% select(RID, group, AGE_AT_BASELINE, SEX, PTEDUCAT_clean, APOE_e4_carrier, BASELINE_MMSE_PROJECTION),
    by = "RID"
  ) %>%
  filter(!is.na(group)) %>%
  mutate(
    group = factor(group, levels = group_levels),
    group_label = factor(group_labels[as.character(group)], levels = group_labels[group_levels]),
    time_years = month / 12
  )

write.csv(mmse_long, file.path(table_dir, "ADNI_current_groups_MMSE_long_for_Cox.csv"), row.names = FALSE, quote = FALSE)

message("    Longitudinal MMSE rows: ", nrow(mmse_long))
message("    Subjects with MMSE: ", length(unique(mmse_long$RID)))
print(table(mmse_long$group, useNA = "ifany"))

message("[4] Creating incident time-to-MMSE-decline event data")

event_data_all <- mmse_long %>%
  arrange(RID, month) %>%
  group_by(RID) %>%
  mutate(
    first_observed_month = min(month, na.rm = TRUE),
    baseline_mmse_from_longitudinal = MMSCORE[which.min(month)],
    baseline_month_from_longitudinal = first_observed_month,
    event_at_visit = MMSCORE < mmse_event_threshold
  ) %>%
  summarise(
    baseline_mmse = first(baseline_mmse_from_longitudinal),
    baseline_month = first(baseline_month_from_longitudinal),
    time = ifelse(any(event_at_visit & month > baseline_month, na.rm = TRUE),
                  month[event_at_visit & month > baseline_month][1],
                  max(month, na.rm = TRUE)),
    event = as.numeric(any(event_at_visit & month > baseline_month, na.rm = TRUE)),
    last_month = max(month, na.rm = TRUE),
    n_mmse_visits = dplyr::n(),
    group = first(group),
    AGE_AT_BASELINE = first(AGE_AT_BASELINE),
    SEX = first(SEX),
    PTEDUCAT_clean = first(PTEDUCAT_clean),
    APOE_e4_carrier = first(APOE_e4_carrier),
    .groups = "drop"
  ) %>%
  filter(!is.na(time), time >= 0, !is.na(group)) %>%
  mutate(
    baseline_event = !is.na(baseline_mmse) & baseline_mmse < mmse_event_threshold,
    has_followup_after_baseline = last_month > baseline_month,
    group = factor(group, levels = group_levels),
    group_label = factor(group_labels[as.character(group)], levels = group_labels[group_levels])
  )

event_data <- event_data_all %>%
  filter(!baseline_event, has_followup_after_baseline)

write.csv(event_data_all, file.path(table_dir, "ADNI_current_groups_MMSE_decline_event_data_all_subjects.csv"), row.names = FALSE, quote = FALSE)
write.csv(event_data, file.path(table_dir, "ADNI_current_groups_MMSE_decline_event_data_primary_incident_riskset.csv"), row.names = FALSE, quote = FALSE)

event_data_anon <- event_data %>%
  mutate(anon_id = paste0("ID_", seq_len(n()))) %>%
  select(anon_id, time, event, group, baseline_mmse, n_mmse_visits)
write.csv(event_data_anon, file.path(table_dir, "ADNI_current_groups_MMSE_decline_event_data_primary_incident_anonymized.csv"), row.names = FALSE, quote = FALSE)

message("    All event data subjects: ", nrow(event_data_all))
message("    Primary incident risk-set subjects: ", nrow(event_data))
print(table(event_data$group, useNA = "ifany"))
print(table(event_data$event, useNA = "ifany"))

message("[5] Running Cox proportional hazards model without baseline MMSE covariate")

run_cox_pairwise <- function(data, covars, model_name) {
  d <- data
  d$group <- relevel(factor(d$group, levels = group_levels), ref = reference_group)
  covars <- keep_usable_covariates(d, covars)
  use_cols <- unique(c("time", "event", "group", covars))
  d <- d[, use_cols, drop = FALSE]

  for (cv in covars) {
    if (cv %in% c("SEX", "APOE_e4_carrier")) {
      d[[cv]] <- droplevels(safe_factor(d[[cv]]))
    } else {
      d[[cv]] <- clean_numeric(d[[cv]])
    }
  }

  d <- d[complete.cases(d), , drop = FALSE]
  d$group <- relevel(droplevels(d$group), ref = reference_group)

  if (nrow(d) < 10 || length(unique(d$event)) < 2 || nlevels(d$group) < 3) {
    result <- data.frame(
      model = model_name,
      contrast = c("cluster1_vs_non_resilience", "cluster2_vs_non_resilience", "cluster1_vs_cluster2"),
      log_HR = NA_real_,
      SE_log_HR = NA_real_,
      HR = NA_real_,
      lower_95CI = NA_real_,
      upper_95CI = NA_real_,
      statistic = NA_real_,
      p.value = NA_real_,
      n = nrow(d),
      events = sum(d$event, na.rm = TRUE),
      formula = NA_character_,
      covariates = paste(covars, collapse = ";"),
      note = "insufficient_data_or_missing_group_level",
      stringsAsFactors = FALSE
    )
    return(list(fit = NULL, data = d, result = result))
  }

  rhs <- c("group", covars)
  form_txt <- paste("Surv(time, event) ~", paste(rhs, collapse = " + "))
  fit <- tryCatch(survival::coxph(as.formula(form_txt), data = d), error = function(e) NULL)

  if (is.null(fit)) {
    result <- data.frame(
      model = model_name,
      contrast = c("cluster1_vs_non_resilience", "cluster2_vs_non_resilience", "cluster1_vs_cluster2"),
      log_HR = NA_real_,
      SE_log_HR = NA_real_,
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
    return(list(fit = NULL, data = d, result = result))
  }

  result <- make_pairwise_cox_contrasts(
    fit = fit,
    data_n = nrow(d),
    event_n = sum(d$event, na.rm = TRUE),
    model_name = model_name,
    formula_text = form_txt,
    covariates_text = paste(covars, collapse = ";")
  ) %>%
    mutate(
      p_adj_BH_within_model = p.adjust(p.value, method = "BH"),
      p_stars = vapply(p.value, p_to_stars, character(1)),
      q_stars = vapply(p_adj_BH_within_model, p_to_stars, character(1)),
      contrast_label = contrast_label(contrast)
    )

  list(fit = fit, data = d, result = result)
}

cox_primary <- run_cox_pairwise(event_data, cox_covariates_primary, "primary_no_baselineMMSE_no_CDR_no_diagnosis")
cox_results <- cox_primary$result

write.csv(cox_results, file.path(table_dir, "ADNI_current_groups_MMSE_Cox_no_baselineMMSE_pairwise_HR.csv"), row.names = FALSE, quote = FALSE)
write.csv(cox_results, file.path(table_dir, "Supplementary_Table9_Cox_no_baselineMMSE_pairwise_HR.csv"), row.names = FALSE, quote = FALSE)

if (!is.null(cox_primary$fit)) {
  sink(file.path(table_dir, "ADNI_current_groups_MMSE_Cox_no_baselineMMSE_summary.txt"))
  print(summary(cox_primary$fit))
  sink()

  zph <- tryCatch(survival::cox.zph(cox_primary$fit), error = function(e) NULL)
  if (!is.null(zph)) {
    zph_df <- as.data.frame(zph$table) %>% tibble::rownames_to_column("term")
    write.csv(zph_df, file.path(table_dir, "ADNI_current_groups_MMSE_Cox_no_baselineMMSE_PH_test_cox_zph.csv"), row.names = FALSE, quote = FALSE)
  }
}

message("[6] Drawing Cox HR forest plot")

if (nrow(cox_results) > 0 && any(!is.na(cox_results$HR))) {
  cox_plot_df <- cox_results %>%
    mutate(
      contrast_label = factor(
        contrast_label,
        levels = c("Cluster 1 vs Cluster 2", "Cluster 2 vs Non-CR", "Cluster 1 vs Non-CR")
      ),
      significant = !is.na(p_adj_BH_within_model) & p_adj_BH_within_model < 0.05
    )

  cox_colors <- c(
    "Cluster 1 vs Non-CR" = group_colors[["cluster1"]],
    "Cluster 2 vs Non-CR" = group_colors[["cluster2"]],
    "Cluster 1 vs Cluster 2" = "#4D4D4D"
  )

  p_cox <- ggplot(cox_plot_df, aes(x = HR, y = contrast_label, color = contrast_label, shape = significant)) +
    geom_vline(xintercept = 1, linetype = "dashed", color = "gray40", linewidth = 0.5) +
    geom_errorbarh(aes(xmin = lower_95CI, xmax = upper_95CI), height = 0.18, linewidth = 0.8, na.rm = TRUE) +
    geom_point(size = 3.4, stroke = 0.9, na.rm = TRUE) +
    scale_x_log10() +
    scale_color_manual(values = cox_colors, name = "Projected group contrast") +
    scale_shape_manual(
      values = c("TRUE" = 16, "FALSE" = 1),
      labels = c("TRUE" = "BH-FDR q < 0.05", "FALSE" = "q >= 0.05"),
      name = "Statistical significance"
    ) +
    labs(
      title = paste0("Cox model: time to MMSE < ", mmse_event_threshold),
      subtitle = "Adjusted for age, sex, education, and APOE e4 carrier status; baseline MMSE not included",
      x = "Hazard ratio for incident MMSE decline",
      y = NULL
    ) +
    theme_bw(base_size = base_font_size) +
    theme(
      plot.title = element_text(face = "bold"),
      legend.position = "right",
      panel.grid.minor = element_blank()
    )

  ggsave(file.path(plot_dir, "ADNI_current_groups_MMSE_Cox_no_baselineMMSE_pairwise_HR_forest.svg"), p_cox, width = forest_width, height = forest_height, units = "in")
  ggsave(file.path(plot_dir, "ADNI_current_groups_MMSE_Cox_no_baselineMMSE_pairwise_HR_forest.pdf"), p_cox, width = forest_width, height = forest_height, units = "in")
  ggsave(file.path(plot_dir, "ADNI_current_groups_MMSE_Cox_no_baselineMMSE_pairwise_HR_forest.png"), p_cox, width = forest_width, height = forest_height, units = "in", dpi = 300)
}

message("[7] Running supplementary KM/log-rank checks")

if (nrow(event_data) >= 10 && length(unique(event_data$event)) >= 2 && nlevels(droplevels(event_data$group)) >= 2) {
  surv_obj <- survival::Surv(time = event_data$time, event = event_data$event)
  km_fit <- survival::survfit(surv_obj ~ group, data = event_data)

  logrank_overall <- survival::survdiff(survival::Surv(time, event) ~ group, data = event_data)
  logrank_overall_df <- data.frame(
    test = "overall_logrank",
    chisq = logrank_overall$chisq,
    df = length(logrank_overall$n) - 1,
    p.value = pchisq(logrank_overall$chisq, df = length(logrank_overall$n) - 1, lower.tail = FALSE),
    stringsAsFactors = FALSE
  ) %>% mutate(p_stars = vapply(p.value, p_to_stars, character(1)))
  write.csv(logrank_overall_df, file.path(table_dir, "ADNI_current_groups_MMSE_logrank_overall_supplementary.csv"), row.names = FALSE, quote = FALSE)

  pair_list <- combn(group_levels, 2, simplify = FALSE)
  pairwise_logrank <- lapply(pair_list, function(pair) {
    d <- event_data %>% filter(group %in% pair) %>% droplevels()
    if (nrow(d) < 10 || length(unique(d$event)) < 2 || nlevels(d$group) < 2) {
      return(data.frame(group1 = pair[1], group2 = pair[2], chisq = NA_real_, df = NA_real_, p.value = NA_real_))
    }
    test <- survival::survdiff(survival::Surv(time, event) ~ group, data = d)
    chisq <- test$chisq
    df_test <- length(test$n) - 1
    p_val <- pchisq(chisq, df = df_test, lower.tail = FALSE)
    data.frame(group1 = pair[1], group2 = pair[2], chisq = chisq, df = df_test, p.value = p_val, stringsAsFactors = FALSE)
  }) %>%
    bind_rows() %>%
    mutate(
      p_adj_bonferroni = p.adjust(p.value, method = "bonferroni"),
      p_adj_fdr = p.adjust(p.value, method = "BH"),
      p_stars = vapply(p.value, p_to_stars, character(1)),
      q_stars = vapply(p_adj_fdr, p_to_stars, character(1))
    )
  write.csv(pairwise_logrank, file.path(table_dir, "ADNI_current_groups_MMSE_pairwise_logrank_tests_supplementary.csv"), row.names = FALSE, quote = FALSE)

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
    title = paste0("Time to MMSE < ", mmse_event_threshold, " by projected group"),
    risk.table.height = 0.28,
    ggtheme = theme_bw(base_size = base_font_size)
  )

  svg(file.path(plot_dir, "ADNI_current_groups_KM_MMSE_decline_no_baselineMMSE_riskset_supplementary.svg"), width = km_width, height = km_height + risk_table_height)
  print(surv_plot)
  dev.off()

  pdf(file.path(plot_dir, "ADNI_current_groups_KM_MMSE_decline_no_baselineMMSE_riskset_supplementary.pdf"), width = km_width, height = km_height + risk_table_height)
  print(surv_plot)
  dev.off()

  png(file.path(plot_dir, "ADNI_current_groups_KM_MMSE_decline_no_baselineMMSE_riskset_supplementary.png"), width = km_width, height = km_height + risk_table_height, units = "in", res = 300)
  print(surv_plot)
  dev.off()
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
    apoe4_carrier_denom = sum(!is.na(APOE_e4_carrier)),
    apoe4_carrier_prop = apoe4_carrier_n / apoe4_carrier_denom,
    .groups = "drop"
  )
write.csv(baseline_summary, file.path(table_dir, "ADNI_current_groups_MMSE_Cox_event_summary_by_group.csv"), row.names = FALSE, quote = FALSE)

settings <- data.frame(
  setting = c(
    "projection_source",
    "mmse_file",
    "ptdemog_file",
    "apoe_file",
    "group_definition",
    "event_definition",
    "primary_risk_set",
    "primary_cox_model",
    "primary_covariates",
    "baseline_MMSE_role",
    "km_logrank_role"
  ),
  value = c(
    projection_source,
    mmse_file,
    ptdemog_file,
    apoe_file,
    "High resilience: predicted_cluster_like; Low resilience: non_resilience",
    paste0("First observed follow-up MMSE < ", mmse_event_threshold, "; censored at last available MMSE visit"),
    paste0("Exclude participants with first available MMSE < ", mmse_event_threshold, " and require follow-up after first MMSE observation"),
    "coxph(Surv(time, event) ~ group + AGE_AT_BASELINE + SEX + PTEDUCAT_clean + APOE_e4_carrier)",
    paste(cox_covariates_primary, collapse = ";"),
    "Baseline MMSE is used only to define the incident risk set and is not included as a covariate.",
    "Kaplan-Meier and log-rank are supplementary visualization/checks."
  ),
  stringsAsFactors = FALSE
)
write.csv(settings, file.path(table_dir, "ADNI_current_groups_MMSE_Cox_no_baselineMMSE_settings.csv"), row.names = FALSE, quote = FALSE)

sink(file.path(outdir, "sessionInfo_ADNI_current_groups_MMSE_Cox_no_baselineMMSE_pairwise.txt"))
print(sessionInfo())
sink()

message("Done. Results saved in: ", outdir)
