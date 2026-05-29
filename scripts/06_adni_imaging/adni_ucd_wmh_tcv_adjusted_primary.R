#!/usr/bin/env Rscript
# Script cleaned for public release. Edit /path/to/... inputs before running.

rm(list = ls())
options(stringsAsFactors = FALSE)

project_dir <- "/path/to/project"
setwd(project_dir)

adni_projection_file <- "out/ADNI/projected_rosmap_msbb_22gene_signature/tables/ADNI_projected_cluster_scores_with_clinical_ADpatho.csv"
adni_projection_all_file <- "out/ADNI/projected_rosmap_msbb_22gene_signature/tables/ADNI_projected_cluster_scores_with_clinical_all.csv"

imaging_file_candidates <- c(
  "/path/to/ADNI/imaging/UCD_WMH_30Jul2025.csv",
  "/path/to/ADNI/UCD_WMH_30Jul2025.csv"
)

imaging_visit <- "scmri"
use_earliest_if_visit_missing <- TRUE

outdir <- "out/ADNI/projected_rosmap_msbb_22gene_signature/imaging_UCD_WMH_TCV_adjusted_primary"
table_dir <- file.path(outdir, "tables")
plot_dir <- file.path(outdir, "plots")
forest_dir <- file.path(outdir, "forest_plots")
dir.create(table_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(plot_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(forest_dir, recursive = TRUE, showWarnings = FALSE)

group_levels <- c("cluster1", "cluster2", "non_resilience")
group_colors <- c(
  "cluster1" = "#009E73",
  "cluster2" = "#E69F00",
  "non_resilience" = "gray60"
)

contrast_levels <- c(
  "cluster1_vs_non_resilience",
  "cluster2_vs_non_resilience",
  "cluster1_vs_cluster2"
)
contrast_colors <- c(
  "cluster1_vs_non_resilience" = group_colors[["cluster1"]],
  "cluster2_vs_non_resilience" = group_colors[["cluster2"]],
  "cluster1_vs_cluster2" = "#4D4D4D"
)

primary_imaging_vars <- c(
  "TOTAL_HIPPO",
  "TOTAL_CSF",
  "TOTAL_GRAY",
  "TOTAL_WHITE",
  "TOTAL_BRAIN",
  "CEREBRUM_TCB",
  "CEREBRUM_TCC",
  "CEREBRUM_GRAY",
  "CEREBRUM_WHITE",
  "LEFT_HIPPO",
  "RIGHT_HIPPO",
  "log10_TOTAL_WMH_plus1"
)

include_cerebrum_tcv_as_descriptive_outcome <- TRUE

descriptive_outcome_vars <- c("CEREBRUM_TCV")

intracranial_proxy <- "CEREBRUM_TCV"

base_covariates_preferred <- c(
  "AGE_AT_MRI",
  "SEX",
  "PTEDUCAT_clean",
  "APOE_e4_carrier",
  "PHASE",
  "MANUFACTURER",
  "MAGNETICFIELDSTRENGTH"
)

main_forest_metrics <- c(
  "TOTAL_CSF",
  "TOTAL_GRAY",
  "CEREBRUM_GRAY",
  "CEREBRUM_TCB",
  "TOTAL_HIPPO",
  "log10_TOTAL_WMH_plus1"
)
main_forest_contrasts <- c(
  "cluster1_vs_non_resilience",
  "cluster2_vs_non_resilience"
)

make_supplementary_all_metric_forest <- TRUE

make_significant_contrast_forest <- TRUE

q_threshold <- 0.05

run_kruskal_dunn <- TRUE

base_size <- 16
main_forest_width <- 8.5
main_forest_height <- 5.4
supp_forest_width <- 9.5
supp_forest_height <- 8.5
sig_forest_width <- 10
sig_forest_max_height <- 14

boxplot_width <- 16
boxplot_height <- 11
jitter_width <- 0.20
point_size <- 1.0

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(data.table)
  library(tibble)
  library(emmeans)
  library(dunn.test)
  library(lubridate)
})

clean_numeric <- function(x) {
  x <- as.character(x)
  x[x %in% c("", "NA", "NaN", "NULL", "null", ".", " ")] <- NA
  x <- ifelse(x == "90+", "99", x)
  suppressWarnings(as.numeric(x))
}

safe_factor <- function(x) {
  droplevels(as.factor(as.character(x)))
}

first_present <- function(candidates, cols) {
  hit <- candidates[candidates %in% cols]
  if (length(hit) == 0) return(NA_character_)
  hit[1]
}

p_to_stars <- function(p) {
  if (is.na(p)) return("ns")
  if (p > 0.05) return("ns")
  if (p > 0.01) return("*")
  if (p > 0.001) return("**")
  if (p > 0.0001) return("***")
  "****"
}

q_to_stars <- p_to_stars

keep_usable_covariates <- function(df, covars) {
  covars <- covars[covars %in% colnames(df)]
  covars[sapply(covars, function(v) {
    z <- df[[v]]
    z <- z[!is.na(z)]
    if (is.factor(z) || is.character(z)) return(length(unique(z)) >= 2)
    length(unique(z)) >= 2 && stats::sd(as.numeric(z), na.rm = TRUE) > 0
  })]
}

metric_label <- function(x) {
  dplyr::case_when(
    x == "TOTAL_HIPPO" ~ "Total hippocampus",
    x == "TOTAL_CSF" ~ "Total CSF",
    x == "TOTAL_GRAY" ~ "Total gray matter",
    x == "TOTAL_WHITE" ~ "Total white matter",
    x == "TOTAL_BRAIN" ~ "Total brain",
    x == "CEREBRUM_TCV" ~ "Cerebrum cranial volume",
    x == "CEREBRUM_TCB" ~ "Cerebrum brain volume",
    x == "CEREBRUM_TCC" ~ "Cerebrum CSF volume",
    x == "CEREBRUM_GRAY" ~ "Cerebrum gray matter",
    x == "CEREBRUM_WHITE" ~ "Cerebrum white matter",
    x == "LEFT_HIPPO" ~ "Left hippocampus",
    x == "RIGHT_HIPPO" ~ "Right hippocampus",
    x == "log10_TOTAL_WMH_plus1" ~ "WMH volume, log10(x + 1)",
    x == "TOTAL_WMH" ~ "Total WMH",
    TRUE ~ x
  )
}

contrast_label <- function(x) {
  dplyr::case_when(
    x == "cluster1_vs_non_resilience" ~ "Cluster 1 vs non-CR",
    x == "cluster2_vs_non_resilience" ~ "Cluster 2 vs non-CR",
    x == "cluster1_vs_cluster2" ~ "Cluster 1 vs Cluster 2",
    TRUE ~ x
  )
}

contrast_label_colors <- setNames(
  contrast_colors[names(contrast_colors)],
  contrast_label(names(contrast_colors))
)

format_q_label <- function(q) {
  dplyr::case_when(
    is.na(q) ~ "q = NA",
    q < 0.001 ~ "q < 0.001",
    TRUE ~ paste0("q = ", sprintf("%.3f", q))
  )
}

read_first_existing <- function(paths) {
  hit <- paths[file.exists(paths)]
  if (length(hit) == 0) stop("None of the candidate files exist: ", paste(paths, collapse = "; "))
  hit[1]
}

extract_contrast_table <- function(emm_obj) {
  contr <- emmeans::contrast(
    emm_obj,
    method = list(
      cluster1_vs_non_resilience = c(1, 0, -1),
      cluster2_vs_non_resilience = c(0, 1, -1),
      cluster1_vs_cluster2 = c(1, -1, 0)
    ),
    adjust = "none"
  )
  as.data.frame(summary(contr, infer = c(TRUE, TRUE)))
}

add_standardized_columns <- function(df) {
  df %>%
    mutate(
      standardized_difference = estimate / sigma,
      standardized_lower.CL = lower.CL / sigma,
      standardized_upper.CL = upper.CL / sigma
    )
}

message("[1] Loading current ADNI projection table")

if (file.exists(adni_projection_file)) {
  adni <- read.csv(adni_projection_file, check.names = FALSE)
  projection_source <- adni_projection_file
} else if (file.exists(adni_projection_all_file)) {
  warning("ADpatho projection table not found. Falling back to all table and filtering ADpatho == 'AD'.")
  adni <- read.csv(adni_projection_all_file, check.names = FALSE)
  projection_source <- adni_projection_all_file
  if (!("ADpatho" %in% colnames(adni))) stop("Fallback all table does not contain ADpatho column.")
  adni <- adni %>% filter(ADpatho == "AD")
} else {
  stop("Neither ADpatho nor all ADNI projection table was found.")
}

required_cols <- c("RID", "resilience", "predicted_cluster_like")
missing_cols <- setdiff(required_cols, colnames(adni))
if (length(missing_cols) > 0) stop("ADNI projection table is missing required columns: ", paste(missing_cols, collapse = ", "))

adni_grouped <- adni %>%
  mutate(
    RID = as.integer(RID),
    resilience = as.character(resilience),
    predicted_cluster_like = as.character(predicted_cluster_like),
    imaging_group = case_when(
      resilience == "Low" ~ "non_resilience",
      resilience == "High" & predicted_cluster_like %in% c("cluster1", "cluster2") ~ predicted_cluster_like,
      TRUE ~ NA_character_
    ),
    imaging_group = factor(imaging_group, levels = group_levels)
  ) %>%
  filter(!is.na(imaging_group), imaging_group %in% group_levels)

message("    Projection source: ", projection_source)
message("    Current group distribution:")
print(table(adni_grouped$imaging_group, useNA = "ifany"))

message("[2] Loading UCD WMH / four-tissue volume data")
imaging_file <- read_first_existing(imaging_file_candidates)
ucd <- read.csv(imaging_file, check.names = FALSE)

if (!("RID" %in% colnames(ucd))) stop("UCD file must contain RID.")
if (!("VISCODE2" %in% colnames(ucd))) stop("UCD file must contain VISCODE2.")
if (!(intracranial_proxy %in% colnames(ucd))) stop("UCD file does not contain required intracranial proxy: ", intracranial_proxy)

ucd <- ucd %>% mutate(RID = as.integer(RID))

if (imaging_visit %in% unique(as.character(ucd$VISCODE2))) {
  ucd_use <- ucd %>%
    filter(VISCODE2 == imaging_visit) %>%
    arrange(RID) %>%
    distinct(RID, .keep_all = TRUE)
  imaging_visit_used <- imaging_visit
} else if (use_earliest_if_visit_missing) {
  warning("Requested imaging_visit not found: ", imaging_visit, ". Using earliest available visit per RID.")
  date_col <- first_present(c("EXAMDATE", "EXAMDATE_bl", "VISDATE", "RUNDATE"), colnames(ucd))
  if (!is.na(date_col)) {
    ucd_use <- ucd %>%
      mutate(.date_tmp = suppressWarnings(lubridate::ymd(.data[[date_col]]))) %>%
      arrange(RID, .date_tmp) %>%
      distinct(RID, .keep_all = TRUE) %>%
      select(-.date_tmp)
  } else {
    ucd_use <- ucd %>% arrange(RID) %>% distinct(RID, .keep_all = TRUE)
  }
  imaging_visit_used <- "earliest_available"
} else {
  stop("Requested imaging_visit not found: ", imaging_visit)
}

ucd_numeric_candidates <- unique(c(
  primary_imaging_vars,
  descriptive_outcome_vars,
  intracranial_proxy,
  "TOTAL_WMH",
  "CEREBRUM_TCV",
  "CEREBRUM_TCB",
  "CEREBRUM_TCC",
  "CEREBRUM_GRAY",
  "CEREBRUM_WHITE",
  "TOTAL_HIPPO",
  "LEFT_HIPPO",
  "RIGHT_HIPPO",
  "TOTAL_CSF",
  "TOTAL_GRAY",
  "TOTAL_WHITE",
  "TOTAL_BRAIN",
  "MAGNETICFIELDSTRENGTH"
))
ucd_numeric_candidates <- ucd_numeric_candidates[ucd_numeric_candidates %in% colnames(ucd_use)]
for (v in ucd_numeric_candidates) ucd_use[[v]] <- clean_numeric(ucd_use[[v]])

if ("TOTAL_WMH" %in% colnames(ucd_use)) {
  ucd_use$log10_TOTAL_WMH_plus1 <- log10(clean_numeric(ucd_use$TOTAL_WMH) + 1)
}

message("[3] Preparing covariates")

apoe_file_candidates <- c(
  "/path/to/ADNI/APOERES_30Jul2025.csv",
  "/path/to/ADNI/APOERES_07Jan2025.csv",
  "/path/to/ADNI/APOERES.csv"
)
ptdemog_file_candidates <- c(
  "/path/to/ADNI/PTDEMOG_30Jul2025.csv",
  "/path/to/ADNI/PTDEMOG_07Jan2025.csv",
  "/path/to/ADNI/PTDEMOG.csv"
)

apoe_df <- NULL
apoe_file <- NA_character_
if (!any(c("APOE_e4_carrier", "APOE4", "APGEN1", "APGEN2") %in% colnames(adni_grouped))) {
  hit <- apoe_file_candidates[file.exists(apoe_file_candidates)]
  if (length(hit) > 0) {
    apoe_file <- hit[1]
    ap <- read.csv(apoe_file, check.names = FALSE)
    if ("RID" %in% colnames(ap)) {
      ap <- ap %>% mutate(RID = as.integer(RID))
      if (all(c("APGEN1", "APGEN2") %in% colnames(ap))) {
        apoe_df <- ap %>%
          mutate(
            APGEN1 = as.character(APGEN1),
            APGEN2 = as.character(APGEN2),
            APOE_e4_carrier = ifelse(APGEN1 == "4" | APGEN2 == "4", 1, 0)
          ) %>%
          select(RID, APOE_e4_carrier) %>%
          distinct(RID, .keep_all = TRUE)
      } else if ("APOE4" %in% colnames(ap)) {
        apoe_df <- ap %>%
          mutate(APOE_e4_carrier = ifelse(clean_numeric(APOE4) > 0, 1, 0)) %>%
          select(RID, APOE_e4_carrier) %>%
          distinct(RID, .keep_all = TRUE)
      }
    }
  }
}

ptdemog_df <- NULL
ptdemog_file <- NA_character_
hit_demo <- ptdemog_file_candidates[file.exists(ptdemog_file_candidates)]
if (length(hit_demo) > 0) {
  ptdemog_file <- hit_demo[1]
  pd <- read.csv(ptdemog_file, check.names = FALSE)
  if ("RID" %in% colnames(pd)) {
    pd <- pd %>% mutate(RID = as.integer(RID))
    if ("VISCODE2" %in% colnames(pd)) {
      pref <- c("sc", "bl", "init", "f")
      pd <- pd %>%
        mutate(.visit_rank = match(VISCODE2, pref), .visit_rank = ifelse(is.na(.visit_rank), 999, .visit_rank)) %>%
        arrange(RID, .visit_rank) %>%
        group_by(RID) %>% slice(1) %>% ungroup()
    } else {
      pd <- pd %>% arrange(RID) %>% distinct(RID, .keep_all = TRUE)
    }
    ptdemog_df <- pd %>%
      transmute(
        RID,
        SEX_from_ptdemog = if ("PTGENDER" %in% colnames(pd)) as.character(PTGENDER) else NA_character_,
        PTEDUCAT_from_ptdemog = if ("PTEDUCAT" %in% colnames(pd)) clean_numeric(PTEDUCAT) else NA_real_,
        PTDOBYY = if ("PTDOBYY" %in% colnames(pd)) clean_numeric(PTDOBYY) else NA_real_,
        PTDOB = if ("PTDOB" %in% colnames(pd)) as.character(PTDOB) else NA_character_
      ) %>%
      distinct(RID, .keep_all = TRUE)
  }
}

message("[4] Merging data")

merged <- adni_grouped %>%
  left_join(ucd_use, by = "RID", suffix = c(".adni", ".ucd"))

if (!is.null(apoe_df)) merged <- merged %>% left_join(apoe_df, by = "RID", suffix = c("", ".apoe"))
if (!is.null(ptdemog_df)) merged <- merged %>% left_join(ptdemog_df, by = "RID")

sex_col <- first_present(c("sex", "SEX", "PTGENDER", "SEX_from_ptdemog"), colnames(merged))
if (!is.na(sex_col)) {
  sx <- merged[[sex_col]]
  sx_chr <- as.character(sx)
  sx_chr[sx_chr %in% c("1", "M", "Male", "male")] <- "Male"
  sx_chr[sx_chr %in% c("2", "F", "Female", "female")] <- "Female"
  merged$SEX <- factor(sx_chr)
} else {
  merged$SEX <- factor(NA_character_)
}

edu_col <- first_present(c("PTEDUCAT", "PTEDUCAT_clean", "education", "PTEDUCAT_from_ptdemog"), colnames(merged))
if (!is.na(edu_col)) merged$PTEDUCAT_clean <- clean_numeric(merged[[edu_col]]) else merged$PTEDUCAT_clean <- NA_real_

if ("APOE_e4_carrier" %in% colnames(merged)) {
  merged$APOE_e4_carrier <- clean_numeric(merged$APOE_e4_carrier)
} else if ("APOE4" %in% colnames(merged)) {
  merged$APOE_e4_carrier <- ifelse(clean_numeric(merged$APOE4) > 0, 1, 0)
} else if (all(c("APGEN1", "APGEN2") %in% colnames(merged))) {
  merged$APOE_e4_carrier <- ifelse(as.character(merged$APGEN1) == "4" | as.character(merged$APGEN2) == "4", 1, 0)
} else {
  merged$APOE_e4_carrier <- NA_real_
}
merged$APOE_e4_carrier <- factor(merged$APOE_e4_carrier, levels = c(0, 1), labels = c("noncarrier", "carrier"))

if ("PHASE" %in% colnames(merged)) merged$PHASE <- safe_factor(merged$PHASE) else merged$PHASE <- factor(NA_character_)
if ("MANUFACTURER" %in% colnames(merged)) merged$MANUFACTURER <- safe_factor(merged$MANUFACTURER) else merged$MANUFACTURER <- factor(NA_character_)
if ("MAGNETICFIELDSTRENGTH" %in% colnames(merged)) {
  merged$MAGNETICFIELDSTRENGTH <- factor(clean_numeric(merged$MAGNETICFIELDSTRENGTH))
} else {
  merged$MAGNETICFIELDSTRENGTH <- factor(NA_character_)
}

examdate_col <- first_present(c("EXAMDATE", "EXAMDATE.ucd", "VISDATE", "RUNDATE"), colnames(merged))
if (!is.na(examdate_col)) {
  merged$EXAMDATE_parsed <- suppressWarnings(lubridate::ymd(merged[[examdate_col]]))
} else {
  merged$EXAMDATE_parsed <- as.Date(NA)
}

if ("PTDOB" %in% colnames(merged)) {
  merged$PTDOB_parsed <- suppressWarnings(lubridate::ymd(merged$PTDOB))
} else {
  merged$PTDOB_parsed <- as.Date(NA)
}

merged$AGE_AT_MRI_from_dob <- as.numeric(difftime(merged$EXAMDATE_parsed, merged$PTDOB_parsed, units = "days")) / 365.25
if ("PTDOBYY" %in% colnames(merged)) {
  merged$AGE_AT_MRI_from_year <- lubridate::year(merged$EXAMDATE_parsed) - clean_numeric(merged$PTDOBYY)
} else {
  merged$AGE_AT_MRI_from_year <- NA_real_
}

age_projection_col <- first_present(c("age_at_visit", "AGE", "AGE_AT_VISIT"), colnames(merged))
if (!is.na(age_projection_col)) merged$AGE_AT_PROJECTION <- clean_numeric(merged[[age_projection_col]]) else merged$AGE_AT_PROJECTION <- NA_real_

merged$AGE_AT_MRI <- dplyr::case_when(
  is.finite(merged$AGE_AT_MRI_from_dob) & merged$AGE_AT_MRI_from_dob > 0 ~ merged$AGE_AT_MRI_from_dob,
  is.finite(merged$AGE_AT_MRI_from_year) & merged$AGE_AT_MRI_from_year > 0 ~ merged$AGE_AT_MRI_from_year,
  is.finite(merged$AGE_AT_PROJECTION) ~ merged$AGE_AT_PROJECTION,
  TRUE ~ NA_real_
)

merged[[intracranial_proxy]] <- clean_numeric(merged[[intracranial_proxy]])

outcome_vars <- primary_imaging_vars[primary_imaging_vars %in% colnames(merged)]
if (include_cerebrum_tcv_as_descriptive_outcome) {
  outcome_vars <- unique(c(outcome_vars, descriptive_outcome_vars[descriptive_outcome_vars %in% colnames(merged)]))
}

if (length(outcome_vars) == 0) stop("No primary UCD outcomes found in merged data.")
for (v in outcome_vars) merged[[v]] <- clean_numeric(merged[[v]])

write.csv(
  merged,
  file.path(table_dir, "ADNI_current_groups_merged_UCD_WMH_with_covariates_TCV_adjusted.csv"),
  row.names = FALSE,
  quote = FALSE
)

message("    UCD file: ", imaging_file)
message("    Imaging visit used: ", imaging_visit_used)
message("    Outcomes: ", paste(outcome_vars, collapse = ", "))
message("    Intracranial-size proxy: ", intracranial_proxy)
message("    Merged samples: ", nrow(merged))
message("    Samples with any outcome: ", sum(rowSums(!is.na(merged[, outcome_vars, drop = FALSE])) > 0))

covar_availability <- data.frame(
  covariate = c("AGE_AT_MRI", "SEX", "PTEDUCAT_clean", "APOE_e4_carrier", "PHASE", "MANUFACTURER", "MAGNETICFIELDSTRENGTH", intracranial_proxy),
  n_nonmissing = sapply(c("AGE_AT_MRI", "SEX", "PTEDUCAT_clean", "APOE_e4_carrier", "PHASE", "MANUFACTURER", "MAGNETICFIELDSTRENGTH", intracranial_proxy), function(v) sum(!is.na(merged[[v]]))),
  n_unique_nonmissing = sapply(c("AGE_AT_MRI", "SEX", "PTEDUCAT_clean", "APOE_e4_carrier", "PHASE", "MANUFACTURER", "MAGNETICFIELDSTRENGTH", intracranial_proxy), function(v) length(unique(merged[[v]][!is.na(merged[[v]])]))),
  stringsAsFactors = FALSE
)
write.csv(covar_availability, file.path(table_dir, "UCD_WMH_TCV_adjusted_covariate_availability_summary.csv"), row.names = FALSE, quote = FALSE)

message("[5] Running primary TCV-adjusted linear models")

primary_omnibus_rows <- list()
primary_contrast_rows <- list()
coefficient_rows <- list()

for (metric in outcome_vars) {
  covars <- base_covariates_preferred

  if (metric != intracranial_proxy && intracranial_proxy %in% colnames(merged)) {
    covars <- c(covars, intracranial_proxy)
  }

  covars <- keep_usable_covariates(merged, covars)
  use_cols <- c(metric, "imaging_group", covars)
  d <- merged[, use_cols, drop = FALSE]
  d[[metric]] <- clean_numeric(d[[metric]])
  d$imaging_group <- factor(d$imaging_group, levels = group_levels)
  for (cc in intersect(c("SEX", "APOE_e4_carrier", "PHASE", "MANUFACTURER", "MAGNETICFIELDSTRENGTH"), colnames(d))) {
    d[[cc]] <- droplevels(as.factor(d[[cc]]))
  }
  d <- d[complete.cases(d), , drop = FALSE]
  d$imaging_group <- droplevels(factor(d$imaging_group, levels = group_levels))

  if (nrow(d) < 20 || length(unique(d$imaging_group)) < 2 || length(unique(d[[metric]])) < 2) {
    primary_omnibus_rows[[length(primary_omnibus_rows) + 1]] <- data.frame(
      metric = metric,
      statistic = NA_real_,
      p.value = NA_real_,
      partial_eta2 = NA_real_,
      n = nrow(d),
      n_groups = length(unique(d$imaging_group)),
      formula = NA_character_,
      covariates = paste(covars, collapse = ";"),
      note = "insufficient_data",
      stringsAsFactors = FALSE
    )
    for (cn in contrast_levels) {
      primary_contrast_rows[[length(primary_contrast_rows) + 1]] <- data.frame(
        metric = metric,
        contrast = cn,
        estimate = NA_real_,
        SE = NA_real_,
        df = NA_real_,
        t.ratio = NA_real_,
        p.value = NA_real_,
        lower.CL = NA_real_,
        upper.CL = NA_real_,
        sigma = NA_real_,
        standardized_difference = NA_real_,
        standardized_lower.CL = NA_real_,
        standardized_upper.CL = NA_real_,
        n = nrow(d),
        formula = NA_character_,
        covariates = paste(covars, collapse = ";"),
        note = "insufficient_data",
        stringsAsFactors = FALSE
      )
    }
    next
  }

  usable_covars <- keep_usable_covariates(d, covars)
  rhs <- c("imaging_group", usable_covars)
  full_form <- as.formula(paste(metric, "~", paste(rhs, collapse = " + ")))
  reduced_form <- if (length(usable_covars) == 0) {
    as.formula(paste(metric, "~ 1"))
  } else {
    as.formula(paste(metric, "~", paste(usable_covars, collapse = " + ")))
  }

  fit_full <- tryCatch(lm(full_form, data = d), error = function(e) NULL)
  fit_reduced <- tryCatch(lm(reduced_form, data = d), error = function(e) NULL)
  an <- if (is.null(fit_full) || is.null(fit_reduced)) NULL else tryCatch(anova(fit_reduced, fit_full), error = function(e) NULL)

  if (is.null(fit_full) || is.null(an)) {
    primary_omnibus_rows[[length(primary_omnibus_rows) + 1]] <- data.frame(
      metric = metric,
      statistic = NA_real_,
      p.value = NA_real_,
      partial_eta2 = NA_real_,
      n = nrow(d),
      n_groups = length(unique(d$imaging_group)),
      formula = paste(deparse(full_form), collapse = ""),
      covariates = paste(usable_covars, collapse = ";"),
      note = "model_failed",
      stringsAsFactors = FALSE
    )
    for (cn in contrast_levels) {
      primary_contrast_rows[[length(primary_contrast_rows) + 1]] <- data.frame(
        metric = metric,
        contrast = cn,
        estimate = NA_real_, SE = NA_real_, df = NA_real_, t.ratio = NA_real_, p.value = NA_real_,
        lower.CL = NA_real_, upper.CL = NA_real_, sigma = NA_real_, standardized_difference = NA_real_,
        standardized_lower.CL = NA_real_, standardized_upper.CL = NA_real_, n = nrow(d),
        formula = paste(deparse(full_form), collapse = ""),
        covariates = paste(usable_covars, collapse = ";"),
        note = "model_failed",
        stringsAsFactors = FALSE
      )
    }
    next
  }

  p_group <- an$`Pr(>F)`[2]
  f_stat <- an$F[2]
  ss_group <- an$RSS[1] - an$RSS[2]
  ss_resid <- an$RSS[2]
  partial_eta2 <- ss_group / (ss_group + ss_resid)
  sig <- sigma(fit_full)

  primary_omnibus_rows[[length(primary_omnibus_rows) + 1]] <- data.frame(
    metric = metric,
    statistic = f_stat,
    p.value = p_group,
    partial_eta2 = partial_eta2,
    n = nrow(d),
    n_groups = length(unique(d$imaging_group)),
    formula = paste(deparse(full_form), collapse = ""),
    covariates = paste(usable_covars, collapse = ";"),
    note = "",
    stringsAsFactors = FALSE
  )

  coef_df <- tryCatch({
    sm <- summary(fit_full)$coefficients
    data.frame(
      metric = metric,
      term = rownames(sm),
      estimate = sm[, "Estimate"],
      SE = sm[, "Std. Error"],
      statistic = sm[, "t value"],
      p.value = sm[, "Pr(>|t|)"],
      n = nrow(d),
      formula = paste(deparse(full_form), collapse = ""),
      covariates = paste(usable_covars, collapse = ";"),
      stringsAsFactors = FALSE
    )
  }, error = function(e) NULL)
  if (!is.null(coef_df)) coefficient_rows[[length(coefficient_rows) + 1]] <- coef_df

  present_groups <- levels(droplevels(d$imaging_group))
  if (all(group_levels %in% present_groups)) {
    emm <- tryCatch(emmeans::emmeans(fit_full, ~ imaging_group), error = function(e) NULL)
    con <- if (is.null(emm)) NULL else tryCatch(extract_contrast_table(emm), error = function(e) NULL)
    if (!is.null(con)) {
      con <- con %>%
        rename(contrast = contrast) %>%
        mutate(
          metric = metric,
          sigma = sig,
          n = nrow(d),
          formula = paste(deparse(full_form), collapse = ""),
          covariates = paste(usable_covars, collapse = ";"),
          note = ""
        ) %>%
        add_standardized_columns() %>%
        select(
          metric, contrast, estimate, SE, df, t.ratio, p.value, lower.CL, upper.CL,
          sigma, standardized_difference, standardized_lower.CL, standardized_upper.CL,
          n, formula, covariates, note
        )
      primary_contrast_rows[[length(primary_contrast_rows) + 1]] <- con
    } else {
      for (cn in contrast_levels) {
        primary_contrast_rows[[length(primary_contrast_rows) + 1]] <- data.frame(
          metric = metric, contrast = cn, estimate = NA_real_, SE = NA_real_, df = NA_real_, t.ratio = NA_real_,
          p.value = NA_real_, lower.CL = NA_real_, upper.CL = NA_real_, sigma = sig,
          standardized_difference = NA_real_, standardized_lower.CL = NA_real_, standardized_upper.CL = NA_real_,
          n = nrow(d), formula = paste(deparse(full_form), collapse = ""),
          covariates = paste(usable_covars, collapse = ";"), note = "emmeans_failed", stringsAsFactors = FALSE
        )
      }
    }
  } else {
    for (cn in contrast_levels) {
      primary_contrast_rows[[length(primary_contrast_rows) + 1]] <- data.frame(
        metric = metric, contrast = cn, estimate = NA_real_, SE = NA_real_, df = NA_real_, t.ratio = NA_real_,
        p.value = NA_real_, lower.CL = NA_real_, upper.CL = NA_real_, sigma = sig,
        standardized_difference = NA_real_, standardized_lower.CL = NA_real_, standardized_upper.CL = NA_real_,
        n = nrow(d), formula = paste(deparse(full_form), collapse = ""),
        covariates = paste(usable_covars, collapse = ";"), note = "contrast_groups_not_all_present", stringsAsFactors = FALSE
      )
    }
  }
}

primary_omnibus <- bind_rows(primary_omnibus_rows) %>%
  mutate(
    FDR_across_metrics = p.adjust(p.value, method = "BH"),
    p_stars = vapply(p.value, p_to_stars, character(1)),
    q_stars = vapply(FDR_across_metrics, q_to_stars, character(1))
  )

primary_contrasts <- bind_rows(primary_contrast_rows) %>%
  mutate(
    FDR_all_metrics_all_contrasts = p.adjust(p.value, method = "BH"),
    p_stars = vapply(p.value, p_to_stars, character(1)),
    q_stars = vapply(FDR_all_metrics_all_contrasts, q_to_stars, character(1)),
    metric_label = metric_label(metric),
    contrast_label = contrast_label(contrast),
    significant_q05 = !is.na(FDR_all_metrics_all_contrasts) & FDR_all_metrics_all_contrasts < q_threshold
  )

primary_coefficients <- bind_rows(coefficient_rows)
if (nrow(primary_coefficients) > 0) {
  primary_coefficients <- primary_coefficients %>%
    group_by(metric) %>%
    mutate(FDR_within_metric = p.adjust(p.value, method = "BH")) %>%
    ungroup()
}

write.csv(primary_omnibus, file.path(table_dir, "UCD_WMH_TCV_adjusted_primary_LM_omnibus_group_effect.csv"), row.names = FALSE, quote = FALSE)
write.csv(primary_contrasts, file.path(table_dir, "UCD_WMH_TCV_adjusted_primary_LM_pairwise_contrasts_all_metrics_BH_FDR.csv"), row.names = FALSE, quote = FALSE)
write.csv(primary_coefficients, file.path(table_dir, "UCD_WMH_TCV_adjusted_primary_LM_coefficients_long.csv"), row.names = FALSE, quote = FALSE)

supp_table <- primary_contrasts %>%
  transmute(
    metric,
    contrast,
    estimate,
    lower_95CI = lower.CL,
    upper_95CI = upper.CL,
    standardized_difference,
    standardized_lower_95CI = standardized_lower.CL,
    standardized_upper_95CI = standardized_upper.CL,
    SE,
    df,
    t.ratio,
    p.value,
    FDR_BH = FDR_all_metrics_all_contrasts,
    n,
    covariates,
    note
  )
write.csv(supp_table, file.path(table_dir, "Supplementary_Table_UCD_TCV_adjusted_primary_pairwise_contrasts.csv"), row.names = FALSE, quote = FALSE)
write.csv(supp_table %>% filter(!is.na(FDR_BH), FDR_BH < q_threshold), file.path(table_dir, "Supplementary_Table_UCD_TCV_adjusted_primary_pairwise_contrasts_significant_q05.csv"), row.names = FALSE, quote = FALSE)

message("[6] Drawing primary forest plots")

plot_forest <- function(df, filename_prefix, title, subtitle, width, height) {
  if (nrow(df) == 0) {
    warning("No rows for forest plot: ", filename_prefix)
    return(NULL)
  }
  df <- df %>%
    mutate(
      contrast = factor(contrast, levels = contrast_levels),
      contrast_label = factor(contrast_label, levels = contrast_label(contrast_levels)),
      metric_label = factor(metric_label, levels = rev(unique(metric_label)))
    )

  p <- ggplot(
    df,
    aes(
      x = standardized_difference,
      y = metric_label,
      color = contrast_label,
      shape = significant_q05
    )
  ) +
    geom_vline(xintercept = 0, linetype = "dashed", linewidth = 0.5, color = "gray40") +
    geom_errorbarh(
      aes(xmin = standardized_lower.CL, xmax = standardized_upper.CL),
      height = 0.18,
      linewidth = 0.7,
      position = position_dodge(width = 0.60),
      na.rm = TRUE
    ) +
    geom_point(
      size = 3.0,
      stroke = 0.9,
      position = position_dodge(width = 0.60),
      na.rm = TRUE
    ) +
    scale_color_manual(
      values = contrast_label_colors,
      name = "Projected group contrast"
    ) +
    scale_shape_manual(
      values = c("TRUE" = 16, "FALSE" = 1),
      labels = c("TRUE" = "BH-FDR q < 0.05", "FALSE" = "q >= 0.05"),
      name = "Statistical significance"
    ) +
    labs(
      title = title,
      subtitle = subtitle,
      x = "Standardized adjusted difference",
      y = NULL
    ) +
    theme_bw(base_size = base_size) +
    theme(
      plot.title = element_text(face = "bold", size = base_size + 2),
      plot.subtitle = element_text(size = base_size - 3),
      axis.text.y = element_text(size = base_size),
      axis.text.x = element_text(size = base_size - 1),
      axis.title.x = element_text(size = base_size),
      legend.position = "right",
      legend.title = element_text(size = base_size - 2),
      legend.text = element_text(size = base_size - 3),
      panel.grid.major.y = element_line(color = "gray90"),
      panel.grid.minor = element_blank()
    )

  ggsave(file.path(forest_dir, paste0(filename_prefix, ".svg")), p, width = width, height = height, units = "in")
  ggsave(file.path(forest_dir, paste0(filename_prefix, ".pdf")), p, width = width, height = height, units = "in")
  ggsave(file.path(forest_dir, paste0(filename_prefix, ".png")), p, width = width, height = height, units = "in", dpi = 300)
  p
}

main_df <- primary_contrasts %>%
  filter(metric %in% main_forest_metrics, contrast %in% main_forest_contrasts) %>%
  mutate(
    metric = factor(metric, levels = rev(main_forest_metrics)),
    metric_label = factor(metric_label, levels = metric_label(rev(main_forest_metrics)))
  ) %>%
  arrange(metric, contrast)

plot_forest(
  main_df,
  "UCD_TCV_adjusted_primary_forest_main_representative_metrics",
  "UCD WMH/global-volume metrics: TCV-adjusted primary analysis",
  "Models adjusted for age at MRI, sex, education, APOE e4, ADNI phase, manufacturer, field strength when estimable, and CEREBRUM_TCV as intracranial-size proxy",
  main_forest_width,
  main_forest_height
)

if (make_supplementary_all_metric_forest) {
  supp_metrics <- primary_imaging_vars[primary_imaging_vars %in% primary_contrasts$metric]
  supp_df <- primary_contrasts %>%
    filter(metric %in% supp_metrics, contrast %in% main_forest_contrasts) %>%
    mutate(
      metric = factor(metric, levels = rev(supp_metrics)),
      metric_label = factor(metric_label, levels = metric_label(rev(supp_metrics)))
    ) %>%
    arrange(metric, contrast)

  plot_forest(
    supp_df,
    "UCD_TCV_adjusted_primary_forest_supplementary_all_metrics_cluster_vs_nonCR",
    "All UCD WMH/global-volume metrics: TCV-adjusted primary contrasts",
    "Cluster 1 and cluster 2 compared with non-CR; all UCD primary metrics shown",
    supp_forest_width,
    supp_forest_height
  )
}

if (make_significant_contrast_forest) {
  sig_df <- primary_contrasts %>%
    filter(!is.na(FDR_all_metrics_all_contrasts), FDR_all_metrics_all_contrasts < q_threshold) %>%
    filter(metric != intracranial_proxy) %>%
    arrange(FDR_all_metrics_all_contrasts)

  if (nrow(sig_df) > 0) {
    sig_height <- min(sig_forest_max_height, max(4.5, 0.35 * nrow(sig_df) + 2.5))
    plot_forest(
      sig_df,
      "UCD_TCV_adjusted_primary_forest_significant_contrasts_only",
      "UCD TCV-adjusted primary analysis: significant contrasts only",
      "Only BH-FDR q < 0.05 contrasts are shown",
      sig_forest_width,
      sig_height
    )
  } else {
    message("    No significant contrasts for significant-only forest plot.")
  }
}

if (run_kruskal_dunn) {
  message("[7] Running supplementary Kruskal-Wallis and Dunn tests")
  stat_summary_list <- list()
  dunn_result_list <- list()

  for (metric in outcome_vars) {
    testdata <- merged %>%
      select(imaging_group, all_of(metric)) %>%
      filter(!is.na(imaging_group), !is.na(.data[[metric]]))

    if (nrow(testdata) < 3 || length(unique(testdata$imaging_group)) < 2) {
      stat_summary_list[[metric]] <- data.frame(metric = metric, test = "Kruskal-Wallis", statistic = NA_real_, df = NA_real_, p.value = NA_real_, n = nrow(testdata), note = "insufficient_data", stringsAsFactors = FALSE)
      next
    }

    kw <- tryCatch(kruskal.test(as.formula(paste(metric, "~ imaging_group")), data = testdata), error = function(e) NULL)
    stat_summary_list[[metric]] <- data.frame(
      metric = metric,
      test = "Kruskal-Wallis",
      statistic = if (is.null(kw)) NA_real_ else as.numeric(kw$statistic),
      df = if (is.null(kw)) NA_real_ else as.numeric(kw$parameter),
      p.value = if (is.null(kw)) NA_real_ else kw$p.value,
      n = nrow(testdata),
      note = if (is.null(kw)) "test_failed" else "",
      stringsAsFactors = FALSE
    )

    if (length(unique(testdata$imaging_group)) >= 3) {
      dunn <- tryCatch(dunn.test::dunn.test(testdata[[metric]], testdata$imaging_group, method = "bonferroni", kw = FALSE), error = function(e) NULL)
      if (!is.null(dunn)) {
        dunn_result_list[[metric]] <- data.frame(
          metric = metric,
          comparison = dunn$comparisons,
          p.value = dunn$P.adjusted,
          stringsAsFactors = FALSE
        )
      }
    }
  }

  stat_summary <- bind_rows(stat_summary_list) %>%
    mutate(FDR = p.adjust(p.value, method = "BH"), p_stars = vapply(p.value, p_to_stars, character(1)))
  dunn_result <- bind_rows(dunn_result_list) %>%
    mutate(FDR_global = p.adjust(p.value, method = "BH"), p_stars = vapply(p.value, p_to_stars, character(1)), q_stars = vapply(FDR_global, q_to_stars, character(1)))

  write.csv(stat_summary, file.path(table_dir, "UCD_WMH_TCV_adjusted_supplementary_kruskal_summary.csv"), row.names = FALSE, quote = FALSE)
  write.csv(dunn_result, file.path(table_dir, "UCD_WMH_TCV_adjusted_supplementary_dunn_bonferroni_with_global_BH_FDR.csv"), row.names = FALSE, quote = FALSE)
}

message("[8] Drawing supplementary unadjusted boxplot")

box_metrics <- main_forest_metrics[main_forest_metrics %in% colnames(merged)]
if (length(box_metrics) > 0) {
  df_long <- merged %>%
    select(RID, imaging_group, all_of(box_metrics)) %>%
    pivot_longer(cols = all_of(box_metrics), names_to = "metric", values_to = "value") %>%
    filter(!is.na(imaging_group), !is.na(value)) %>%
    mutate(
      imaging_group = factor(imaging_group, levels = group_levels),
      metric_label = factor(metric_label(metric), levels = metric_label(box_metrics))
    )

  p_box <- ggplot(df_long, aes(x = imaging_group, y = value, fill = imaging_group)) +
    geom_boxplot(outlier.shape = NA, alpha = 0.85) +
    geom_jitter(width = jitter_width, alpha = 0.40, size = point_size, color = "black") +
    scale_fill_manual(values = group_colors, name = "Projected group") +
    labs(
      title = "UCD WMH/global-volume metrics by projected group",
      subtitle = "Unadjusted distributions shown for visualization only; primary inference uses TCV-adjusted linear models",
      x = "Projected group",
      y = "Value"
    ) +
    facet_wrap(~ metric_label, scales = "free_y", ncol = 3) +
    theme_bw(base_size = base_size) +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1),
      plot.title = element_text(face = "bold"),
      strip.text = element_text(size = base_size - 2)
    )

  ggsave(file.path(plot_dir, "UCD_TCV_adjusted_supplementary_unadjusted_boxplot_representative_metrics.svg"), p_box, width = boxplot_width, height = boxplot_height, units = "in")
  ggsave(file.path(plot_dir, "UCD_TCV_adjusted_supplementary_unadjusted_boxplot_representative_metrics.png"), p_box, width = boxplot_width, height = boxplot_height, units = "in", dpi = 300)

  write.csv(df_long, file.path(table_dir, "UCD_TCV_adjusted_supplementary_unadjusted_boxplot_long.csv"), row.names = FALSE, quote = FALSE)
}

settings <- data.frame(
  setting = c(
    "projection_source",
    "imaging_file",
    "imaging_visit_requested",
    "imaging_visit_used",
    "group_definition",
    "primary_outcomes",
    "intracranial_size_proxy",
    "primary_covariates",
    "FDR_scope",
    "baseline_MMSE_in_primary",
    "DIAGNOSIS_in_primary",
    "UCSF_FreeSurfer_policy"
  ),
  value = c(
    projection_source,
    imaging_file,
    imaging_visit,
    imaging_visit_used,
    "High resilience: predicted_cluster_like; Low resilience: non_resilience",
    paste(outcome_vars, collapse = ";"),
    intracranial_proxy,
    paste(base_covariates_preferred, collapse = ";"),
    "all UCD outcomes x all three pairwise contrasts",
    "No",
    "No",
    "UCSF FreeSurfer results are not re-modeled here; because eTIV was not available, UCSF regional findings should be reported in supplementary tables without eTIV adjustment or as exploratory support."
  ),
  stringsAsFactors = FALSE
)
write.csv(settings, file.path(table_dir, "UCD_WMH_TCV_adjusted_primary_analysis_settings.csv"), row.names = FALSE, quote = FALSE)

sink(file.path(outdir, "sessionInfo_UCD_WMH_TCV_adjusted_primary.txt"))
print(sessionInfo())
sink()

message("Done. Results saved in: ", outdir)
