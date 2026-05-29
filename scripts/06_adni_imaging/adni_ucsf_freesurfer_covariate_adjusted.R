#!/usr/bin/env Rscript
# Script cleaned for public release. Edit /path/to/... inputs before running.

rm(list = ls())
options(stringsAsFactors = FALSE)

project_dir <- "/path/to/project"
setwd(project_dir)

adni_projection_file <- "out/ADNI/projected_rosmap_msbb_22gene_signature/tables/ADNI_projected_cluster_scores_with_clinical_ADpatho.csv"
adni_projection_all_file <- "out/ADNI/projected_rosmap_msbb_22gene_signature/tables/ADNI_projected_cluster_scores_with_clinical_all.csv"

ptdemog_file <- "/path/to/ADNI/PTDEMOG_30Jul2025.csv"
apoe_file <- "/path/to/ADNI/APOERES_07Nov2025.csv"

volume_file <- "/path/to/ADNI/UCSFFSX7_05Aug2025.csv"
freesurfer_id_file <- "/path/to/ADNI/FreeSurfer_id.csv"
volume_visit <- "scmri"
use_earliest_if_visit_absent <- TRUE

outdir <- "out/ADNI/projected_rosmap_msbb_22gene_signature/imaging_UCSF_FreeSurfer_covariate_adjusted_primary"
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

contrast_definitions <- list(
  cluster1_vs_non_resilience = c("cluster1" = 1, "cluster2" = 0, "non_resilience" = -1),
  cluster2_vs_non_resilience = c("cluster1" = 0, "cluster2" = 1, "non_resilience" = -1),
  cluster1_vs_cluster2       = c("cluster1" = 1, "cluster2" = -1, "non_resilience" = 0)
)

contrast_colors <- c(
  "cluster1_vs_non_resilience" = group_colors[["cluster1"]],
  "cluster2_vs_non_resilience" = group_colors[["cluster2"]],
  "cluster1_vs_cluster2" = "#4D4D4D"
)

use_original_st_rule <- TRUE
manual_volume_vars <- c()

base_covariates_preferred <- c(
  "AGE_AT_MRI",
  "SEX",
  "PTEDUCAT_clean",
  "APOE_e4_carrier",
  "PHASE",
  "MANUFACTURER",
  "FIELD_STRENGTH",
  "MAGNETICFIELDSTRENGTH"
)

include_extended_scanner_covariates <- FALSE
extended_scanner_covariates <- c(
  "MANUFACTURERSMODELNAME",
  "MRACQUISITIONTYPE",
  "MRACQUISITIONTYPEFLAIR",
  "MFSVERSION",
  "FSVERSION"
)

icv_candidates_explicit <- c(
  "eTIV", "ETIV", "ICV", "EstimatedTotalIntraCranialVol",
  "EstimatedTotalIntracranialVol", "IntracranialVol", "ST10CV", "ST10CVS"
)
use_icv_covariate_for_volume_metrics <- TRUE

run_kruskal_dunn <- TRUE

q_plot_threshold <- 0.05
plot_top_if_no_q_significant <- TRUE
top_n_if_no_q_significant <- 30

forest_contrasts <- c(
  "cluster1_vs_non_resilience",
  "cluster2_vs_non_resilience",
  "cluster1_vs_cluster2"
)

make_qc_boxplots <- TRUE
boxplot_top_n <- 24
base_font_size <- 18
facet_ncol <- 4
point_size <- 0.7
jitter_width <- 0.2

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(data.table)
  library(tibble)
  library(lubridate)
  library(emmeans)
  library(dunn.test)
  library(forcats)
  library(stringr)
})

clean_numeric <- function(x) {
  x <- as.character(x)
  x[x %in% c("", "NA", "NaN", "NULL", "null", "-4", ".", " ")] <- NA
  suppressWarnings(as.numeric(x))
}

safe_factor <- function(x) {
  x <- as.character(x)
  x[x %in% c("", "NA", "NaN", "NULL", "null", ".", " ")] <- NA
  as.factor(x)
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

model_sigma <- function(fit) {
  s <- tryCatch(summary(fit)$sigma, error = function(e) NA_real_)
  as.numeric(s)
}

partial_eta2_from_anova <- function(fit_reduced, fit_full) {
  an <- tryCatch(anova(fit_reduced, fit_full), error = function(e) NULL)
  if (is.null(an) || nrow(an) < 2) return(NA_real_)
  ss_effect <- an$RSS[1] - an$RSS[2]
  ss_error <- an$RSS[2]
  if (is.na(ss_effect) || is.na(ss_error) || (ss_effect + ss_error) <= 0) return(NA_real_)
  ss_effect / (ss_effect + ss_error)
}

is_volume_metric <- function(v) {
  TRUE
}

format_q_label <- function(q) {
  dplyr::case_when(
    is.na(q) ~ "q = NA",
    q < 0.001 ~ "q < 0.001",
    TRUE ~ paste0("q = ", sprintf("%.3f", q))
  )
}

add_standardized_ci <- function(df) {
  df %>%
    mutate(
      model_sigma_reconstructed = dplyr::case_when(
        !is.na(estimate) & !is.na(standardized_difference) & abs(standardized_difference) > 1e-12 ~ estimate / standardized_difference,
        TRUE ~ NA_real_
      ),
      std_lower.CL = dplyr::case_when(
        !is.na(model_sigma_reconstructed) & abs(model_sigma_reconstructed) > 1e-12 ~ lower.CL / model_sigma_reconstructed,
        TRUE ~ NA_real_
      ),
      std_upper.CL = dplyr::case_when(
        !is.na(model_sigma_reconstructed) & abs(model_sigma_reconstructed) > 1e-12 ~ upper.CL / model_sigma_reconstructed,
        TRUE ~ NA_real_
      ),
      se_std = dplyr::case_when(
        !is.na(model_sigma_reconstructed) & abs(model_sigma_reconstructed) > 1e-12 ~ SE / model_sigma_reconstructed,
        TRUE ~ NA_real_
      ),
      std_lower.CL = ifelse(is.na(std_lower.CL) & !is.na(se_std), standardized_difference - 1.96 * se_std, std_lower.CL),
      std_upper.CL = ifelse(is.na(std_upper.CL) & !is.na(se_std), standardized_difference + 1.96 * se_std, std_upper.CL)
    )
}

safe_metric_label <- function(variable, ids_df = NULL) {
  if (is.null(ids_df) || !("id" %in% colnames(ids_df))) return(variable)
  name_col <- first_present(c("name", "label", "description", "Description", "region", "Region"), colnames(ids_df))
  if (is.na(name_col)) return(variable)
  lab <- ids_df[[name_col]][match(variable, ids_df$id)]
  lab[is.na(lab) | lab == ""] <- variable[is.na(lab) | lab == ""]
  lab <- gsub("_of_", " of ", lab)
  lab <- gsub("_", " ", lab)
  lab
}

find_icv_covariate <- function(vol_df, ids_df, explicit_candidates) {
  hit <- first_present(explicit_candidates, colnames(vol_df))
  if (!is.na(hit)) return(hit)

  if (!is.null(ids_df) && all(c("id") %in% colnames(ids_df))) {
    label_cols <- intersect(c("name", "label", "description", "Description", "region", "Region"), colnames(ids_df))
    if (length(label_cols) > 0) {
      lab_txt <- apply(ids_df[, label_cols, drop = FALSE], 1, paste, collapse = " ")
      idx <- grep("eTIV|intracranial|intra cranial|ICV|estimated total", lab_txt, ignore.case = TRUE)
      ids_hit <- ids_df$id[idx]
      ids_hit <- ids_hit[ids_hit %in% colnames(vol_df)]
      if (length(ids_hit) > 0) return(ids_hit[1])
    }
  }
  NA_character_
}

message("[1] Loading current ADNI projection table")

if (file.exists(adni_projection_file)) {
  adni <- read.csv(adni_projection_file, check.names = FALSE)
  projection_source <- adni_projection_file
} else if (file.exists(adni_projection_all_file)) {
  warning("ADpatho projection table not found. Falling back to all table and filtering ADpatho == 'AD'.")
  adni <- read.csv(adni_projection_all_file, check.names = FALSE)
  projection_source <- adni_projection_all_file
  if (!("ADpatho" %in% colnames(adni))) {
    stop("Fallback all table does not contain ADpatho column.")
  }
  adni <- adni %>% filter(ADpatho == "AD")
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
    imaging_group = case_when(
      resilience == "Low" ~ "non_resilience",
      resilience == "High" & predicted_cluster_like %in% c("cluster1", "cluster2") ~ predicted_cluster_like,
      TRUE ~ NA_character_
    ),
    imaging_group = factor(imaging_group, levels = group_levels)
  ) %>%
  filter(!is.na(imaging_group), imaging_group %in% group_levels)

if ("sex" %in% colnames(adni)) {
  adni$SEX <- harmonize_sex(adni$sex)
} else if ("PTGENDER" %in% colnames(adni)) {
  adni$SEX <- harmonize_sex(adni$PTGENDER)
} else {
  adni$SEX <- NA_character_
}

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

apoe_proj_col <- first_present(c("GENOTYPE", "APOE", "APOEGENOTYPE", "APOE4", "APOE_e4_carrier"), colnames(adni))
if (!is.na(apoe_proj_col) && apoe_proj_col != "APOE_e4_carrier") {
  adni$APOE_e4_carrier <- make_apoe4(adni[[apoe_proj_col]])
} else if ("APOE_e4_carrier" %in% colnames(adni)) {
  x <- as.character(adni$APOE_e4_carrier)
  adni$APOE_e4_carrier <- dplyr::case_when(
    x %in% c("1", "Carrier", "carrier", "E4", "e4") ~ "Carrier",
    x %in% c("0", "Non-carrier", "Noncarrier", "non-carrier", "noncarrier") ~ "Non-carrier",
    TRUE ~ NA_character_
  )
} else {
  adni$APOE_e4_carrier <- NA_character_
}

message("    Projection source: ", projection_source)
message("    Current group distribution:")
print(table(adni$imaging_group, useNA = "ifany"))

message("[2] Adding PTDEMOG/APOE metadata")

if (file.exists(ptdemog_file)) {
  ptdemog <- read.csv(ptdemog_file, check.names = FALSE)
  ptdemog_sub <- ptdemog %>%
    mutate(RID = as.integer(RID)) %>%
    select(any_of(c("RID", "PTGENDER", "PTEDUCAT", "PTDOB", "PTDOBYY", "VISDATE", "VISCODE", "VISCODE2"))) %>%
    arrange(RID) %>%
    group_by(RID) %>%
    slice(1) %>%
    ungroup()

  adni <- adni %>% left_join(ptdemog_sub, by = "RID", suffix = c("", ".ptdemog"))

  if ("PTGENDER" %in% colnames(adni)) {
    sex_pt <- harmonize_sex(adni$PTGENDER)
    adni$SEX <- ifelse(is.na(adni$SEX), sex_pt, adni$SEX)
  }
  if ("PTEDUCAT.ptdemog" %in% colnames(adni)) {
    edu_pt <- clean_numeric(adni$PTEDUCAT.ptdemog)
    adni$PTEDUCAT_clean <- ifelse(is.na(adni$PTEDUCAT_clean), edu_pt, adni$PTEDUCAT_clean)
  }
} else {
  warning("PTDEMOG file not found: ", ptdemog_file)
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

adni$SEX <- factor(adni$SEX, levels = c("Male", "Female"))
adni$APOE_e4_carrier <- factor(adni$APOE_e4_carrier, levels = c("Non-carrier", "Carrier"))

write.csv(
  adni,
  file.path(table_dir, "ADNI_current_projected_groups_for_UCSF_FreeSurfer.csv"),
  row.names = FALSE,
  quote = FALSE
)

message("[3] Loading UCSF FreeSurfer data")

if (!file.exists(volume_file)) stop("Volume file not found: ", volume_file)
if (!file.exists(freesurfer_id_file)) stop("FreeSurfer id file not found: ", freesurfer_id_file)

vol <- read.csv(volume_file, check.names = FALSE)
ids <- read.csv(freesurfer_id_file, check.names = FALSE)

if (!("RID" %in% colnames(vol))) stop("Volume file must contain RID.")
if (!("VISCODE2" %in% colnames(vol))) stop("Volume file must contain VISCODE2.")

vol <- vol %>%
  mutate(
    RID = as.integer(RID),
    EXAMDATE_parsed = if ("EXAMDATE" %in% colnames(.)) suppressWarnings(lubridate::parse_date_time(EXAMDATE, orders = c("Y/m/d", "m/d/Y", "Y-m-d", "d/m/Y"))) else as.POSIXct(NA)
  )

if (!is.null(volume_visit)) {
  vol_visit <- vol %>% filter(VISCODE2 == volume_visit)
  if (nrow(vol_visit) == 0 && use_earliest_if_visit_absent) {
    warning("No UCSF rows found for VISCODE2 == '", volume_visit, "'. Using earliest available scan per RID instead.")
    vol_visit <- vol %>% arrange(RID, EXAMDATE_parsed, VISCODE2) %>% group_by(RID) %>% slice(1) %>% ungroup()
    volume_visit_used <- "earliest_available"
  } else {
    volume_visit_used <- volume_visit
  }
} else {
  vol_visit <- vol %>% arrange(RID, EXAMDATE_parsed, VISCODE2) %>% group_by(RID) %>% slice(1) %>% ungroup()
  volume_visit_used <- "earliest_available"
}

vol_visit <- vol_visit %>% arrange(RID, EXAMDATE_parsed) %>% distinct(RID, .keep_all = TRUE)

if (use_original_st_rule) {
  vars <- names(vol_visit)[
    grepl("ST", names(vol_visit)) &
      names(vol_visit) != "STATUS" &
      names(vol_visit) != "FIELD_STRENGTH" &
      !grepl("SA$", names(vol_visit)) &
      !grepl("TS$", names(vol_visit))
  ]
} else {
  vars <- manual_volume_vars[manual_volume_vars %in% colnames(vol_visit)]
}

vars <- setdiff(vars, icv_candidates_explicit)
vars <- vars[vars %in% colnames(vol_visit)]

if (length(vars) == 0) stop("No UCSF FreeSurfer variables were selected.")

for (v in vars) vol_visit[[v]] <- clean_numeric(vol_visit[[v]])

scanner_cols <- c("PHASE", "MANUFACTURER", "MANUFACTURERSMODELNAME", "FIELD_STRENGTH", "MAGNETICFIELDSTRENGTH", "MRACQUISITIONTYPE", "MRACQUISITIONTYPEFLAIR", "MFSVERSION", "FSVERSION")
for (v in scanner_cols) {
  if (v %in% colnames(vol_visit)) vol_visit[[v]] <- droplevels(safe_factor(vol_visit[[v]]))
}

icv_covariate <- find_icv_covariate(vol_visit, ids, icv_candidates_explicit)
if (!is.na(icv_covariate)) {
  vol_visit[[icv_covariate]] <- clean_numeric(vol_visit[[icv_covariate]])
}

message("    UCSF visit used: ", volume_visit_used)
message("    UCSF variables selected: ", length(vars))
message("    ICV/eTIV covariate candidate: ", ifelse(is.na(icv_covariate), "none", icv_covariate))

message("[4] Merging projected groups with UCSF FreeSurfer data")

df_merged <- adni %>% left_join(vol_visit, by = "RID", suffix = c(".adni", ".fs"))

if ("EXAMDATE_parsed" %in% colnames(df_merged) && "PTDOB" %in% colnames(df_merged)) {
  ptdob_date <- suppressWarnings(lubridate::parse_date_time(df_merged$PTDOB, orders = c("m/Y", "m/d/Y", "Y-m-d", "d/m/Y", "b/Y")))
  df_merged$AGE_AT_MRI_from_dob <- ifelse(
    !is.na(ptdob_date) & !is.na(df_merged$EXAMDATE_parsed),
    as.numeric(lubridate::interval(ptdob_date, df_merged$EXAMDATE_parsed) / lubridate::years(1)),
    NA_real_
  )
} else {
  df_merged$AGE_AT_MRI_from_dob <- NA_real_
}

if ("PTDOBYY" %in% colnames(df_merged) && "EXAMDATE_parsed" %in% colnames(df_merged)) {
  birth_year <- clean_numeric(df_merged$PTDOBYY)
  exam_year <- lubridate::year(df_merged$EXAMDATE_parsed)
  df_merged$AGE_AT_MRI_from_year <- ifelse(!is.na(birth_year) & !is.na(exam_year), exam_year - birth_year, NA_real_)
} else {
  df_merged$AGE_AT_MRI_from_year <- NA_real_
}

df_merged$AGE_AT_MRI <- dplyr::case_when(
  !is.na(df_merged$AGE_AT_MRI_from_dob) ~ df_merged$AGE_AT_MRI_from_dob,
  !is.na(df_merged$AGE_AT_MRI_from_year) ~ df_merged$AGE_AT_MRI_from_year,
  !is.na(df_merged$AGE_AT_PROJECTION) ~ df_merged$AGE_AT_PROJECTION,
  TRUE ~ NA_real_
)

df_merged$imaging_group <- factor(df_merged$imaging_group, levels = group_levels)
df_merged$SEX <- factor(df_merged$SEX, levels = c("Male", "Female"))
df_merged$APOE_e4_carrier <- factor(df_merged$APOE_e4_carrier, levels = c("Non-carrier", "Carrier"))
for (v in scanner_cols) {
  if (v %in% colnames(df_merged)) df_merged[[v]] <- droplevels(safe_factor(df_merged[[v]]))
}
if (!is.na(icv_covariate) && icv_covariate %in% colnames(df_merged)) {
  df_merged[[icv_covariate]] <- clean_numeric(df_merged[[icv_covariate]])
}

covariates_preferred <- base_covariates_preferred
if (include_extended_scanner_covariates) {
  covariates_preferred <- unique(c(covariates_preferred, extended_scanner_covariates))
}

write.csv(
  df_merged,
  file.path(table_dir, "ADNI_current_groups_merged_UCSF_FreeSurfer_with_covariates.csv"),
  row.names = FALSE,
  quote = FALSE
)

message("    Merged samples: ", nrow(df_merged))
message("    Samples with any selected FreeSurfer metric: ", sum(rowSums(!is.na(df_merged[, vars, drop = FALSE])) > 0))
message("    Group distribution among samples with any selected metric:")
print(table(df_merged$imaging_group[rowSums(!is.na(df_merged[, vars, drop = FALSE])) > 0], useNA = "ifany"))

nonmissing_counts <- df_merged %>%
  select(RID, imaging_group, all_of(vars)) %>%
  pivot_longer(cols = all_of(vars), names_to = "variable", values_to = "value") %>%
  filter(!is.na(value)) %>%
  count(imaging_group, variable, name = "n_nonmissing")
write.csv(nonmissing_counts, file.path(table_dir, "UCSF_FreeSurfer_nonmissing_counts_by_group.csv"), row.names = FALSE, quote = FALSE)

covariate_summary <- data.frame(
  variable = unique(c("AGE_AT_MRI", "SEX", "PTEDUCAT_clean", "APOE_e4_carrier", "PHASE", "MANUFACTURER", "FIELD_STRENGTH", "MAGNETICFIELDSTRENGTH", icv_covariate)),
  present = NA,
  n_nonmissing = NA,
  n_missing = NA,
  n_unique = NA,
  stringsAsFactors = FALSE
) %>%
  filter(!is.na(variable)) %>%
  rowwise() %>%
  mutate(
    present = variable %in% colnames(df_merged),
    n_nonmissing = ifelse(present, sum(!is.na(df_merged[[variable]])), NA_integer_),
    n_missing = ifelse(present, sum(is.na(df_merged[[variable]])), NA_integer_),
    n_unique = ifelse(present, length(unique(df_merged[[variable]][!is.na(df_merged[[variable]])])), NA_integer_)
  ) %>%
  ungroup()
write.csv(covariate_summary, file.path(table_dir, "UCSF_FreeSurfer_covariate_availability_summary.csv"), row.names = FALSE, quote = FALSE)

message("[5] Writing descriptive summaries")

desc_summary <- df_merged %>%
  select(RID, imaging_group, all_of(vars)) %>%
  pivot_longer(cols = all_of(vars), names_to = "variable", values_to = "value") %>%
  group_by(variable, imaging_group) %>%
  summarise(
    n = sum(!is.na(value)),
    mean = mean(value, na.rm = TRUE),
    sd = sd(value, na.rm = TRUE),
    median = median(value, na.rm = TRUE),
    q1 = quantile(value, 0.25, na.rm = TRUE),
    q3 = quantile(value, 0.75, na.rm = TRUE),
    .groups = "drop"
  )

if ("id" %in% colnames(ids)) {
  desc_summary <- left_join(desc_summary, ids, by = c("variable" = "id"))
}

write.csv(desc_summary, file.path(table_dir, "UCSF_FreeSurfer_descriptive_summary_by_group.csv"), row.names = FALSE, quote = FALSE)

message("[6] Running primary covariate-adjusted linear models")

primary_omnibus_rows <- list()
primary_contrast_rows <- list()
coefficient_rows <- list()

for (variable in vars) {
  df_work <- df_merged
  df_work[[variable]] <- clean_numeric(df_work[[variable]])

  covars_metric <- covariates_preferred
  if (use_icv_covariate_for_volume_metrics && !is.na(icv_covariate) && is_volume_metric(variable) && variable != icv_covariate) {
    covars_metric <- unique(c(covars_metric, icv_covariate))
  }
  covars_metric <- keep_usable_covariates(df_work, covars_metric)

  use_cols <- unique(c(variable, "imaging_group", covars_metric))
  d <- df_work[, use_cols, drop = FALSE]
  d$imaging_group <- factor(d$imaging_group, levels = group_levels)

  for (cv in covars_metric) {
    if (cv %in% c("SEX", "APOE_e4_carrier", "PHASE", "MANUFACTURER", "MANUFACTURERSMODELNAME", "FIELD_STRENGTH", "MAGNETICFIELDSTRENGTH", "MRACQUISITIONTYPE", "MRACQUISITIONTYPEFLAIR", "MFSVERSION", "FSVERSION")) {
      d[[cv]] <- droplevels(safe_factor(d[[cv]]))
    } else {
      d[[cv]] <- clean_numeric(d[[cv]])
    }
  }

  d <- d[complete.cases(d), , drop = FALSE]
  d$imaging_group <- droplevels(d$imaging_group)

  usable_covars <- c()
  for (cv in covars_metric) {
    if (!(cv %in% colnames(d))) next
    z <- d[[cv]]
    if (is.factor(z) && nlevels(droplevels(z)) < 2) next
    if (!is.factor(z) && length(unique(z)) < 2) next
    usable_covars <- c(usable_covars, cv)
  }

  if (nrow(d) < 10 || length(unique(d$imaging_group)) < 2 || length(unique(d[[variable]])) < 2) {
    primary_omnibus_rows[[length(primary_omnibus_rows) + 1]] <- data.frame(
      variable = variable,
      test = "lm_nested_anova_group",
      statistic = NA_real_,
      df1 = NA_real_,
      df2 = NA_real_,
      p.value = NA_real_,
      partial_eta2 = NA_real_,
      n = nrow(d),
      formula = NA_character_,
      covariates = paste(usable_covars, collapse = ";"),
      note = "insufficient_data",
      stringsAsFactors = FALSE
    )
    for (cname in names(contrast_definitions)) {
      primary_contrast_rows[[length(primary_contrast_rows) + 1]] <- data.frame(
        variable = variable,
        contrast = cname,
        estimate = NA_real_,
        SE = NA_real_,
        df = NA_real_,
        t.ratio = NA_real_,
        p.value = NA_real_,
        lower.CL = NA_real_,
        upper.CL = NA_real_,
        standardized_difference = NA_real_,
        n = nrow(d),
        formula = NA_character_,
        covariates = paste(usable_covars, collapse = ";"),
        note = "insufficient_data",
        stringsAsFactors = FALSE
      )
    }
    next
  }

  full_rhs <- c("imaging_group", usable_covars)
  full_form_txt <- paste(variable, "~", paste(full_rhs, collapse = " + "))
  reduced_form_txt <- if (length(usable_covars) == 0) paste(variable, "~ 1") else paste(variable, "~", paste(usable_covars, collapse = " + "))

  fit_full <- tryCatch(lm(as.formula(full_form_txt), data = d), error = function(e) NULL)
  fit_reduced <- tryCatch(lm(as.formula(reduced_form_txt), data = d), error = function(e) NULL)
  an <- if (is.null(fit_full) || is.null(fit_reduced)) NULL else tryCatch(anova(fit_reduced, fit_full), error = function(e) NULL)

  if (is.null(fit_full) || is.null(fit_reduced) || is.null(an)) {
    primary_omnibus_rows[[length(primary_omnibus_rows) + 1]] <- data.frame(
      variable = variable,
      test = "lm_nested_anova_group",
      statistic = NA_real_,
      df1 = NA_real_,
      df2 = NA_real_,
      p.value = NA_real_,
      partial_eta2 = NA_real_,
      n = nrow(d),
      formula = full_form_txt,
      covariates = paste(usable_covars, collapse = ";"),
      note = "model_failed",
      stringsAsFactors = FALSE
    )
    for (cname in names(contrast_definitions)) {
      primary_contrast_rows[[length(primary_contrast_rows) + 1]] <- data.frame(
        variable = variable,
        contrast = cname,
        estimate = NA_real_,
        SE = NA_real_,
        df = NA_real_,
        t.ratio = NA_real_,
        p.value = NA_real_,
        lower.CL = NA_real_,
        upper.CL = NA_real_,
        standardized_difference = NA_real_,
        n = nrow(d),
        formula = full_form_txt,
        covariates = paste(usable_covars, collapse = ";"),
        note = "model_failed",
        stringsAsFactors = FALSE
      )
    }
    next
  }

  p_group <- an$`Pr(>F)`[2]
  f_stat <- an$F[2]
  df1 <- an$Df[2]
  df2 <- an$Res.Df[2]
  petasq <- partial_eta2_from_anova(fit_reduced, fit_full)

  primary_omnibus_rows[[length(primary_omnibus_rows) + 1]] <- data.frame(
    variable = variable,
    test = "lm_nested_anova_group",
    statistic = f_stat,
    df1 = df1,
    df2 = df2,
    p.value = p_group,
    partial_eta2 = petasq,
    n = nrow(d),
    formula = full_form_txt,
    covariates = paste(usable_covars, collapse = ";"),
    note = "",
    stringsAsFactors = FALSE
  )

  coef_df <- tryCatch({
    sm <- summary(fit_full)$coefficients
    data.frame(
      variable = variable,
      term = rownames(sm),
      estimate = sm[, "Estimate"],
      statistic = sm[, "t value"],
      p.value = sm[, "Pr(>|t|)"],
      n = nrow(d),
      formula = full_form_txt,
      stringsAsFactors = FALSE
    )
  }, error = function(e) NULL)
  if (!is.null(coef_df)) coefficient_rows[[length(coefficient_rows) + 1]] <- coef_df

  emm <- tryCatch(emmeans::emmeans(fit_full, ~ imaging_group), error = function(e) NULL)
  sigma_fit <- model_sigma(fit_full)

  for (cname in names(contrast_definitions)) {
    if (is.null(emm)) {
      primary_contrast_rows[[length(primary_contrast_rows) + 1]] <- data.frame(
        variable = variable,
        contrast = cname,
        estimate = NA_real_, SE = NA_real_, df = NA_real_, t.ratio = NA_real_, p.value = NA_real_,
        lower.CL = NA_real_, upper.CL = NA_real_, standardized_difference = NA_real_,
        n = nrow(d), formula = full_form_txt, covariates = paste(usable_covars, collapse = ";"),
        note = "emmeans_failed", stringsAsFactors = FALSE
      )
      next
    }

    weights <- contrast_definitions[[cname]]
    present_groups <- levels(droplevels(d$imaging_group))
    if (!all(names(weights)[weights != 0] %in% present_groups)) {
      primary_contrast_rows[[length(primary_contrast_rows) + 1]] <- data.frame(
        variable = variable,
        contrast = cname,
        estimate = NA_real_, SE = NA_real_, df = NA_real_, t.ratio = NA_real_, p.value = NA_real_,
        lower.CL = NA_real_, upper.CL = NA_real_, standardized_difference = NA_real_,
        n = nrow(d), formula = full_form_txt, covariates = paste(usable_covars, collapse = ";"),
        note = "contrast_groups_not_all_present", stringsAsFactors = FALSE
      )
      next
    }

    w <- weights[present_groups]
    contr <- tryCatch(emmeans::contrast(emm, method = list(tmp = w)), error = function(e) NULL)
    contr_sum <- if (is.null(contr)) NULL else tryCatch(summary(contr, infer = c(TRUE, TRUE), adjust = "none"), error = function(e) NULL)

    if (is.null(contr_sum)) {
      primary_contrast_rows[[length(primary_contrast_rows) + 1]] <- data.frame(
        variable = variable,
        contrast = cname,
        estimate = NA_real_, SE = NA_real_, df = NA_real_, t.ratio = NA_real_, p.value = NA_real_,
        lower.CL = NA_real_, upper.CL = NA_real_, standardized_difference = NA_real_,
        n = nrow(d), formula = full_form_txt, covariates = paste(usable_covars, collapse = ";"),
        note = "contrast_failed", stringsAsFactors = FALSE
      )
    } else {
      primary_contrast_rows[[length(primary_contrast_rows) + 1]] <- data.frame(
        variable = variable,
        contrast = cname,
        estimate = contr_sum$estimate[1],
        SE = contr_sum$SE[1],
        df = contr_sum$df[1],
        t.ratio = contr_sum$t.ratio[1],
        p.value = contr_sum$p.value[1],
        lower.CL = contr_sum$lower.CL[1],
        upper.CL = contr_sum$upper.CL[1],
        standardized_difference = ifelse(!is.na(sigma_fit) && sigma_fit > 0, contr_sum$estimate[1] / sigma_fit, NA_real_),
        n = nrow(d),
        formula = full_form_txt,
        covariates = paste(usable_covars, collapse = ";"),
        note = "",
        stringsAsFactors = FALSE
      )
    }
  }
}

primary_omnibus <- bind_rows(primary_omnibus_rows) %>%
  mutate(
    FDR_across_metrics = p.adjust(p.value, method = "BH"),
    p_stars = vapply(p.value, p_to_stars, character(1)),
    q_stars = vapply(FDR_across_metrics, p_to_stars, character(1))
  )

primary_contrasts <- bind_rows(primary_contrast_rows) %>%
  mutate(
    FDR_all_metrics_all_contrasts = p.adjust(p.value, method = "BH"),
    p_stars = vapply(p.value, p_to_stars, character(1)),
    q_stars = vapply(FDR_all_metrics_all_contrasts, p_to_stars, character(1))
  ) %>%
  add_standardized_ci()

primary_coefficients <- bind_rows(coefficient_rows)
if (nrow(primary_coefficients) > 0) {
  primary_coefficients <- primary_coefficients %>%
    group_by(variable) %>%
    mutate(FDR_within_metric = p.adjust(p.value, method = "BH")) %>%
    ungroup()
}

if ("id" %in% colnames(ids)) {
  primary_omnibus <- left_join(primary_omnibus, ids, by = c("variable" = "id"))
  primary_contrasts <- left_join(primary_contrasts, ids, by = c("variable" = "id"))
  primary_coefficients <- left_join(primary_coefficients, ids, by = c("variable" = "id"))
}

write.csv(primary_omnibus, file.path(table_dir, "UCSF_FreeSurfer_primary_adjusted_LM_omnibus_group_effect.csv"), row.names = FALSE, quote = FALSE)
write.csv(primary_contrasts, file.path(table_dir, "UCSF_FreeSurfer_primary_adjusted_LM_pairwise_contrasts_all_metrics_BH_FDR.csv"), row.names = FALSE, quote = FALSE)
write.csv(primary_coefficients, file.path(table_dir, "UCSF_FreeSurfer_primary_adjusted_LM_coefficients_long.csv"), row.names = FALSE, quote = FALSE)

sig_counts_primary <- primary_contrasts %>%
  mutate(significant_q05 = !is.na(FDR_all_metrics_all_contrasts) & FDR_all_metrics_all_contrasts < 0.05) %>%
  group_by(contrast) %>%
  summarise(
    n_tested = sum(!is.na(p.value)),
    n_significant_q05 = sum(significant_q05),
    significant_metrics_q05 = paste(variable[significant_q05], collapse = ";"),
    .groups = "drop"
  )
write.csv(sig_counts_primary, file.path(table_dir, "UCSF_FreeSurfer_primary_significant_counts_by_contrast.csv"), row.names = FALSE, quote = FALSE)

message("    Primary pairwise significant counts:")
print(sig_counts_primary)

if (run_kruskal_dunn) {
  message("[7] Running secondary distribution-free tests")

  kw_rows <- list()
  dunn_rows <- list()

  for (variable in vars) {
    testdata <- df_merged %>%
      select(imaging_group, all_of(variable)) %>%
      filter(!is.na(imaging_group), !is.na(.data[[variable]]))
    testdata[[variable]] <- clean_numeric(testdata[[variable]])
    testdata <- testdata %>% filter(!is.na(.data[[variable]]))

    if (nrow(testdata) < 3 || length(unique(testdata$imaging_group)) < 2) {
      kw_rows[[variable]] <- data.frame(
        variable = variable,
        test = "Kruskal-Wallis",
        statistic = NA_real_,
        df = NA_real_,
        p.value = NA_real_,
        n = nrow(testdata),
        note = "insufficient_data",
        stringsAsFactors = FALSE
      )
      next
    }

    kw <- tryCatch(kruskal.test(as.formula(paste(variable, "~ imaging_group")), data = testdata), error = function(e) NULL)
    kw_rows[[variable]] <- data.frame(
      variable = variable,
      test = "Kruskal-Wallis",
      statistic = if (is.null(kw)) NA_real_ else as.numeric(kw$statistic),
      df = if (is.null(kw)) NA_real_ else as.numeric(kw$parameter),
      p.value = if (is.null(kw)) NA_real_ else kw$p.value,
      n = nrow(testdata),
      note = if (is.null(kw)) "test_failed" else "",
      stringsAsFactors = FALSE
    )

    if (length(unique(testdata$imaging_group)) >= 3) {
      dn <- tryCatch(dunn.test::dunn.test(testdata[[variable]], testdata$imaging_group, method = "bonferroni", kw = FALSE), error = function(e) NULL)
      if (!is.null(dn)) {
        dunn_rows[[variable]] <- data.frame(
          variable = variable,
          comparison = dn$comparisons,
          p.value = dn$P.adjusted,
          note = "",
          stringsAsFactors = FALSE
        )
      } else {
        dunn_rows[[variable]] <- data.frame(
          variable = variable,
          comparison = NA_character_,
          p.value = NA_real_,
          note = "test_failed",
          stringsAsFactors = FALSE
        )
      }
    }
  }

  kw_summary <- bind_rows(kw_rows) %>%
    mutate(
      FDR_across_metrics = p.adjust(p.value, method = "BH"),
      p_stars = vapply(p.value, p_to_stars, character(1)),
      q_stars = vapply(FDR_across_metrics, p_to_stars, character(1))
    )
  dunn_summary <- bind_rows(dunn_rows) %>%
    mutate(
      FDR_all_metrics_all_comparisons = p.adjust(p.value, method = "BH"),
      p_stars = vapply(p.value, p_to_stars, character(1)),
      q_stars = vapply(FDR_all_metrics_all_comparisons, p_to_stars, character(1))
    )

  if ("id" %in% colnames(ids)) {
    kw_summary <- left_join(kw_summary, ids, by = c("variable" = "id"))
    dunn_summary <- left_join(dunn_summary, ids, by = c("variable" = "id"))
  }

  write.csv(kw_summary, file.path(table_dir, "UCSF_FreeSurfer_secondary_kruskal_summary.csv"), row.names = FALSE, quote = FALSE)
  write.csv(dunn_summary, file.path(table_dir, "UCSF_FreeSurfer_secondary_dunn_bonferroni_with_global_BH_FDR.csv"), row.names = FALSE, quote = FALSE)
}

message("[8] Drawing forest plots for significant primary contrasts")

plot_df <- primary_contrasts %>%
  filter(contrast %in% forest_contrasts) %>%
  mutate(
    significant_q05 = !is.na(FDR_all_metrics_all_contrasts) & FDR_all_metrics_all_contrasts < q_plot_threshold,
    variable_label = safe_metric_label(variable, ids),
    contrast_label = contrast_label(contrast),
    contrast_label = factor(contrast_label, levels = contrast_label(forest_contrasts)),
    q_label = format_q_label(FDR_all_metrics_all_contrasts)
  )

plot_sig <- plot_df %>% filter(significant_q05)

if (nrow(plot_sig) == 0 && plot_top_if_no_q_significant) {
  warning("No BH-FDR significant contrasts found. Plotting top ", top_n_if_no_q_significant, " contrasts by q-value instead.")
  plot_sig <- plot_df %>%
    filter(!is.na(FDR_all_metrics_all_contrasts)) %>%
    arrange(FDR_all_metrics_all_contrasts) %>%
    slice_head(n = top_n_if_no_q_significant) %>%
    mutate(significant_q05 = FALSE)
}

write.csv(plot_sig, file.path(forest_dir, "UCSF_FreeSurfer_primary_forest_plot_rows_significant_or_top.csv"), row.names = FALSE, quote = FALSE)

if (nrow(plot_sig) > 0) {
  variable_order <- plot_sig %>%
    group_by(variable_label) %>%
    summarise(
      min_q = min(FDR_all_metrics_all_contrasts, na.rm = TRUE),
      max_abs_effect = max(abs(standardized_difference), na.rm = TRUE),
      .groups = "drop"
    ) %>%
    arrange(min_q, desc(max_abs_effect)) %>%
    pull(variable_label)

  plot_sig <- plot_sig %>%
    mutate(
      variable_label = factor(variable_label, levels = rev(variable_order)),
      label_text = ifelse(significant_q05, paste0(q_stars, " ", q_label), q_label)
    )

  n_metrics <- length(unique(plot_sig$variable_label))
  forest_height <- max(4.5, min(18, 0.32 * n_metrics + 2.5))
  forest_width <- 10.5

  p_forest <- ggplot(
    plot_sig,
    aes(
      x = standardized_difference,
      y = variable_label,
      color = contrast_label,
      shape = significant_q05
    )
  ) +
    geom_vline(xintercept = 0, linetype = "dashed", linewidth = 0.5, color = "gray40") +
    geom_errorbarh(
      aes(xmin = std_lower.CL, xmax = std_upper.CL),
      height = 0.18,
      linewidth = 0.65,
      position = position_dodge(width = 0.65),
      na.rm = TRUE
    ) +
    geom_point(
      size = 2.8,
      stroke = 0.8,
      position = position_dodge(width = 0.65),
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
      title = "UCSF FreeSurfer regional volumes: significant adjusted contrasts",
      subtitle = "Primary covariate-adjusted models; plotted rows are BH-FDR significant unless no significant rows are available",
      x = "Standardized adjusted difference",
      y = NULL
    ) +
    theme_bw(base_size = 14) +
    theme(
      plot.title = element_text(face = "bold", size = 16),
      plot.subtitle = element_text(size = 11),
      axis.text.y = element_text(size = 10),
      axis.text.x = element_text(size = 12),
      axis.title.x = element_text(size = 13),
      legend.position = "right",
      legend.title = element_text(size = 11),
      legend.text = element_text(size = 10),
      panel.grid.major.y = element_line(color = "gray90"),
      panel.grid.minor = element_blank()
    )

  ggsave(file.path(forest_dir, "UCSF_FreeSurfer_primary_forest_significant_adjusted_contrasts.svg"), p_forest, width = forest_width, height = forest_height, units = "in")
  ggsave(file.path(forest_dir, "UCSF_FreeSurfer_primary_forest_significant_adjusted_contrasts.pdf"), p_forest, width = forest_width, height = forest_height, units = "in")
  ggsave(file.path(forest_dir, "UCSF_FreeSurfer_primary_forest_significant_adjusted_contrasts.png"), p_forest, width = forest_width, height = forest_height, units = "in", dpi = 300)

  p_forest_facet <- ggplot(
    plot_sig,
    aes(x = standardized_difference, y = variable_label, color = contrast_label)
  ) +
    geom_vline(xintercept = 0, linetype = "dashed", linewidth = 0.5, color = "gray40") +
    geom_errorbarh(aes(xmin = std_lower.CL, xmax = std_upper.CL), height = 0.18, linewidth = 0.65, na.rm = TRUE) +
    geom_point(size = 2.5, na.rm = TRUE) +
    facet_wrap(~ contrast_label, scales = "free_y", ncol = 1) +
    scale_color_manual(values = contrast_label_colors, guide = "none") +
    labs(
      title = "UCSF FreeSurfer regional volumes: significant adjusted contrasts by comparison",
      x = "Standardized adjusted difference",
      y = NULL
    ) +
    theme_bw(base_size = 14) +
    theme(
      plot.title = element_text(face = "bold", size = 16),
      strip.text = element_text(face = "bold", size = 12),
      axis.text.y = element_text(size = 9),
      panel.grid.major.y = element_line(color = "gray90"),
      panel.grid.minor = element_blank()
    )

  ggsave(file.path(forest_dir, "UCSF_FreeSurfer_primary_forest_significant_adjusted_contrasts_faceted.svg"), p_forest_facet, width = forest_width, height = max(6, min(24, 0.42 * nrow(plot_sig) + 2.5)), units = "in")
  ggsave(file.path(forest_dir, "UCSF_FreeSurfer_primary_forest_significant_adjusted_contrasts_faceted.pdf"), p_forest_facet, width = forest_width, height = max(6, min(24, 0.42 * nrow(plot_sig) + 2.5)), units = "in")
  ggsave(file.path(forest_dir, "UCSF_FreeSurfer_primary_forest_significant_adjusted_contrasts_faceted.png"), p_forest_facet, width = forest_width, height = max(6, min(24, 0.42 * nrow(plot_sig) + 2.5)), units = "in", dpi = 300)
}

if (make_qc_boxplots) {
  message("[9] Drawing QC boxplots")

  top_vars <- primary_contrasts %>%
    filter(!is.na(FDR_all_metrics_all_contrasts)) %>%
    arrange(FDR_all_metrics_all_contrasts) %>%
    distinct(variable, .keep_all = TRUE) %>%
    slice_head(n = boxplot_top_n) %>%
    pull(variable)

  top_vars <- top_vars[top_vars %in% vars]

  if (length(top_vars) > 0) {
    df_long_plot <- df_merged %>%
      select(RID, imaging_group, all_of(top_vars)) %>%
      pivot_longer(cols = all_of(top_vars), names_to = "variable", values_to = "value") %>%
      filter(!is.na(imaging_group), !is.na(value)) %>%
      mutate(
        imaging_group = factor(imaging_group, levels = group_levels),
        variable_label = safe_metric_label(variable, ids),
        variable_label = gsub(" of ", "\nof ", variable_label)
      )

    write.csv(df_long_plot, file.path(table_dir, "UCSF_FreeSurfer_QC_boxplot_top_adjusted_metrics_long.csv"), row.names = FALSE, quote = FALSE)

    p_box <- ggplot(df_long_plot, aes(x = imaging_group, y = value, fill = imaging_group)) +
      geom_boxplot(outlier.shape = NA, alpha = 0.85) +
      geom_jitter(width = jitter_width, alpha = 0.35, size = point_size, color = "black") +
      scale_fill_manual(values = group_colors, na.value = "gray80") +
      labs(
        x = "Projected group",
        y = "Volume",
        title = "UCSF FreeSurfer volumes by current projected group",
        subtitle = "QC plot: unadjusted distributions for top metrics by adjusted-model FDR"
      ) +
      theme_minimal(base_size = base_font_size) +
      theme(
        axis.text.x = element_text(angle = 45, hjust = 1),
        strip.text = element_text(size = 10),
        plot.title = element_text(face = "bold"),
        legend.position = "right"
      ) +
      facet_wrap(~ variable_label, scales = "free_y", ncol = facet_ncol)

    ggsave(file.path(plot_dir, "UCSF_FreeSurfer_QC_boxplots_top_adjusted_metrics.svg"), p_box, width = 16, height = 14, units = "in")
    ggsave(file.path(plot_dir, "UCSF_FreeSurfer_QC_boxplots_top_adjusted_metrics.png"), p_box, width = 16, height = 14, units = "in", dpi = 300)
  }
}

message("[10] Writing simplified supplementary tables")

supp_pairwise <- primary_contrasts %>%
  transmute(
    variable,
    region_label = safe_metric_label(variable, ids),
    contrast,
    estimate,
    lower_95CI = lower.CL,
    upper_95CI = upper.CL,
    standardized_difference,
    standardized_lower_95CI = std_lower.CL,
    standardized_upper_95CI = std_upper.CL,
    SE,
    df,
    t.ratio,
    p.value,
    FDR_BH = FDR_all_metrics_all_contrasts,
    n,
    covariates,
    note
  )
write.csv(supp_pairwise, file.path(table_dir, "Supplementary_Table_UCSF_FreeSurfer_primary_adjusted_pairwise_contrasts.csv"), row.names = FALSE, quote = FALSE)
write.csv(supp_pairwise %>% filter(!is.na(FDR_BH), FDR_BH < 0.05), file.path(table_dir, "Supplementary_Table_UCSF_FreeSurfer_primary_adjusted_pairwise_contrasts_significant_q05.csv"), row.names = FALSE, quote = FALSE)

settings <- data.frame(
  setting = c(
    "projection_source",
    "volume_file",
    "freesurfer_id_file",
    "volume_visit_requested",
    "volume_visit_used",
    "group_definition",
    "n_selected_freesurfer_vars",
    "feature_selection_rule",
    "primary_model",
    "primary_covariates_preferred",
    "extended_scanner_covariates_included",
    "icv_covariate_used",
    "primary_contrasts",
    "primary_FDR_scope",
    "forest_plot_rule",
    "secondary_tests"
  ),
  value = c(
    projection_source,
    volume_file,
    freesurfer_id_file,
    ifelse(is.null(volume_visit), "NULL", volume_visit),
    volume_visit_used,
    "High resilience: predicted_cluster_like; Low resilience: non_resilience",
    as.character(length(vars)),
    ifelse(use_original_st_rule, "ST variables excluding STATUS/FIELD_STRENGTH and suffix SA/TS", "manual_volume_vars"),
    "lm(metric ~ imaging_group + covariates); nested ANOVA for omnibus group effect; emmeans for pairwise contrasts",
    paste(covariates_preferred, collapse = ";"),
    as.character(include_extended_scanner_covariates),
    ifelse(is.na(icv_covariate), "none", icv_covariate),
    paste(names(contrast_definitions), collapse = ";"),
    "BH-FDR across all UCSF FreeSurfer metrics x all prespecified pairwise contrasts",
    paste0("plot only rows with BH-FDR q < ", q_plot_threshold, "; if none, top ", top_n_if_no_q_significant, " rows by q"),
    "Kruskal-Wallis omnibus; Dunn post hoc with Bonferroni; BH-FDR also reported across all metrics/comparisons"
  ),
  stringsAsFactors = FALSE
)
write.csv(settings, file.path(table_dir, "UCSF_FreeSurfer_covariate_adjusted_primary_analysis_settings.csv"), row.names = FALSE, quote = FALSE)

sink(file.path(outdir, "sessionInfo_UCSF_FreeSurfer_covariate_adjusted_primary.txt"))
print(sessionInfo())
sink()

message("Done. Results saved in: ", outdir)
