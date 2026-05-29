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

outdir <- "out/ADNI/projected_rosmap_msbb_22gene_signature/demog_current_projected_groups"
table_dir <- file.path(outdir, "tables")
plot_dir <- file.path(outdir, "plots")
dir.create(table_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(plot_dir, recursive = TRUE, showWarnings = FALSE)

group_levels <- c("cluster1", "cluster2", "non_resilience")

group_colors <- c(
  "cluster1" = "#009E73",
  "cluster2" = "#E69F00",
  "non_resilience" = "gray60"
)

sex_colors <- c(
  "Male" = "#56B4E9",
  "Female" = "#D55E00"
)

apoe_colors <- c(
  "Non-carrier" = "#56B4E9",
  "Carrier" = "#D55E00"
)

plot_base_size <- 20
combined_plot_width <- 14
combined_plot_height <- 11
single_plot_width <- 6
single_plot_height <- 5

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(tidyr)
  library(lubridate)
  library(patchwork)
  library(dunn.test)
})

clean_numeric <- function(x) {
  x <- as.character(x)
  x[x %in% c("", "NA", "NaN", "NULL", "null", "-4")] <- NA
  suppressWarnings(as.numeric(x))
}

p_to_stars <- function(p) {
  if (is.na(p)) return("")
  if (p > 0.05) return("ns")
  if (p > 0.01) return("*")
  if (p > 0.001) return("**")
  if (p > 0.0001) return("***")
  return("****")
}

safe_kruskal <- function(df, value_col, group_col = "group") {
  d <- df %>%
    dplyr::select(dplyr::all_of(c(group_col, value_col))) %>%
    dplyr::filter(!is.na(.data[[group_col]]), !is.na(.data[[value_col]]))

  if (nrow(d) < 3 || length(unique(d[[group_col]])) < 2) {
    return(data.frame(
      variable = value_col,
      test = "Kruskal-Wallis",
      statistic = NA_real_,
      df = NA_real_,
      p.value = NA_real_,
      n = nrow(d),
      note = "insufficient_data",
      stringsAsFactors = FALSE
    ))
  }

  kw <- tryCatch(
    kruskal.test(as.formula(paste(value_col, "~", group_col)), data = d),
    error = function(e) NULL
  )

  data.frame(
    variable = value_col,
    test = "Kruskal-Wallis",
    statistic = if (is.null(kw)) NA_real_ else as.numeric(kw$statistic),
    df = if (is.null(kw)) NA_real_ else as.numeric(kw$parameter),
    p.value = if (is.null(kw)) NA_real_ else kw$p.value,
    n = nrow(d),
    note = if (is.null(kw)) "test_failed" else "",
    stringsAsFactors = FALSE
  )
}

safe_dunn <- function(df, value_col, group_col = "group") {
  d <- df %>%
    dplyr::select(dplyr::all_of(c(group_col, value_col))) %>%
    dplyr::filter(!is.na(.data[[group_col]]), !is.na(.data[[value_col]]))

  if (nrow(d) < 3 || length(unique(d[[group_col]])) < 3) {
    return(data.frame(
      variable = value_col,
      comparison = NA_character_,
      p.value = NA_real_,
      note = "insufficient_data",
      stringsAsFactors = FALSE
    ))
  }

  dn <- tryCatch(
    dunn.test::dunn.test(d[[value_col]], d[[group_col]], method = "bonferroni", kw = FALSE),
    error = function(e) NULL
  )

  if (is.null(dn)) {
    return(data.frame(
      variable = value_col,
      comparison = NA_character_,
      p.value = NA_real_,
      note = "test_failed",
      stringsAsFactors = FALSE
    ))
  }

  data.frame(
    variable = value_col,
    comparison = dn$comparisons,
    p.value = dn$P.adjusted,
    note = "",
    stringsAsFactors = FALSE
  )
}

safe_categorical_test <- function(df, var_col, group_col = "group") {
  d <- df %>%
    dplyr::select(dplyr::all_of(c(group_col, var_col))) %>%
    dplyr::filter(!is.na(.data[[group_col]]), !is.na(.data[[var_col]]))

  tab <- table(d[[group_col]], d[[var_col]])

  if (nrow(tab) < 2 || ncol(tab) < 2) {
    return(list(
      summary = data.frame(
        variable = var_col,
        test = "Chi-squared",
        statistic = NA_real_,
        df = NA_real_,
        p.value = NA_real_,
        n = nrow(d),
        note = "insufficient_data",
        stringsAsFactors = FALSE
      ),
      table = as.data.frame(tab)
    ))
  }

  chi <- suppressWarnings(tryCatch(chisq.test(tab), error = function(e) NULL))
  fisher <- tryCatch(fisher.test(tab), error = function(e) NULL)

  summary <- data.frame(
    variable = c(var_col, var_col),
    test = c("Chi-squared", "Fisher exact"),
    statistic = c(if (is.null(chi)) NA_real_ else as.numeric(chi$statistic), NA_real_),
    df = c(if (is.null(chi)) NA_real_ else as.numeric(chi$parameter), NA_real_),
    p.value = c(if (is.null(chi)) NA_real_ else chi$p.value,
                if (is.null(fisher)) NA_real_ else fisher$p.value),
    n = nrow(d),
    note = c("", ""),
    stringsAsFactors = FALSE
  )

  list(summary = summary, table = as.data.frame(tab))
}

plot_continuous_box <- function(df, value_col, y_label, title = NULL) {
  ggplot(df %>% dplyr::filter(!is.na(.data[[value_col]])),
         aes(x = group, y = .data[[value_col]], fill = group)) +
    geom_boxplot(outlier.shape = NA, alpha = 0.85) +
    geom_jitter(width = 0.18, alpha = 0.45, size = 1.1, color = "black") +
    scale_fill_manual(values = group_colors, na.value = "gray80") +
    labs(x = "Projected group", y = y_label, title = title) +
    theme_minimal(base_size = plot_base_size) +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1),
      plot.title = element_text(face = "bold"),
      legend.position = "none"
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
  if (!("ADpatho" %in% colnames(adni))) {
    stop("Fallback all table does not contain ADpatho.")
  }
  adni <- adni %>% dplyr::filter(ADpatho == "AD")
} else {
  stop("Neither ADpatho nor all ADNI projection table was found.")
}

required_cols <- c("RID", "resilience", "predicted_cluster_like")
missing_cols <- setdiff(required_cols, colnames(adni))
if (length(missing_cols) > 0) {
  stop("ADNI projection table is missing required columns: ", paste(missing_cols, collapse = ", "))
}

df <- adni %>%
  dplyr::mutate(
    RID = as.integer(RID),
    resilience = as.character(resilience),
    predicted_cluster_like = as.character(predicted_cluster_like),
    group = dplyr::case_when(
      resilience == "Low" ~ "non_resilience",
      resilience == "High" & predicted_cluster_like %in% c("cluster1", "cluster2") ~ predicted_cluster_like,
      TRUE ~ NA_character_
    ),
    group = factor(group, levels = group_levels)
  ) %>%
  dplyr::filter(!is.na(group), group %in% group_levels)

message("    Projection source: ", projection_source)
message("    Group distribution:")
print(table(df$group, useNA = "ifany"))

message("[2] Harmonizing demographics")

if (file.exists(ptdemog_file)) {
  ptdemog <- read.csv(ptdemog_file, check.names = FALSE)

  ptdemog_sub <- ptdemog %>%
    dplyr::mutate(RID = as.integer(RID)) %>%
    dplyr::select(dplyr::any_of(c("RID", "PTGENDER", "PTEDUCAT", "PTDOB", "PTDOBYY", "VISDATE", "VISCODE", "VISCODE2"))) %>%
    dplyr::group_by(RID) %>%
    dplyr::arrange(RID) %>%
    dplyr::slice(1) %>%
    dplyr::ungroup()

  df <- df %>% dplyr::left_join(ptdemog_sub, by = "RID", suffix = c("", ".ptdemog"))
} else {
  warning("PTDEMOG file not found: ", ptdemog_file)
}

if (file.exists(apoe_file)) {
  apoe <- read.csv(apoe_file, check.names = FALSE)
  if (all(c("RID", "GENOTYPE") %in% colnames(apoe))) {
    apoe <- apoe %>%
      dplyr::mutate(RID = as.integer(RID)) %>%
      dplyr::select(RID, GENOTYPE) %>%
      dplyr::distinct(RID, .keep_all = TRUE)
    df <- df %>% dplyr::left_join(apoe, by = "RID")
  } else {
    warning("APOE file does not contain RID and GENOTYPE.")
  }
} else {
  warning("APOE file not found: ", apoe_file)
}

if ("age_at_visit" %in% colnames(df)) {
  df$AGE <- clean_numeric(df$age_at_visit)
} else if ("AGE" %in% colnames(df)) {
  df$AGE <- clean_numeric(df$AGE)
} else {
  exam_col <- intersect(c("EXAMDATE", "VISDATE", "EXAMDATE_date"), colnames(df))[1]
  if (!is.na(exam_col) && "PTDOB" %in% colnames(df)) {
    ptdob_date <- suppressWarnings(lubridate::parse_date_time(df$PTDOB, orders = c("m/Y", "m/d/Y", "Y-m-d", "d/m/Y", "b/Y")))
    exam_date <- suppressWarnings(lubridate::parse_date_time(df[[exam_col]], orders = c("m/d/Y", "Y-m-d", "d/m/Y", "Y/m/d", "m/Y", "b/Y")))
    df$AGE <- ifelse(
      !is.na(ptdob_date) & !is.na(exam_date),
      as.numeric(lubridate::interval(ptdob_date, exam_date) / lubridate::years(1)),
      NA_real_
    )
  } else if ("PTDOBYY" %in% colnames(df) && "VISDATE" %in% colnames(df)) {
    birth_year <- clean_numeric(df$PTDOBYY)
    visit_year <- lubridate::year(suppressWarnings(lubridate::parse_date_time(df$VISDATE, orders = c("Y/m/d", "m/d/Y", "Y-m-d"))))
    df$AGE <- visit_year - birth_year
  } else {
    df$AGE <- NA_real_
  }
}

if ("PTEDUCAT" %in% colnames(df)) {
  df$PTEDUCAT_clean <- clean_numeric(df$PTEDUCAT)
} else if ("education" %in% colnames(df)) {
  df$PTEDUCAT_clean <- clean_numeric(df$education)
} else {
  df$PTEDUCAT_clean <- NA_real_
}

if ("sex" %in% colnames(df)) {
  sex_chr <- as.character(df$sex)
  df$SEX <- dplyr::case_when(
    sex_chr %in% c("1", "M", "Male", "male", "MALE") ~ "Male",
    sex_chr %in% c("2", "F", "Female", "female", "FEMALE") ~ "Female",
    TRUE ~ sex_chr
  )
} else if ("PTGENDER" %in% colnames(df)) {
  df$SEX <- dplyr::case_when(
    as.character(df$PTGENDER) == "1" ~ "Male",
    as.character(df$PTGENDER) == "2" ~ "Female",
    TRUE ~ NA_character_
  )
} else {
  df$SEX <- NA_character_
}
df$SEX <- factor(df$SEX, levels = c("Male", "Female"))

if ("resilience_score" %in% colnames(df)) {
  df$resilience_score_clean <- clean_numeric(df$resilience_score)
} else {
  df$resilience_score_clean <- NA_real_
}

if ("GENOTYPE" %in% colnames(df)) {
  df$APOE_e4_carrier <- dplyr::case_when(
    is.na(df$GENOTYPE) | df$GENOTYPE == "" ~ NA_character_,
    grepl("4", df$GENOTYPE) ~ "Carrier",
    TRUE ~ "Non-carrier"
  )
} else if ("APOE_e4_carrier" %in% colnames(df)) {
  df$APOE_e4_carrier <- as.character(df$APOE_e4_carrier)
} else {
  df$APOE_e4_carrier <- NA_character_
}
df$APOE_e4_carrier <- factor(df$APOE_e4_carrier, levels = c("Non-carrier", "Carrier"))

write.csv(
  df,
  file.path(table_dir, "ADNI_current_projected_groups_demographics_merged.csv"),
  row.names = FALSE,
  quote = FALSE
)

message("[3] Creating summary tables")

continuous_summary <- df %>%
  dplyr::select(group, AGE, PTEDUCAT_clean, resilience_score_clean) %>%
  tidyr::pivot_longer(
    cols = c("AGE", "PTEDUCAT_clean", "resilience_score_clean"),
    names_to = "variable",
    values_to = "value"
  ) %>%
  dplyr::filter(!is.na(value)) %>%
  dplyr::group_by(variable, group) %>%
  dplyr::summarise(
    n = dplyr::n(),
    mean = mean(value, na.rm = TRUE),
    sd = sd(value, na.rm = TRUE),
    median = median(value, na.rm = TRUE),
    q1 = quantile(value, 0.25, na.rm = TRUE),
    q3 = quantile(value, 0.75, na.rm = TRUE),
    .groups = "drop"
  )

categorical_summary_sex <- df %>%
  dplyr::filter(!is.na(SEX)) %>%
  dplyr::count(group, SEX, name = "n") %>%
  dplyr::group_by(group) %>%
  dplyr::mutate(prop = n / sum(n)) %>%
  dplyr::ungroup()

categorical_summary_apoe <- df %>%
  dplyr::filter(!is.na(APOE_e4_carrier)) %>%
  dplyr::count(group, APOE_e4_carrier, name = "n") %>%
  dplyr::group_by(group) %>%
  dplyr::mutate(prop = n / sum(n)) %>%
  dplyr::ungroup()

write.csv(continuous_summary, file.path(table_dir, "demographics_continuous_summary_by_group.csv"), row.names = FALSE, quote = FALSE)
write.csv(categorical_summary_sex, file.path(table_dir, "demographics_sex_summary_by_group.csv"), row.names = FALSE, quote = FALSE)
write.csv(categorical_summary_apoe, file.path(table_dir, "demographics_APOE_e4_summary_by_group.csv"), row.names = FALSE, quote = FALSE)

message("[4] Running statistical tests")

stat_summary <- dplyr::bind_rows(
  safe_kruskal(df, "AGE"),
  safe_kruskal(df, "PTEDUCAT_clean"),
  safe_kruskal(df, "resilience_score_clean"),
  safe_categorical_test(df, "SEX")$summary,
  safe_categorical_test(df, "APOE_e4_carrier")$summary
) %>%
  dplyr::mutate(
    FDR = p.adjust(p.value, method = "BH"),
    p_stars = vapply(p.value, p_to_stars, character(1))
  )

dunn_all <- dplyr::bind_rows(
  safe_dunn(df, "AGE"),
  safe_dunn(df, "PTEDUCAT_clean"),
  safe_dunn(df, "resilience_score_clean")
) %>%
  dplyr::mutate(
    p_stars = vapply(p.value, p_to_stars, character(1))
  )

sex_tests <- safe_categorical_test(df, "SEX")
apoe_tests <- safe_categorical_test(df, "APOE_e4_carrier")

write.csv(stat_summary, file.path(table_dir, "demographics_group_stat_summary.csv"), row.names = FALSE, quote = FALSE)
write.csv(dunn_all, file.path(table_dir, "demographics_group_dunn_tests_bonferroni.csv"), row.names = FALSE, quote = FALSE)
write.csv(sex_tests$table, file.path(table_dir, "demographics_sex_contingency_table.csv"), row.names = FALSE, quote = FALSE)
write.csv(apoe_tests$table, file.path(table_dir, "demographics_APOE_e4_contingency_table.csv"), row.names = FALSE, quote = FALSE)

print(stat_summary)

message("[5] Drawing plots")

p_age <- plot_continuous_box(df, "AGE", "Age", "Age")
p_educ <- plot_continuous_box(df, "PTEDUCAT_clean", "Education (years)", "Education")
p_res <- plot_continuous_box(df, "resilience_score_clean", "Resilience score", "Resilience score")

p_sex <- ggplot(df %>% dplyr::filter(!is.na(SEX)),
                aes(x = group, fill = SEX)) +
  geom_bar(position = "fill") +
  scale_fill_manual(values = sex_colors, na.value = "gray80") +
  labs(x = "Projected group", y = "Proportion", fill = "Sex", title = "Sex") +
  theme_minimal(base_size = plot_base_size) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    plot.title = element_text(face = "bold")
  )

p_apoe <- ggplot(df %>% dplyr::filter(!is.na(APOE_e4_carrier)),
                 aes(x = group, fill = APOE_e4_carrier)) +
  geom_bar(position = "fill") +
  scale_fill_manual(values = apoe_colors, na.value = "gray80") +
  labs(x = "Projected group", y = "Proportion", fill = "APOE e4", title = "APOE e4 carrier") +
  theme_minimal(base_size = plot_base_size) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    plot.title = element_text(face = "bold")
  )

combined_plot <- (p_age | p_educ | p_res) / (p_sex | p_apoe | patchwork::plot_spacer())

ggsave(
  file.path(plot_dir, "ADNI_current_projected_groups_demographics_comparison.svg"),
  combined_plot,
  width = combined_plot_width,
  height = combined_plot_height,
  units = "in"
)

ggsave(
  file.path(plot_dir, "ADNI_current_projected_groups_demographics_comparison.png"),
  combined_plot,
  width = combined_plot_width,
  height = combined_plot_height,
  units = "in",
  dpi = 300
)

ggsave(file.path(plot_dir, "ADNI_current_groups_age_boxplot.svg"), p_age, width = single_plot_width, height = single_plot_height, units = "in")
ggsave(file.path(plot_dir, "ADNI_current_groups_education_boxplot.svg"), p_educ, width = single_plot_width, height = single_plot_height, units = "in")
ggsave(file.path(plot_dir, "ADNI_current_groups_resilience_score_boxplot.svg"), p_res, width = single_plot_width, height = single_plot_height, units = "in")
ggsave(file.path(plot_dir, "ADNI_current_groups_sex_barplot.svg"), p_sex, width = single_plot_width, height = single_plot_height, units = "in")
ggsave(file.path(plot_dir, "ADNI_current_groups_APOE_e4_barplot.svg"), p_apoe, width = single_plot_width, height = single_plot_height, units = "in")

settings <- data.frame(
  setting = c(
    "projection_source",
    "ptdemog_file",
    "apoe_file",
    "group_definition",
    "continuous_tests",
    "categorical_tests"
  ),
  value = c(
    projection_source,
    ptdemog_file,
    apoe_file,
    "High resilience: predicted_cluster_like; Low resilience: non_resilience",
    "Kruskal-Wallis; Dunn post hoc with Bonferroni correction",
    "Chi-squared and Fisher exact tests"
  )
)

write.csv(settings, file.path(table_dir, "demographics_current_groups_settings.csv"), row.names = FALSE, quote = FALSE)

sink(file.path(outdir, "sessionInfo_demographics_current_groups.txt"))
print(sessionInfo())
sink()

message("Done. Results saved in: ", outdir)
