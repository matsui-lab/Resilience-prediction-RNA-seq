#!/usr/bin/env Rscript
# Script cleaned for public release. Edit /path/to/... inputs before running.

rm(list = ls())
options(stringsAsFactors = FALSE)

project_dir <- "/path/to/project"
setwd(project_dir)

outdir <- "out/ADNI/projected_rosmap_msbb_22gene_signature"
table_dir <- file.path(outdir, "tables")
heatmap_dir <- file.path(outdir, "heatmaps")
plot_dir <- file.path(outdir, "plots")
dir.create(table_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(heatmap_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(plot_dir, recursive = TRUE, showWarnings = FALSE)

projection_reference_file <- "out/clustering/adni_projection_signature/tables/ROSMAP_MSBB_22gene_projection_reference.rds"

somascan_matrix_file <- "/path/to/ADNI/CruchagaLab_CSF_SOMAscan7k_Protein_matrix_postQC_20230620.csv"
somascan_info_file <- "/path/to/ADNI/ADNI_Cruchaga_lab_CSF_SOMAscan7k_analyte_information_20_06_2023.csv"
dxsum_file <- "/path/to/ADNI/DXSUM_07Jan2025.csv"
biomarker_file <- "/path/to/ADNI/UPENNBIOMK_ROCHE_ELECSYS_07Jan2025.csv"
mmse_file <- "/path/to/ADNI/All_Subjects_MMSE_19Jul2025.csv"
ptdemog_file <- "/path/to/ADNI/PTDEMOG_30Jul2025.csv"

protein_visit <- "bl"
dx_visit <- "bl"
biomarker_visit <- "bl"
mmse_visit <- "sc"
ptdemog_visits_preference <- c("sc", "bl", "init", "f")

include_demographics_in_resilience_model <- FALSE
projection_methods <- c("centroid_correlation", "centroid_negative_euclidean", "marker_delta_weighted")
primary_projection_method <- "centroid_correlation"
projection_sample_set <- "ADpatho_CR"  # all, ADpatho, ADpatho_CR
heatmap_order_method <- "predicted_cluster_then_score"

show_column_names <- FALSE
cluster_rows_heatmap <- FALSE
cluster_cols_heatmap <- FALSE
heatmap_width <- 12
heatmap_height <- 8

cluster_palette <- c(
  "cluster1" = "#009E73", "cluster2" = "#E69F00", "cluster3" = "#7570b3",
  "cluster4" = "#e7298a", "cluster5" = "#66a61e",
  "1" = "#009E73", "2" = "#E69F00", "3" = "#7570b3", "4" = "#e7298a", "5" = "#66a61e"
)

clinical_outcomes_continuous <- c("MMSCORE", "resilience_score", "ABETA42", "TAU", "PTAU", "age_at_visit", "PTEDUCAT")
clinical_outcomes_categorical <- c("DIAGNOSIS", "sex", "ADpatho", "resilience")
base_covariates_for_score_association <- c("age_at_visit", "sex", "PTEDUCAT", "DIAGNOSIS")

suppressPackageStartupMessages({
  library(dplyr)
  library(data.table)
  library(tidyr)
  library(tibble)
  library(ggplot2)
  library(pheatmap)
  library(RColorBrewer)
  library(grid)
  library(lubridate)
})

clean_numeric <- function(x) {
  x <- as.character(x)
  x[x %in% c("", "NA", "NaN", "NULL", "null")] <- NA
  x <- ifelse(x == "90+", "99", x)
  suppressWarnings(as.numeric(x))
}

p_to_stars <- function(p) {
  if (is.na(p)) return("")
  if (p > 0.05) return("")
  if (p > 0.01) return("*")
  if (p > 0.001) return("**")
  if (p > 0.0001) return("***")
  return("****")
}

zscore_cols_features <- function(df, feature_cols) {
  out <- df
  for (g in feature_cols) out[[g]] <- as.numeric(scale(clean_numeric(out[[g]])))
  out
}

safe_cor <- function(x, y) {
  ok <- is.finite(x) & is.finite(y)
  if (sum(ok) < 3 || length(unique(x[ok])) < 2 || length(unique(y[ok])) < 2) return(NA_real_)
  suppressWarnings(cor(x[ok], y[ok], method = "pearson"))
}

save_pheatmap_multi <- function(ph, prefix, width, height) {
  svg(paste0(prefix, ".svg"), width = width, height = height)
  grid::grid.newpage(); grid::grid.draw(ph$gtable); dev.off()
  pdf(paste0(prefix, ".pdf"), width = width, height = height)
  grid::grid.newpage(); grid::grid.draw(ph$gtable); dev.off()
  png(paste0(prefix, ".png"), width = width, height = height, units = "in", res = 300)
  grid::grid.newpage(); grid::grid.draw(ph$gtable); dev.off()
}

prepare_sample_set <- function(df, sample_set) {
  if (sample_set == "all") return(df)
  if (sample_set == "ADpatho") return(df %>% filter(ADpatho == "AD"))
  if (sample_set == "ADpatho_CR") return(df %>% filter(ADpatho == "AD", resilience == "High"))
  stop("Unknown projection_sample_set: ", sample_set)
}

message("[1] Loading projection reference")
if (!file.exists(projection_reference_file)) stop("Projection reference file not found: ", projection_reference_file)
ref <- readRDS(projection_reference_file)
shared_genes <- ref$shared_genes
centroid_df <- ref$centroid_df
marker_table <- ref$marker_table
clusters <- ref$clusters
message("    Clusters: ", paste(clusters, collapse = ", "))

message("[2] Loading ADNI SOMAscan proteins")
protein_raw <- read.csv(somascan_matrix_file, check.names = FALSE)
somascan_info <- read.csv(somascan_info_file, check.names = FALSE)
if (!all(c("RID", "VISCODE2") %in% colnames(protein_raw))) stop("SOMAscan matrix must contain RID and VISCODE2")
if (!all(c("Analytes", "EntrezGeneSymbol") %in% colnames(somascan_info))) stop("SOMAscan info must contain Analytes and EntrezGeneSymbol")

somascan_selected <- somascan_info %>%
  filter(EntrezGeneSymbol %in% unique(shared_genes$Symbol)) %>%
  filter(Analytes %in% colnames(protein_raw))
if (nrow(somascan_selected) == 0) stop("No ADNI analytes matched reference symbols")

protein_bl <- protein_raw %>%
  filter(VISCODE2 == protein_visit) %>%
  select(RID, VISCODE2, all_of(unique(somascan_selected$Analytes))) %>%
  distinct(RID, .keep_all = TRUE)

analyte_to_symbol <- setNames(somascan_selected$EntrezGeneSymbol, somascan_selected$Analytes)
protein_symbol <- protein_bl
for (a in unique(somascan_selected$Analytes)) {
  colnames(protein_symbol)[colnames(protein_symbol) == a] <- analyte_to_symbol[[a]]
}

protein_id_cols <- c("RID", "VISCODE2")
protein_feature_cols_raw <- setdiff(colnames(protein_symbol), protein_id_cols)
protein_long <- protein_symbol %>%
  pivot_longer(cols = all_of(protein_feature_cols_raw), names_to = "Symbol", values_to = "Expression") %>%
  mutate(Expression = clean_numeric(Expression)) %>%
  group_by(RID, VISCODE2, Symbol) %>%
  summarise(Expression = mean(Expression, na.rm = TRUE), .groups = "drop") %>%
  mutate(Expression = ifelse(is.nan(Expression), NA_real_, Expression))
protein_symbol <- protein_long %>% pivot_wider(names_from = Symbol, values_from = Expression)

centroid_symbols <- centroid_df$Symbol
available_symbols <- intersect(centroid_symbols, setdiff(colnames(protein_symbol), protein_id_cols))
if (length(available_symbols) < 2) stop("Fewer than 2 projection symbols are available in ADNI")
protein_symbol <- protein_symbol %>% select(RID, VISCODE2, all_of(available_symbols))
write.csv(somascan_selected, file.path(table_dir, "ADNI_somascan_analytes_used_for_projection.csv"), row.names = FALSE, quote = FALSE)
message("    ADNI samples: ", nrow(protein_symbol))
message("    Projection proteins: ", length(available_symbols))

message("[3] Loading ADNI clinical data")
dxsum <- read.csv(dxsum_file, check.names = FALSE)
biomarker <- read.csv(biomarker_file, check.names = FALSE)
mmse <- read.csv(mmse_file, check.names = FALSE)
ptdemog <- read.csv(ptdemog_file, check.names = FALSE)

dx_bl <- dxsum %>% filter(VISCODE2 == dx_visit) %>% select(RID, DIAGNOSIS) %>% distinct(RID, .keep_all = TRUE)
biomarker_bl <- biomarker %>% filter(VISCODE2 == biomarker_visit) %>%
  select(RID, ABETA42, TAU, PTAU) %>% distinct(RID, .keep_all = TRUE) %>%
  mutate(ABETA42 = clean_numeric(ABETA42), TAU = clean_numeric(TAU), PTAU = clean_numeric(PTAU))
mmse_sc <- mmse %>% filter(VISCODE2 == mmse_visit) %>% select(RID, MMSCORE) %>% distinct(RID, .keep_all = TRUE) %>% mutate(MMSCORE = clean_numeric(MMSCORE))
if (!("PTGENDER" %in% colnames(ptdemog))) stop("PTDEMOG file must contain PTGENDER")

ptdemog2 <- ptdemog %>%
  mutate(
    visit_rank = match(VISCODE2, ptdemog_visits_preference),
    visit_rank = ifelse(is.na(visit_rank), 999, visit_rank),
    PTGENDER = clean_numeric(PTGENDER),
    PTEDUCAT = if ("PTEDUCAT" %in% colnames(.)) clean_numeric(PTEDUCAT) else NA_real_,
    PTDOBYY = if ("PTDOBYY" %in% colnames(.)) clean_numeric(PTDOBYY) else NA_real_
  ) %>%
  arrange(RID, visit_rank) %>% group_by(RID) %>% slice(1) %>% ungroup() %>%
  select(RID, PTID, PTGENDER, PTEDUCAT, PTDOBYY, VISDATE, VISCODE2) %>%
  mutate(
    sex = factor(PTGENDER, levels = c(1, 2), labels = c("Male", "Female")),
    VISDATE_parsed = suppressWarnings(lubridate::ymd(VISDATE)),
    visit_year = lubridate::year(VISDATE_parsed),
    age_at_visit = ifelse(!is.na(visit_year) & !is.na(PTDOBYY), visit_year - PTDOBYY, NA_real_)
  )

clinical <- mmse_sc %>%
  inner_join(biomarker_bl, by = "RID") %>%
  inner_join(dx_bl, by = "RID") %>%
  left_join(ptdemog2 %>% select(RID, PTID, sex, PTGENDER, PTEDUCAT, PTDOBYY, age_at_visit), by = "RID") %>%
  filter(RID %in% protein_symbol$RID) %>%
  mutate(DIAGNOSIS = factor(DIAGNOSIS, levels = c(1, 2, 3), labels = c("CN", "MCI", "Dementia"))) %>%
  mutate(
    cutoff1 = TAU / ABETA42,
    cutoff2 = PTAU / ABETA42,
    cutoff1_binary = ifelse(cutoff1 < 0.33, 0, 1),
    cutoff2_binary = ifelse(cutoff2 < 0.028, 0, 1),
    cutoff3_binary = ifelse(ABETA42 > 880, 0, 1),
    ADpatho = ifelse(cutoff1_binary == 1 & cutoff2_binary == 1 & cutoff3_binary == 1, 1, 0),
    ADpatho = factor(ADpatho, levels = c(0, 1), labels = c("No AD", "AD"))
  )

if (include_demographics_in_resilience_model) {
  res_model_formula <- as.formula("MMSCORE ~ ABETA42 + TAU + PTAU + age_at_visit + sex + PTEDUCAT")
} else {
  res_model_formula <- as.formula("MMSCORE ~ ABETA42 + TAU + PTAU")
}
res_model <- lm(res_model_formula, data = clinical)
clinical$mmse_pred <- predict(res_model, newdata = clinical)
clinical$resilience_score <- clinical$MMSCORE - clinical$mmse_pred
clinical$resilience_binary <- ifelse(clinical$resilience_score < 0, 0, 1)
clinical$resilience <- factor(clinical$resilience_binary, levels = c(0, 1), labels = c("Low", "High"))
sink(file.path(table_dir, "ADNI_resilience_model_summary.txt")); print(summary(res_model)); sink()

message("[4] Preparing ADNI projection matrix")
df_adni <- protein_symbol %>% select(-VISCODE2) %>% inner_join(clinical, by = "RID")
for (s in available_symbols) df_adni[[s]] <- clean_numeric(df_adni[[s]])
df_adni_z <- zscore_cols_features(df_adni, available_symbols)
write.csv(df_adni, file.path(table_dir, "ADNI_projection_protein_clinical_merged_raw.csv"), row.names = FALSE, quote = FALSE)
write.csv(df_adni_z, file.path(table_dir, "ADNI_projection_protein_clinical_merged_z.csv"), row.names = FALSE, quote = FALSE)

message("[5] Projecting ADNI samples")
df_proj_base <- prepare_sample_set(df_adni_z, projection_sample_set)
centroid_symbol_mat <- centroid_df %>% select(Symbol, all_of(clusters)) %>% distinct(Symbol, .keep_all = TRUE) %>% filter(Symbol %in% available_symbols) %>% column_to_rownames("Symbol") %>% as.matrix()
common_symbols <- intersect(available_symbols, rownames(centroid_symbol_mat))
common_symbols <- common_symbols[common_symbols %in% colnames(df_proj_base)]
if (length(common_symbols) < 2) stop("Fewer than 2 common symbols between ADNI and reference")
centroid_symbol_mat <- centroid_symbol_mat[common_symbols, , drop = FALSE]

score_rows <- list()
for (i in seq_len(nrow(df_proj_base))) {
  rid <- df_proj_base$RID[i]
  x <- as.numeric(df_proj_base[i, common_symbols, drop = TRUE])
  names(x) <- common_symbols
  one <- data.frame(RID = rid, stringsAsFactors = FALSE)
  for (cl in colnames(centroid_symbol_mat)) {
    cvec <- as.numeric(centroid_symbol_mat[, cl]); names(cvec) <- rownames(centroid_symbol_mat)
    if ("centroid_correlation" %in% projection_methods) one[[paste0("score_", cl, "_centroid_correlation")]] <- safe_cor(x, cvec)
    if ("centroid_negative_euclidean" %in% projection_methods) {
      ok <- is.finite(x) & is.finite(cvec)
      one[[paste0("score_", cl, "_centroid_negative_euclidean")]] <- ifelse(sum(ok) >= 2, -sqrt(sum((x[ok] - cvec[ok])^2)), NA_real_)
    }
    if ("marker_delta_weighted" %in% projection_methods) {
      mt <- marker_table %>% filter(cluster == cl, Symbol %in% common_symbols)
      w <- mt$delta_z_cluster_vs_others; names(w) <- mt$Symbol
      common_w <- intersect(names(w), names(x))
      one[[paste0("score_", cl, "_marker_delta_weighted")]] <- ifelse(length(common_w) >= 2, sum(x[common_w] * w[common_w], na.rm = TRUE) / sum(abs(w[common_w]), na.rm = TRUE), NA_real_)
    }
  }
  score_rows[[length(score_rows) + 1]] <- one
}
score_df <- bind_rows(score_rows)

primary_score_cols <- paste0("score_", clusters, "_", primary_projection_method)
primary_score_cols <- primary_score_cols[primary_score_cols %in% colnames(score_df)]
if (length(primary_score_cols) == 0) stop("No primary projection score columns generated")
max_idx <- apply(score_df[, primary_score_cols, drop = FALSE], 1, function(z) if (all(is.na(z))) NA_integer_ else which.max(z))
pred_cluster <- rep(NA_character_, nrow(score_df))
pred_cluster[!is.na(max_idx)] <- clusters[max_idx[!is.na(max_idx)]]
score_df$predicted_cluster_like <- factor(pred_cluster, levels = clusters)
score_df$max_primary_score <- apply(score_df[, primary_score_cols, drop = FALSE], 1, function(z) if (all(is.na(z))) NA_real_ else max(z, na.rm = TRUE))
score_df$score_margin <- apply(score_df[, primary_score_cols, drop = FALSE], 1, function(z) { z <- sort(z[is.finite(z)], decreasing = TRUE); if (length(z) < 2) NA_real_ else z[1] - z[2] })

df_projected <- df_proj_base %>% left_join(score_df, by = "RID")
write.csv(score_df, file.path(table_dir, paste0("ADNI_projected_cluster_scores_", projection_sample_set, ".csv")), row.names = FALSE, quote = FALSE)
write.csv(df_projected, file.path(table_dir, paste0("ADNI_projected_cluster_scores_with_clinical_", projection_sample_set, ".csv")), row.names = FALSE, quote = FALSE)
print(table(df_projected$predicted_cluster_like, useNA = "ifany"))

message("[6] Drawing projection heatmap")
expr_heat <- df_projected %>% select(RID, all_of(common_symbols)) %>% column_to_rownames("RID") %>% as.matrix()
if (heatmap_order_method == "predicted_cluster_then_score") {
  ord <- df_projected %>% mutate(predicted_cluster_like = factor(predicted_cluster_like, levels = clusters)) %>% arrange(predicted_cluster_like, desc(max_primary_score)) %>% pull(RID)
} else {
  ord <- df_projected %>% arrange(desc(max_primary_score)) %>% pull(RID)
}
ord <- as.character(ord)
expr_heat <- expr_heat[ord, , drop = FALSE]
heat_mat <- t(expr_heat)

annotation_col <- df_projected %>% mutate(RID_chr = as.character(RID)) %>% filter(RID_chr %in% colnames(heat_mat)) %>% arrange(match(RID_chr, colnames(heat_mat))) %>%
  select(RID_chr, predicted_cluster_like, max_primary_score, score_margin, MMSCORE, resilience_score, ABETA42, TAU, PTAU, DIAGNOSIS, ADpatho, resilience, sex, age_at_visit, PTEDUCAT) %>% column_to_rownames("RID_chr")
ann_colors <- list(
  predicted_cluster_like = cluster_palette[intersect(names(cluster_palette), levels(df_projected$predicted_cluster_like))],
  DIAGNOSIS = c(CN = "#4daf4a", MCI = "#ff7f00", Dementia = "#e41a1c"),
  ADpatho = c("No AD" = "#999999", "AD" = "#377eb8"),
  resilience = c(Low = "#999999", High = "#377eb8"),
  sex = c(Male = "#377eb8", Female = "#e78ac3")
)
ph <- pheatmap(
  heat_mat,
  annotation_col = annotation_col,
  annotation_colors = ann_colors,
  cluster_rows = cluster_rows_heatmap,
  cluster_cols = cluster_cols_heatmap,
  show_colnames = show_column_names,
  main = paste0("ADNI projection of ROSMAP/MSBB 22-gene cluster signatures\nordered by ", primary_projection_method),
  color = colorRampPalette(rev(brewer.pal(n = 11, name = "RdBu")))(100),
  border_color = NA,
  fontsize = 18,
  fontsize_row = 16,
  fontsize_col = 10,
  silent = TRUE
)
save_pheatmap_multi(ph, file.path(heatmap_dir, paste0("ADNI_22protein_projection_heatmap_", projection_sample_set, "_", primary_projection_method)), heatmap_width, heatmap_height)

message("[7] Testing score-clinical associations")
score_cols_all <- grep("^score_.*_(centroid_correlation|centroid_negative_euclidean|marker_delta_weighted)$", colnames(df_projected), value = TRUE)
association_rows <- list()
for (score_col in score_cols_all) {
  for (outcome in c(clinical_outcomes_continuous, clinical_outcomes_categorical)) {
    if (!(outcome %in% colnames(df_projected))) next
    covars <- setdiff(base_covariates_for_score_association, outcome)
    covars <- covars[covars %in% colnames(df_projected)]
    use_cols <- c(outcome, score_col, covars)
    d <- df_projected[, use_cols, drop = FALSE]
    d <- d[complete.cases(d), , drop = FALSE]
    if (nrow(d) < 10 || length(unique(d[[score_col]])) < 2 || length(unique(d[[outcome]])) < 2) {
      association_rows[[length(association_rows) + 1]] <- data.frame(score = score_col, outcome = outcome, outcome_type = ifelse(outcome %in% clinical_outcomes_categorical, "categorical", "continuous"), beta_or_statistic = NA_real_, p_value = NA_real_, n = nrow(d), formula = NA_character_, test = "insufficient_variation", stringsAsFactors = FALSE)
      next
    }
    for (term in names(d)) if (term %in% c(clinical_outcomes_categorical, "sex", "DIAGNOSIS", "ADpatho", "resilience")) d[[term]] <- droplevels(as.factor(d[[term]]))
    if (outcome %in% clinical_outcomes_categorical || is.factor(d[[outcome]])) {
      rhs <- c(outcome, covars); rhs <- rhs[rhs %in% colnames(d)]
      usable_rhs <- c()
      for (term in rhs) {
        z <- d[[term]]
        if (is.factor(z) && nlevels(droplevels(z)) < 2) next
        if (!is.factor(z) && length(unique(z)) < 2) next
        usable_rhs <- c(usable_rhs, term)
      }
      if (!(outcome %in% usable_rhs)) next
      full_form <- as.formula(paste(score_col, "~", paste(usable_rhs, collapse = " + ")))
      reduced_rhs <- setdiff(usable_rhs, outcome)
      reduced_form <- if (length(reduced_rhs) == 0) as.formula(paste(score_col, "~ 1")) else as.formula(paste(score_col, "~", paste(reduced_rhs, collapse = " + ")))
      fit <- tryCatch(lm(full_form, data = d), error = function(e) NULL)
      fit0 <- tryCatch(lm(reduced_form, data = d), error = function(e) NULL)
      an <- if (is.null(fit) || is.null(fit0)) NULL else tryCatch(anova(fit0, fit), error = function(e) NULL)
      p <- if (is.null(an)) NA_real_ else an$`Pr(>F)`[2]
      stat <- if (is.null(an)) NA_real_ else an$F[2]
      association_rows[[length(association_rows) + 1]] <- data.frame(score = score_col, outcome = outcome, outcome_type = "categorical", beta_or_statistic = stat, p_value = p, n = nrow(d), formula = paste(deparse(full_form), collapse = ""), test = "score_lm_nested_anova_outcome", stringsAsFactors = FALSE)
    } else {
      rhs <- c(score_col, covars); rhs <- rhs[rhs %in% colnames(d)]
      usable_rhs <- c()
      for (term in rhs) {
        z <- d[[term]]
        if (is.factor(z) && nlevels(droplevels(z)) < 2) next
        if (!is.factor(z) && length(unique(z)) < 2) next
        usable_rhs <- c(usable_rhs, term)
      }
      if (!(score_col %in% usable_rhs)) next
      form <- as.formula(paste(outcome, "~", paste(usable_rhs, collapse = " + ")))
      fit <- tryCatch(lm(form, data = d), error = function(e) NULL)
      sm <- if (is.null(fit)) NULL else summary(fit)$coefficients
      beta <- if (is.null(sm) || !(score_col %in% rownames(sm))) NA_real_ else sm[score_col, "Estimate"]
      p <- if (is.null(sm) || !(score_col %in% rownames(sm))) NA_real_ else sm[score_col, "Pr(>|t|)"]
      association_rows[[length(association_rows) + 1]] <- data.frame(score = score_col, outcome = outcome, outcome_type = "continuous", beta_or_statistic = beta, p_value = p, n = nrow(d), formula = paste(deparse(form), collapse = ""), test = "outcome_lm_score_beta", stringsAsFactors = FALSE)
    }
  }
}
assoc_df <- bind_rows(association_rows) %>% group_by(outcome) %>% mutate(FDR_by_outcome = p.adjust(p_value, method = "BH")) %>% ungroup() %>% mutate(p_stars = vapply(p_value, p_to_stars, character(1)))
write.csv(assoc_df, file.path(table_dir, paste0("ADNI_projected_signature_score_clinical_associations_", projection_sample_set, ".csv")), row.names = FALSE, quote = FALSE)

message("[8] Drawing score plots by projection method")

score_long <- df_projected %>%
  select(
    RID,
    predicted_cluster_like,
    max_primary_score,
    score_margin,
    DIAGNOSIS,
    ADpatho,
    resilience,
    sex,
    all_of(score_cols_all)
  ) %>%
  pivot_longer(
    cols = all_of(score_cols_all),
    names_to = "score_name",
    values_to = "score_value"
  ) %>%
  mutate(
    score_cluster = sub("^score_([^_]+)_.*$", "\\1", score_name),
    projection_method = sub("^score_[^_]+_", "", score_name),
    projection_method = factor(projection_method, levels = projection_methods),
    score_cluster = factor(score_cluster, levels = clusters),
    score_label = paste0(as.character(score_cluster), " score")
  )

write.csv(
  score_long,
  file.path(table_dir, paste0("ADNI_projected_scores_long_", projection_sample_set, ".csv")),
  row.names = FALSE,
  quote = FALSE
)

projection_method_titles <- c(
  "centroid_correlation" = "Centroid correlation",
  "centroid_negative_euclidean" = "Negative Euclidean distance to centroid",
  "marker_delta_weighted" = "Marker delta-weighted score"
)

projection_method_ylabels <- c(
  "centroid_correlation" = "Projection score: Pearson correlation",
  "centroid_negative_euclidean" = "Projection score: negative Euclidean distance",
  "marker_delta_weighted" = "Projection score: marker delta-weighted mean"
)

score_plot_width <- 8
score_plot_height <- 5
score_plot_ncol <- min(length(clusters), 5)

for (pm in projection_methods) {
  d_pm <- score_long %>%
    filter(projection_method == pm) %>%
    filter(!is.na(predicted_cluster_like), !is.na(score_value))

  if (nrow(d_pm) == 0) {
    warning("No score data available for projection method: ", pm)
    next
  }

  title_pm <- ifelse(
    pm %in% names(projection_method_titles),
    projection_method_titles[[pm]],
    pm
  )

  ylab_pm <- ifelse(
    pm %in% names(projection_method_ylabels),
    projection_method_ylabels[[pm]],
    "Projection score"
  )

  p_pm <- ggplot(
    d_pm,
    aes(x = predicted_cluster_like, y = score_value, fill = predicted_cluster_like)
  ) +
    geom_boxplot(
      outlier.shape = NA,
      alpha = 0.9,
      width = 0.65,
      linewidth = 0.5
    ) +
    facet_wrap(
      ~ score_label,
      scales = "free_y",
      ncol = score_plot_ncol
    ) +
    scale_fill_manual(values = cluster_palette, na.value = "gray80") +
    labs(
      title = paste0("ADNI projected ROSMAP/MSBB cluster-like scores: ", title_pm),
      subtitle = paste0(
        "Sample set: ", projection_sample_set,
        "; descriptive plot only; no circular score-vs-assigned-cluster test shown"
      ),
      x = "Predicted cluster-like group",
      y = ylab_pm,
      fill = "Predicted\ncluster"
    ) +
    theme_bw(base_size = 22) +
    theme(
      legend.position = "right",
      axis.text.x = element_text(angle = 45, hjust = 1, size = 17),
      axis.text.y = element_text(size = 17),
      axis.title = element_text(size = 21),
      strip.text = element_text(size = 17, face = "plain", margin = margin(t = 6, r = 4, b = 6, l = 4)),
      strip.background = element_rect(fill = "gray95", color = "gray70", linewidth = 0.4),
      plot.title = element_text(size = 24, face = "bold"),
      plot.subtitle = element_text(size = 16),
      legend.text = element_text(size = 16),
      legend.title = element_text(size = 17),
      panel.spacing = unit(1.0, "lines"),
      plot.margin = margin(t = 10, r = 12, b = 10, l = 10)
    )

  out_prefix_pm <- file.path(
    plot_dir,
    paste0(
      "ADNI_projected_scores_by_predicted_cluster_",
      projection_sample_set,
      "_",
      pm
    )
  )

  ggsave(
    paste0(out_prefix_pm, ".svg"),
    p_pm,
    width = score_plot_width,
    height = score_plot_height,
    units = "in"
  )
  ggsave(
    paste0(out_prefix_pm, ".pdf"),
    p_pm,
    width = score_plot_width,
    height = score_plot_height,
    units = "in"
  )
  ggsave(
    paste0(out_prefix_pm, ".png"),
    p_pm,
    width = score_plot_width,
    height = score_plot_height,
    units = "in",
    dpi = 300
  )
}

p_score_overview <- ggplot(
  score_long %>% filter(!is.na(predicted_cluster_like), !is.na(score_value)),
  aes(x = predicted_cluster_like, y = score_value, fill = predicted_cluster_like)
) +
  geom_boxplot(
    outlier.shape = NA,
    alpha = 0.9,
    width = 0.65,
    linewidth = 0.35
  ) +
  facet_grid(
    projection_method ~ score_label,
    scales = "free_y"
  ) +
  scale_fill_manual(values = cluster_palette, na.value = "gray80") +
  labs(
    title = "ADNI projected ROSMAP/MSBB cluster-like scores",
    subtitle = paste0("Sample set: ", projection_sample_set, "; overview across projection methods"),
    x = "Predicted cluster-like group",
    y = "Projection score",
    fill = "Predicted\ncluster"
  ) +
  theme_bw(base_size = 18) +
  theme(
    legend.position = "right",
    axis.text.x = element_text(angle = 45, hjust = 1, size = 13),
    axis.text.y = element_text(size = 13),
    axis.title = element_text(size = 17),
    strip.text.x = element_text(size = 13, face = "plain", margin = margin(t = 5, r = 3, b = 5, l = 3)),
    strip.text.y = element_text(size = 13, face = "plain", angle = 0, margin = margin(t = 3, r = 5, b = 3, l = 5)),
    strip.background = element_rect(fill = "gray95", color = "gray70", linewidth = 0.4),
    plot.title = element_text(size = 21, face = "bold"),
    plot.subtitle = element_text(size = 14),
    legend.text = element_text(size = 13),
    legend.title = element_text(size = 14),
    panel.spacing = unit(0.8, "lines"),
    plot.margin = margin(t = 10, r = 12, b = 10, l = 10)
  )

ggsave(
  file.path(plot_dir, paste0("ADNI_projected_scores_by_predicted_cluster_", projection_sample_set, "_overview_all_methods.svg")),
  p_score_overview,
  width = 14,
  height = 9,
  units = "in"
)
ggsave(
  file.path(plot_dir, paste0("ADNI_projected_scores_by_predicted_cluster_", projection_sample_set, "_overview_all_methods.pdf")),
  p_score_overview,
  width = 14,
  height = 9,
  units = "in"
)
ggsave(
  file.path(plot_dir, paste0("ADNI_projected_scores_by_predicted_cluster_", projection_sample_set, "_overview_all_methods.png")),
  p_score_overview,
  width = 14,
  height = 9,
  units = "in",
  dpi = 300
)

settings <- data.frame(
  setting = c("projection_reference_file", "primary_projection_method", "projection_methods", "projection_sample_set", "available_symbols", "common_symbols"),
  value = c(projection_reference_file, primary_projection_method, paste(projection_methods, collapse = ";"), projection_sample_set, paste(available_symbols, collapse = ";"), paste(common_symbols, collapse = ";"))
)
write.csv(settings, file.path(table_dir, "ADNI_projection_settings.csv"), row.names = FALSE, quote = FALSE)
sink(file.path(outdir, "sessionInfo_project_adni_signature.txt")); print(sessionInfo()); sink()
message("Done. ADNI projection results saved in: ", outdir)
