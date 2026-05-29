# Cognitive Resilience Molecular Subtype Analysis

This repository contains the analysis scripts used for the manuscript:

**Machine Learning-Driven Identification of Molecular Subtypes of Cognitive Resilience in Alzheimer's Disease**

The scripts reproduce the main and supplementary analyses based on ROSMAP and MSBB RNA-seq data and ADNI CSF proteomics, MRI-derived measures, and longitudinal MMSE data.

## Repository contents

```text
scripts/
  01_preprocessing/
    preprocessing_rosmap_resilience_scores.R
    preprocessing_msbb_resilience_scores.R
    expression_integration_combat.R
  02_ml/
    ML_nested_cv_external_validation.py
    refit_final_models_eval_shap.py
  03_model_features/
    feature_clustering.R
    feature_correlation_exploratory.R
    feature_heatmap_redraw.R
  04_adni_projection/
    build_rosmap_msbb_22gene_signature.R
    project_adni_to_22gene_signature.R
    redraw_adni_projection_heatmap.R
    plot_adni_all22_proteins.R
    plot_adni_selected_proteins.R
  05_adni_clinical_proteomics/
    adni_demographics_comparison.R
    adni_22proteins_lm_clinical_heatmap.R
    adni_somascan_dep_volcano.R
  06_adni_imaging/
    adni_ucd_wmh_tcv_adjusted_primary.R
    ucd_primary_forest_plot.R
    adni_ucsf_freesurfer_covariate_adjusted.R
  07_adni_longitudinal/
    adni_mmse_survival_lmm.R
    adni_mmse_cox_no_baseline_pairwise.R
  08_rega/
    prepare_rega_inputs_current_ml.py
    ui.R
    server.R
```

## Analysis overview

1. Compute residual-based cognitive resilience scores in ROSMAP and MSBB.
2. Merge ROSMAP and MSBB RNA-seq matrices and correct batch effects using ComBat.
3. Train cognitive-resilience classifiers in ROSMAP using leakage-safe nested cross-validation.
4. Evaluate fixed ROSMAP-trained models once in the independent MSBB cohort.
5. Refit final models and calculate model-derived gene contribution scores.
6. Cluster CR individuals using model-derived positive-contribution genes.
7. Build a 22-gene/protein ROSMAP/MSBB subtype signature and project it onto ADNI CSF SOMAscan data.
8. Evaluate projected ADNI groups using demographics, CSF proteomics, MRI-derived measures, and longitudinal MMSE analyses.
9. Prepare input files and Shiny code for REGA.

## Required input data

The scripts expect processed or downloaded files from:

- ROSMAP RNA-seq harmonization data
- MSBB RNA-seq harmonization data
- ADNI clinical, CSF biomarker, CSF SOMAscan, APOE, MMSE, UCD WMH, and UCSF FreeSurfer tables
- Gene annotation files
- Optional MSigDB GMT files, if enrichment analyses are run separately

All absolute paths have been masked as `/path/to/...`. Edit the path variables at the top of each script before running.

## Suggested execution order

```bash
Rscript scripts/01_preprocessing/preprocessing_rosmap_resilience_scores.R
Rscript scripts/01_preprocessing/preprocessing_msbb_resilience_scores.R
Rscript scripts/01_preprocessing/expression_integration_combat.R

python scripts/02_ml/ML_nested_cv_external_validation.py
python scripts/02_ml/refit_final_models_eval_shap.py --help

Rscript scripts/03_model_features/feature_clustering.R
Rscript scripts/03_model_features/feature_correlation_exploratory.R
Rscript scripts/03_model_features/feature_heatmap_redraw.R

Rscript scripts/04_adni_projection/build_rosmap_msbb_22gene_signature.R
Rscript scripts/04_adni_projection/project_adni_to_22gene_signature.R
Rscript scripts/04_adni_projection/redraw_adni_projection_heatmap.R
Rscript scripts/04_adni_projection/plot_adni_all22_proteins.R
Rscript scripts/04_adni_projection/plot_adni_selected_proteins.R

Rscript scripts/05_adni_clinical_proteomics/adni_demographics_comparison.R
Rscript scripts/05_adni_clinical_proteomics/adni_22proteins_lm_clinical_heatmap.R
Rscript scripts/05_adni_clinical_proteomics/adni_somascan_dep_volcano.R

Rscript scripts/06_adni_imaging/adni_ucd_wmh_tcv_adjusted_primary.R
Rscript scripts/06_adni_imaging/ucd_primary_forest_plot.R
Rscript scripts/06_adni_imaging/adni_ucsf_freesurfer_covariate_adjusted.R

Rscript scripts/07_adni_longitudinal/adni_mmse_survival_lmm.R
Rscript scripts/07_adni_longitudinal/adni_mmse_cox_no_baseline_pairwise.R

python scripts/08_rega/prepare_rega_inputs_current_ml.py --help
```

To run REGA locally after preparing the input files, place `ui.R`, `server.R`, and the REGA input CSV files in the same app directory and run:

```r
shiny::runApp("scripts/08_rega")
```

## Main software dependencies

### R

Core packages used across scripts include:

`data.table`, `dplyr`, `tidyr`, `ggplot2`, `pheatmap`, `RColorBrewer`, `sva`, `emmeans`, `dunn.test`, `survival`, `survminer`, `lme4`, `lmerTest`, `broom`, `broom.mixed`, `patchwork`, `ggpubr`, `plotly`, `DT`, `shiny`, `lubridate`, `tibble`, `readr`, `stringr`, `forcats`.

### Python

Core packages used across scripts include:

`numpy`, `pandas`, `scikit-learn`, `matplotlib`, `joblib`, `xgboost`, `shap`.

## Notes on public release

- Absolute local paths were replaced with `/path/to/...` placeholders.
- Japanese comments and internal scratch comments were removed.
- Scripts that were superseded by the final manuscript analyses or represented unused sensitivity/archived analyses were not included in this cleaned release bundle.
- Data files are not included. Access to ROSMAP, MSBB, and ADNI source data must follow each cohort's data-use requirements.

