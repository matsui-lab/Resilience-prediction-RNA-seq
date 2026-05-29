#!/usr/bin/env python3
# Script cleaned for public release. Edit /path/to/... inputs before running.

import os

os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import ast
import json
import platform
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve

warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    xgb = None
    HAS_XGB = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    shap = None
    HAS_SHAP = False

RANDOM_STATE = 42
N_BOOTSTRAP = 1000
N_JOBS_MODEL = 1

def make_model(model_name, params=None):
    if params is None:
        params = {}

    if model_name == "ElasticNet":
        clf = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            max_iter=10000,
            random_state=RANDOM_STATE,
            C=params.get("C", 1.0),
            l1_ratio=params.get("l1_ratio", 0.5),
        )
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", clf),
        ])

    if model_name == "SVM":
        clf = SVC(
            probability=True,
            random_state=RANDOM_STATE,
            C=params.get("C", 1.0),
            gamma=params.get("gamma", "scale"),
            kernel=params.get("kernel", "rbf"),
        )
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", clf),
        ])

    if model_name == "RandomForest":
        clf = RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_estimators=params.get("n_estimators", 200),
            max_depth=params.get("max_depth", None),
            min_samples_leaf=params.get("min_samples_leaf", 1),
            n_jobs=N_JOBS_MODEL,
        )
        return Pipeline([
            ("clf", clf),
        ])

    if model_name == "XGBoost":
        if not HAS_XGB:
            raise ImportError("xgboost is not installed, but XGBoost was requested.")
        clf = xgb.XGBClassifier(
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", 3),
            learning_rate=params.get("learning_rate", 0.1),
            subsample=params.get("subsample", 1.0),
            colsample_bytree=params.get("colsample_bytree", 1.0),
            n_jobs=N_JOBS_MODEL,
        )
        return Pipeline([
            ("clf", clf),
        ])

    raise ValueError(f"Unknown model: {model_name}")

def load_data(meta_path, expr_path):
    print("Loading data...")
    meta = pd.read_csv(meta_path)
    expr = pd.read_csv(expr_path, index_col=0)

    expr = expr[meta["specimenID"].values]
    meta = meta.set_index("specimenID")

    rosmap_idx = meta[meta["cohort"] == "rosmap"].index
    msbb_idx = meta[meta["cohort"] == "MSBB"].index

    X_rosmap = expr[rosmap_idx].T
    y_rosmap = meta.loc[rosmap_idx, "resilience"].astype(int)

    X_msbb = expr[msbb_idx].T
    y_msbb = meta.loc[msbb_idx, "resilience"].astype(int)

    print(f"ROSMAP: {X_rosmap.shape}")
    print(f"MSBB:   {X_msbb.shape}")

    return meta, X_rosmap, y_rosmap, X_msbb, y_msbb, msbb_idx

def read_previous_best_config(previous_out_dir, model_name):
    summary_path = os.path.join(previous_out_dir, f"{model_name}_final_ROSMAP_inner_summary.csv")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Missing previous summary file: {summary_path}")

    summary = pd.read_csv(summary_path)
    if summary.empty:
        raise ValueError(f"Previous summary file is empty: {summary_path}")

    required = {"params", "n_features"}
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"{summary_path} is missing required columns: {sorted(missing)}")

    best_row = summary.iloc[0]
    params = ast.literal_eval(best_row["params"])
    n_features = int(best_row["n_features"])

    return params, n_features, summary_path, best_row.to_dict()

def read_previous_final_features(previous_out_dir, model_name):
    features_path = os.path.join(previous_out_dir, f"{model_name}_final_features_ROSMAP_trained.csv")
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Missing previous final-features file: {features_path}")

    df = pd.read_csv(features_path)
    if df.empty:
        raise ValueError(f"Previous final-features file is empty: {features_path}")

    if "gene" in df.columns:
        features = df["gene"].astype(str).tolist()
    elif "feature" in df.columns:
        features = df["feature"].astype(str).tolist()
    else:
        features = df.iloc[:, 0].astype(str).tolist()

    return features, features_path

def ensure_features_present(X, features, model_name, dataset_name):
    missing = [f for f in features if f not in X.columns]
    if missing:
        example = ", ".join(missing[:10])
        raise ValueError(
            f"{model_name}: {len(missing)} selected features are missing from {dataset_name}. "
            f"Examples: {example}"
        )

def calculate_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)

    return {
        "AUROC": roc_auc_score(y_true, y_prob),
        "AUPRC": average_precision_score(y_true, y_prob),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred),
        "Brier": brier_score_loss(y_true, y_prob),
    }

def bootstrap_ci(y_true, y_prob, n_bootstrap=N_BOOTSTRAP):
    rng = np.random.default_rng(RANDOM_STATE)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    boot_records = []
    for _ in range(n_bootstrap):
        idx = rng.choice(np.arange(len(y_true)), size=len(y_true), replace=True)
        y_b = y_true[idx]
        p_b = y_prob[idx]

        if len(np.unique(y_b)) < 2:
            continue

        boot_records.append(calculate_metrics(y_b, p_b))

    boot_df = pd.DataFrame(boot_records)

    ci_records = []
    for metric in boot_df.columns:
        ci_records.append({
            "metric": metric,
            "lower_95CI": np.percentile(boot_df[metric], 2.5),
            "upper_95CI": np.percentile(boot_df[metric], 97.5),
        })

    return pd.DataFrame(ci_records), boot_df

def save_calibration_plot(y_true, y_prob, title, out_path):
    prob_true, prob_pred = calibration_curve(
        y_true,
        y_prob,
        n_bins=10,
        strategy="quantile",
    )

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    plt.plot(prob_pred, prob_true, marker="o", label="Model")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed fraction of positives")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def evaluate_and_save(y_true, y_prob, model_name, dataset_name, out_dir):
    metrics = calculate_metrics(y_true, y_prob)
    ci_df, boot_df = bootstrap_ci(y_true, y_prob)

    metrics_df = pd.DataFrame({
        "metric": list(metrics.keys()),
        "value": list(metrics.values()),
    })
    metrics_df = metrics_df.merge(ci_df, on="metric", how="left")
    metrics_df["model"] = model_name
    metrics_df["dataset"] = dataset_name

    prefix = f"{model_name}_{dataset_name}".replace(" ", "_")

    metrics_df.to_csv(
        os.path.join(out_dir, f"{prefix}_metrics_bootstrap_ci.csv"),
        index=False,
    )
    boot_df.to_csv(
        os.path.join(out_dir, f"{prefix}_bootstrap_distribution.csv"),
        index=False,
    )

    save_calibration_plot(
        y_true,
        y_prob,
        title=f"{model_name} calibration: {dataset_name}",
        out_path=os.path.join(out_dir, f"{prefix}_calibration_curve.svg"),
    )

    y_pred = (np.asarray(y_prob) >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=["true_0", "true_1"],
        columns=["pred_0", "pred_1"],
    )
    cm_df.to_csv(os.path.join(out_dir, f"{prefix}_confusion_matrix.csv"))

    return metrics_df

def compare_with_previous_predictions(previous_out_dir, out_dir, model_name, dataset_name, new_pred_df):
    previous_path = os.path.join(previous_out_dir, f"{model_name}_{dataset_name}_predictions.csv")
    if not os.path.exists(previous_path):
        print(f"Previous prediction file not found, skipping comparison: {previous_path}")
        return None

    prev = pd.read_csv(previous_path)
    needed = {"specimenID", "y_prob"}
    if not needed.issubset(prev.columns):
        print(f"Previous prediction file lacks required columns, skipping comparison: {previous_path}")
        return None

    merged = prev[["specimenID", "y_prob"]].rename(columns={"y_prob": "y_prob_previous"}).merge(
        new_pred_df[["specimenID", "y_prob"]].rename(columns={"y_prob": "y_prob_refit"}),
        on="specimenID",
        how="inner",
    )

    if merged.empty:
        print(f"No overlapping specimenID values for comparison: {model_name}, {dataset_name}")
        return None

    merged["abs_diff"] = np.abs(merged["y_prob_refit"] - merged["y_prob_previous"])
    corr = merged[["y_prob_previous", "y_prob_refit"]].corr(method="pearson").iloc[0, 1]

    summary = pd.DataFrame([{
        "model": model_name,
        "dataset": dataset_name,
        "n_overlap": len(merged),
        "max_abs_diff_y_prob": merged["abs_diff"].max(),
        "mean_abs_diff_y_prob": merged["abs_diff"].mean(),
        "median_abs_diff_y_prob": merged["abs_diff"].median(),
        "pearson_corr_y_prob": corr,
    }])

    prefix = f"{model_name}_{dataset_name}"
    merged.to_csv(os.path.join(out_dir, f"{prefix}_prediction_comparison_with_previous.csv"), index=False)
    summary.to_csv(os.path.join(out_dir, f"{prefix}_prediction_comparison_summary.csv"), index=False)

    return summary

def compare_metrics(previous_out_dir, out_dir, model_name, dataset_name, new_metrics_df):
    previous_path = os.path.join(previous_out_dir, f"{model_name}_{dataset_name}_metrics_bootstrap_ci.csv")
    if not os.path.exists(previous_path):
        return None

    prev = pd.read_csv(previous_path)
    if "metric" not in prev.columns or "value" not in prev.columns:
        return None

    merged = prev[["metric", "value"]].rename(columns={"value": "value_previous"}).merge(
        new_metrics_df[["metric", "value"]].rename(columns={"value": "value_refit"}),
        on="metric",
        how="inner",
    )
    merged["delta_refit_minus_previous"] = merged["value_refit"] - merged["value_previous"]

    prefix = f"{model_name}_{dataset_name}"
    merged.to_csv(os.path.join(out_dir, f"{prefix}_metric_comparison_with_previous.csv"), index=False)
    return merged

def get_estimator_and_model_matrix(pipeline, X_subset):
    if "scaler" in pipeline.named_steps:
        X_model = pipeline.named_steps["scaler"].transform(X_subset)
    else:
        X_model = X_subset.values
    clf = pipeline.named_steps["clf"]
    return clf, X_model

def extract_class1_shap_values(shap_values):
    """Return class-1 SHAP matrix with shape n_samples x n_features."""
    if isinstance(shap_values, list):
        if len(shap_values) == 1:
            return np.asarray(shap_values[0])
        return np.asarray(shap_values[1])

    arr = np.asarray(shap_values)

    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        if arr.shape[-1] == 2:
            return arr[:, :, 1]
        if arr.shape[0] == 2:
            return arr[1, :, :]

    raise ValueError(f"Unsupported SHAP array shape: {arr.shape}")

def save_elasticnet_coefficients(pipeline, features, out_dir, model_name):
    clf = pipeline.named_steps["clf"]
    coef = clf.coef_[0]

    df = pd.DataFrame({
        "feature": features,
        "importance": coef,
        "abs_importance": np.abs(coef),
    }).sort_values("abs_importance", ascending=False)

    df[["feature", "importance"]].to_csv(
        os.path.join(out_dir, f"feature_importance_{model_name}.csv"),
        index=False,
    )
    df.to_csv(
        os.path.join(out_dir, f"feature_importance_{model_name}_coefficients.csv"),
        index=False,
    )

    return df

def calculate_and_save_shap(
    pipeline,
    model_name,
    features,
    X_explain,
    specimen_ids,
    shap_out_dir,
    shap_background_k=50,
    shap_nsamples=200,
    shap_max_samples=None,
    save_long=True,
):
    os.makedirs(shap_out_dir, exist_ok=True)

    if model_name == "ElasticNet":
        print("  Saving ElasticNet coefficients as feature importance.")
        return save_elasticnet_coefficients(pipeline, features, shap_out_dir, model_name)

    if not HAS_SHAP:
        raise ImportError("shap is not installed. Install shap or run without --calculate_shap.")

    X_subset = X_explain[features]
    specimen_ids = pd.Index(specimen_ids)

    if shap_max_samples is not None and shap_max_samples > 0 and len(X_subset) > shap_max_samples:
        X_subset = X_subset.iloc[:shap_max_samples].copy()
        specimen_ids = specimen_ids[:shap_max_samples]

    clf, X_model = get_estimator_and_model_matrix(pipeline, X_subset)
    X_model_df = pd.DataFrame(X_model, columns=features, index=specimen_ids)

    print(f"  Calculating SHAP for {model_name}: n_samples={X_model_df.shape[0]}, n_features={X_model_df.shape[1]}")

    if model_name in ["RandomForest", "XGBoost"]:
        explainer = shap.TreeExplainer(clf)
        raw_shap_values = explainer.shap_values(X_model_df)

    elif model_name == "SVM":
        background_k = min(shap_background_k, X_model.shape[0])
        background = shap.kmeans(X_model, background_k)
        explainer = shap.KernelExplainer(clf.predict_proba, background)
        raw_shap_values = explainer.shap_values(X_model, nsamples=shap_nsamples)

    else:
        raise ValueError(f"SHAP calculation is not implemented for {model_name}")

    shap_class1 = extract_class1_shap_values(raw_shap_values)
    shap_class1 = np.asarray(shap_class1)

    if shap_class1.shape != X_model_df.shape:
        raise ValueError(
            f"Unexpected SHAP shape for {model_name}: {shap_class1.shape}; expected {X_model_df.shape}"
        )

    mean_abs = np.abs(shap_class1).mean(axis=0)
    mean_signed = shap_class1.mean(axis=0)

    importance_df = pd.DataFrame({
        "feature": features,
        "importance": mean_abs,
    }).sort_values("importance", ascending=False)

    importance_df.to_csv(
        os.path.join(shap_out_dir, f"feature_importance_{model_name}.csv"),
        index=False,
    )

    detailed_df = pd.DataFrame({
        "feature": features,
        "importance_mean_abs_shap": mean_abs,
        "mean_shap": mean_signed,
    }).sort_values("importance_mean_abs_shap", ascending=False)
    detailed_df.to_csv(
        os.path.join(shap_out_dir, f"feature_importance_{model_name}_mean_abs_SHAP.csv"),
        index=False,
    )

    wide_df = pd.DataFrame(shap_class1, columns=features, index=specimen_ids)
    wide_df.index.name = "specimenID"
    wide_df.to_csv(os.path.join(shap_out_dir, f"{model_name}_MSBB_SHAP_values_wide.csv"))

    if save_long:
        long_df = wide_df.reset_index().melt(
            id_vars="specimenID",
            var_name="feature",
            value_name="shap_value",
        )
        long_df["model"] = model_name
        long_df.to_csv(os.path.join(shap_out_dir, f"{model_name}_MSBB_SHAP_values_long.csv"), index=False)

    return detailed_df

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Refit final ROSMAP-trained models using previous final configs/features, "
            "save models, recompute MSBB evaluation, and optionally calculate SHAP."
        )
    )

    parser.add_argument(
        "--base_dir",
        type=str,
        default="/path/to/project",
        help="Project base directory. Defaults to the path shown in the previous result listing.",
    )
    parser.add_argument(
        "--previous_out_dir",
        type=str,
        default=None,
        help="Directory containing previous ml_nested_cv outputs. Default: <base_dir>/out/ml_nested_cv",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save refit models/evaluation/SHAP outputs. Default: <base_dir>/out/ml_final_refit_saved_models_shap",
    )
    parser.add_argument(
        "--meta_path",
        type=str,
        default=None,
        help="Path to metadata CSV. Default: <base_dir>/out/combined/meta_merged_exclude89.csv",
    )
    parser.add_argument(
        "--expr_path",
        type=str,
        default=None,
        help="Path to expression CSV. Default: <base_dir>/out/combined/exp_merged_combat_exclude89.csv",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Models to run. Default: ElasticNet SVM RandomForest plus XGBoost if xgboost is installed.",
    )
    parser.add_argument(
        "--calculate_shap",
        action="store_true",
        help="Calculate SHAP outputs for MSBB using the refit final models.",
    )
    parser.add_argument(
        "--shap_background_k",
        type=int,
        default=50,
        help="Number of k-means background centroids for SVM KernelExplainer.",
    )
    parser.add_argument(
        "--shap_nsamples",
        type=int,
        default=200,
        help="nsamples argument for SVM KernelExplainer.",
    )
    parser.add_argument(
        "--shap_max_samples",
        type=int,
        default=None,
        help="Optional maximum number of MSBB samples used for SHAP. Default: all MSBB samples.",
    )
    parser.add_argument(
        "--no_save_shap_long",
        action="store_true",
        help="Do not save long-format sample-by-feature SHAP table.",
    )
    parser.add_argument(
        "--skip_top_bottom",
        action="store_true",
        help="Skip MSBB top/bottom quartile secondary evaluation.",
    )

    return parser.parse_args()

def main():
    args = parse_args()

    np.random.seed(RANDOM_STATE)

    base_dir = args.base_dir
    previous_out_dir = args.previous_out_dir or os.path.join(base_dir, "out", "ml_nested_cv")
    output_dir = args.output_dir or os.path.join(base_dir, "out", "ml_final_refit_saved_models_shap")
    meta_path = "/path/to/project/out/combined/meta_merged_exclude89.csv"
    expr_path = "/path/to/project/out/combined/exp_merged_combat_exclude89.csv"

    os.makedirs(output_dir, exist_ok=True)
    model_dir = os.path.join(output_dir, "saved_models")
    shap_out_dir = os.path.join(output_dir, "shap_values")
    os.makedirs(model_dir, exist_ok=True)

    if args.models is None:
        model_names = ["ElasticNet", "SVM", "RandomForest"]
        if HAS_XGB:
            model_names.append("XGBoost")
    else:
        model_names = args.models

    print("============================================================")
    print("Refit final models from previous outputs")
    print("============================================================")
    print(f"base_dir:         {base_dir}")
    print(f"previous_out_dir: {previous_out_dir}")
    print(f"output_dir:       {output_dir}")
    print(f"meta_path:        {meta_path}")
    print(f"expr_path:        {expr_path}")
    print(f"models:           {model_names}")
    print(f"calculate_shap:   {args.calculate_shap}")

    meta, X_rosmap, y_rosmap, X_msbb, y_msbb, msbb_idx = load_data(meta_path, expr_path)

    all_metrics = []
    all_external_predictions = []
    all_final_features = []
    all_prediction_comparisons = []
    all_metric_comparisons = []
    saved_model_records = []

    for model_name in model_names:
        print("\n============================================================")
        print(f"Processing model: {model_name}")
        print("============================================================")

        final_params, final_n_features, summary_path, best_config_row = read_previous_best_config(
            previous_out_dir, model_name
        )
        final_features, features_path = read_previous_final_features(previous_out_dir, model_name)

        ensure_features_present(X_rosmap, final_features, model_name, "ROSMAP")
        ensure_features_present(X_msbb, final_features, model_name, "MSBB")

        print(f"  Loaded params from:   {summary_path}")
        print(f"  Loaded features from: {features_path}")
        print(f"  Previous n_features setting: {final_n_features}")
        print(f"  Actual loaded final features: {len(final_features)}")
        print(f"  Params: {final_params}")

        print("  Fitting final model on full ROSMAP using previous final features...")
        pipeline = make_model(model_name, final_params)
        pipeline.fit(X_rosmap[final_features], y_rosmap)

        model_path = os.path.join(model_dir, f"{model_name}_ROSMAP_final_refit_model.joblib")
        bundle = {
            "pipeline": pipeline,
            "selected_features": final_features,
            "final_params": final_params,
            "final_n_features_from_previous_summary": final_n_features,
            "actual_n_features": len(final_features),
            "model_name": model_name,
            "training_cohort": "ROSMAP",
            "external_cohort": "MSBB",
            "previous_summary_path": summary_path,
            "previous_features_path": features_path,
            "best_config_row_from_previous_summary": best_config_row,
            "random_state": RANDOM_STATE,
        }
        joblib.dump(bundle, model_path)
        print(f"  Saved model: {model_path}")

        metadata = {
            "model_name": model_name,
            "model_path": model_path,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "random_state": RANDOM_STATE,
            "thread_env": {
                "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
                "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
                "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
                "VECLIB_MAXIMUM_THREADS": os.environ.get("VECLIB_MAXIMUM_THREADS"),
                "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
            },
            "final_params": final_params,
            "final_n_features_from_previous_summary": final_n_features,
            "actual_n_features": len(final_features),
            "previous_summary_path": summary_path,
            "previous_features_path": features_path,
        }
        metadata_path = os.path.join(model_dir, f"{model_name}_ROSMAP_final_refit_metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        pd.DataFrame({"gene": final_features}).to_csv(
            os.path.join(output_dir, f"{model_name}_final_features_ROSMAP_refit.csv"),
            index=False,
        )

        all_final_features.append(pd.DataFrame({
            "model": model_name,
            "gene": final_features,
        }))

        saved_model_records.append({
            "model": model_name,
            "model_path": model_path,
            "metadata_path": metadata_path,
            "n_features": len(final_features),
        })

        print("  Evaluating MSBB full external cohort...")
        msbb_prob = pipeline.predict_proba(X_msbb[final_features])[:, 1]
        msbb_metrics = evaluate_and_save(
            y_true=y_msbb.values,
            y_prob=msbb_prob,
            model_name=model_name,
            dataset_name="MSBB_full_external",
            out_dir=output_dir,
        )
        all_metrics.append(msbb_metrics)

        msbb_pred_df = pd.DataFrame({
            "specimenID": y_msbb.index,
            "y_true": y_msbb.values,
            "y_prob": msbb_prob,
            "model": model_name,
        })
        msbb_pred_path = os.path.join(output_dir, f"{model_name}_MSBB_full_external_predictions.csv")
        msbb_pred_df.to_csv(msbb_pred_path, index=False)
        all_external_predictions.append(msbb_pred_df)

        pred_cmp = compare_with_previous_predictions(
            previous_out_dir=previous_out_dir,
            out_dir=output_dir,
            model_name=model_name,
            dataset_name="MSBB_full_external",
            new_pred_df=msbb_pred_df,
        )
        if pred_cmp is not None:
            all_prediction_comparisons.append(pred_cmp)

        metric_cmp = compare_metrics(
            previous_out_dir=previous_out_dir,
            out_dir=output_dir,
            model_name=model_name,
            dataset_name="MSBB_full_external",
            new_metrics_df=msbb_metrics,
        )
        if metric_cmp is not None:
            metric_cmp["model"] = model_name
            metric_cmp["dataset"] = "MSBB_full_external"
            all_metric_comparisons.append(metric_cmp)

        if not args.skip_top_bottom:
            print("  Evaluating MSBB top/bottom quartile secondary cohort...")
            msbb_meta = meta.loc[msbb_idx]
            upper_thres = msbb_meta["resilience_score"].quantile(0.75)
            lower_thres = msbb_meta["resilience_score"].quantile(0.25)

            sub_idx = msbb_meta[
                (msbb_meta["resilience_score"] >= upper_thres)
                | (msbb_meta["resilience_score"] <= lower_thres)
            ].index

            X_msbb_sub = X_msbb.loc[sub_idx, final_features]
            y_msbb_sub = y_msbb.loc[sub_idx]
            msbb_sub_prob = pipeline.predict_proba(X_msbb_sub)[:, 1]

            msbb_sub_metrics = evaluate_and_save(
                y_true=y_msbb_sub.values,
                y_prob=msbb_sub_prob,
                model_name=model_name,
                dataset_name="MSBB_top_bottom_quartile_secondary",
                out_dir=output_dir,
            )
            all_metrics.append(msbb_sub_metrics)

            msbb_sub_pred_df = pd.DataFrame({
                "specimenID": y_msbb_sub.index,
                "y_true": y_msbb_sub.values,
                "y_prob": msbb_sub_prob,
                "model": model_name,
            })
            msbb_sub_pred_df.to_csv(
                os.path.join(output_dir, f"{model_name}_MSBB_top_bottom_quartile_secondary_predictions.csv"),
                index=False,
            )

            pred_cmp_sub = compare_with_previous_predictions(
                previous_out_dir=previous_out_dir,
                out_dir=output_dir,
                model_name=model_name,
                dataset_name="MSBB_top_bottom_quartile_secondary",
                new_pred_df=msbb_sub_pred_df,
            )
            if pred_cmp_sub is not None:
                all_prediction_comparisons.append(pred_cmp_sub)

            metric_cmp_sub = compare_metrics(
                previous_out_dir=previous_out_dir,
                out_dir=output_dir,
                model_name=model_name,
                dataset_name="MSBB_top_bottom_quartile_secondary",
                new_metrics_df=msbb_sub_metrics,
            )
            if metric_cmp_sub is not None:
                metric_cmp_sub["model"] = model_name
                metric_cmp_sub["dataset"] = "MSBB_top_bottom_quartile_secondary"
                all_metric_comparisons.append(metric_cmp_sub)

        if args.calculate_shap:
            print("  Calculating interpretation outputs...")
            calculate_and_save_shap(
                pipeline=pipeline,
                model_name=model_name,
                features=final_features,
                X_explain=X_msbb,
                specimen_ids=y_msbb.index,
                shap_out_dir=shap_out_dir,
                shap_background_k=args.shap_background_k,
                shap_nsamples=args.shap_nsamples,
                shap_max_samples=args.shap_max_samples,
                save_long=not args.no_save_shap_long,
            )

    if all_metrics:
        pd.concat(all_metrics, axis=0).to_csv(
            os.path.join(output_dir, "all_model_metrics_with_bootstrap_ci.csv"),
            index=False,
        )

    if all_external_predictions:
        pd.concat(all_external_predictions, axis=0).to_csv(
            os.path.join(output_dir, "all_MSBB_full_external_predictions.csv"),
            index=False,
        )

    if all_final_features:
        pd.concat(all_final_features, axis=0).to_csv(
            os.path.join(output_dir, "all_final_selected_features_ROSMAP_refit.csv"),
            index=False,
        )

    if all_prediction_comparisons:
        pd.concat(all_prediction_comparisons, axis=0).to_csv(
            os.path.join(output_dir, "all_prediction_comparison_summaries.csv"),
            index=False,
        )

    if all_metric_comparisons:
        pd.concat(all_metric_comparisons, axis=0).to_csv(
            os.path.join(output_dir, "all_metric_comparisons_with_previous.csv"),
            index=False,
        )

    pd.DataFrame(saved_model_records).to_csv(
        os.path.join(output_dir, "saved_model_paths.csv"),
        index=False,
    )

    run_config = {
        "base_dir": base_dir,
        "previous_out_dir": previous_out_dir,
        "output_dir": output_dir,
        "meta_path": meta_path,
        "expr_path": expr_path,
        "models": model_names,
        "calculate_shap": args.calculate_shap,
        "shap_background_k": args.shap_background_k,
        "shap_nsamples": args.shap_nsamples,
        "shap_max_samples": args.shap_max_samples,
        "save_shap_long": not args.no_save_shap_long,
        "random_state": RANDOM_STATE,
        "n_bootstrap": N_BOOTSTRAP,
        "note": (
            "This run reused previous final configs and final features. "
            "It did not rerun outer nested CV, final inner CV, ElasticNet selection, or permutation ranking."
        ),
    }
    with open(os.path.join(output_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, ensure_ascii=False)

    print("\nDone.")
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main()
