# Script cleaned for public release. Edit /path/to/... inputs before running.
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

N_JOBS_GRID = 16
N_JOBS_PI = 1
N_JOBS_MODEL = 1

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import parallel_backend

from sklearn.model_selection import StratifiedKFold, GridSearchCV, ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
    brier_score_loss
)
from sklearn.calibration import calibration_curve

warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

RANDOM_STATE = 42
N_OUTER = 5
N_INNER = 5
N_BOOTSTRAP = 1000

base_dir = "/path/to/project"
out_dir = os.path.join(base_dir, "out", "ml_nested_cv")
os.makedirs(out_dir, exist_ok=True)

meta_path = os.path.join(base_dir, "out/combined/meta_merged_exclude89.csv")
expr_path = os.path.join(base_dir, "out/combined/exp_merged_combat_exclude89.csv")

feature_count_grid = [5, 10, 20, 50, 100, 200]

print("1. Loading data...")

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

print("ROSMAP:", X_rosmap.shape)
print("MSBB:", X_msbb.shape)

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
            l1_ratio=params.get("l1_ratio", 0.5)
        )
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", clf)
        ])

    if model_name == "SVM":
        clf = SVC(
            probability=True,
            random_state=RANDOM_STATE,
            C=params.get("C", 1.0),
            gamma=params.get("gamma", "scale"),
            kernel=params.get("kernel", "rbf")
        )
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", clf)
        ])

    if model_name == "RandomForest":
        clf = RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_estimators=params.get("n_estimators", 200),
            max_depth=params.get("max_depth", None),
            min_samples_leaf=params.get("min_samples_leaf", 1),
            n_jobs=N_JOBS_MODEL
        )
        return Pipeline([
            ("clf", clf)
        ])

    if model_name == "XGBoost":
        clf = xgb.XGBClassifier(
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", 3),
            learning_rate=params.get("learning_rate", 0.1),
            subsample=params.get("subsample", 1.0),
            colsample_bytree=params.get("colsample_bytree", 1.0),
            n_jobs=N_JOBS_MODEL
        )
        return Pipeline([
            ("clf", clf)
        ])

    raise ValueError(f"Unknown model: {model_name}")

def get_model_param_grid(model_name):
    if model_name == "ElasticNet":
        return list(ParameterGrid({
            "C": [0.01, 0.1, 1, 10, 100],
            "l1_ratio": [0.1, 0.5, 0.9]
        }))

    if model_name == "SVM":
        return list(ParameterGrid({
            "C": [0.1, 1, 10, 100],
            "gamma": ["scale", "auto", 0.01, 0.1],
            "kernel": ["rbf"]
        }))

    if model_name == "RandomForest":
        return list(ParameterGrid({
            "n_estimators": [100, 200],
            "max_depth": [3, 5, 10, None],
            "min_samples_leaf": [1, 3, 5]
        }))

    if model_name == "XGBoost":
        return list(ParameterGrid({
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5],
            "learning_rate": [0.01, 0.1],
            "subsample": [0.7, 1.0],
            "colsample_bytree": [0.7, 1.0]
        }))

    raise ValueError(f"Unknown model: {model_name}")

def fit_elasticnet_selector(X, y):
    cv = StratifiedKFold(
        n_splits=N_INNER,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    en_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            max_iter=10000,
            random_state=RANDOM_STATE
        ))
    ])

    en_param = {
        "logreg__C": [0.001, 0.01, 0.1, 1],
        "logreg__l1_ratio": [0.5, 0.7, 0.9, 1.0]
        }

    gs = GridSearchCV(
        en_pipeline,
        en_param,
        cv=cv,
        scoring="roc_auc",
        n_jobs=N_JOBS_GRID,
        verbose=1
    )
    with parallel_backend("threading"):
        gs.fit(X, y)

    coef = gs.best_estimator_.named_steps["logreg"].coef_[0]

    coef_abs = np.abs(coef)
    nonzero_idx = np.where(coef_abs > 0)[0]

    MAX_EN_FEATURES = 500

    if len(nonzero_idx) > MAX_EN_FEATURES:
        top_idx = nonzero_idx[np.argsort(coef_abs[nonzero_idx])[::-1][:MAX_EN_FEATURES]]
    else:
        top_idx = nonzero_idx

    selected = X.columns[top_idx].tolist()

    return selected, gs.best_params_

def rank_features_by_permutation_importance(model_name, params, X, y, features):
    clf = make_model(model_name, params)
    clf.fit(X[features], y)

    result = permutation_importance(
        clf,
        X[features],
        y,
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS_PI,
        scoring="roc_auc"
    )

    importances = result.importances_mean
    order = np.argsort(importances)[::-1]
    ranked_features = np.array(features)[order].tolist()

    return ranked_features

def get_valid_feature_counts(n_features):
    counts = sorted(set([n for n in feature_count_grid if n <= n_features]))
    if len(counts) == 0:
        counts = [n_features]
    if counts[-1] != n_features:
        counts.append(n_features)
    return counts

def select_config_inner_cv(X, y, model_name):
    print(f"Selecting config for {model_name} using inner CV only...")

    param_grid = get_model_param_grid(model_name)
    inner_cv = StratifiedKFold(
        n_splits=N_INNER,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    records = []

    for inner_fold, (train_idx, val_idx) in enumerate(inner_cv.split(X, y), start=1):
        print(f"  Inner fold {inner_fold}/{N_INNER}: ElasticNet feature selection...")

        X_tr = X.iloc[train_idx]
        y_tr = y.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx]

        selected_features, en_params = fit_elasticnet_selector(X_tr, y_tr)

        if len(selected_features) == 0:
            continue

        print(f"  Inner fold {inner_fold}: {len(selected_features)} features selected by ElasticNet")

        for params in param_grid:
            ranked_features = rank_features_by_permutation_importance(
                model_name=model_name,
                params=params,
                X=X_tr,
                y=y_tr,
                features=selected_features
            )

            valid_counts = get_valid_feature_counts(len(ranked_features))

            for n_features in valid_counts:
                use_features = ranked_features[:n_features]

                clf = make_model(model_name, params)
                clf.fit(X_tr[use_features], y_tr)

                prob = clf.predict_proba(X_val[use_features])[:, 1]
                auc = roc_auc_score(y_val, prob)

                records.append({
                    "model": model_name,
                    "params": str(params),
                    "n_features": n_features,
                    "inner_auc": auc,
                    "inner_fold": inner_fold,
                    "elasticnet_n_selected": len(selected_features),
                    "elasticnet_params": str(en_params)
                })

    result_df = pd.DataFrame(records)

    if result_df.empty:
        raise RuntimeError(f"No valid inner CV result for {model_name}")

    summary = (
        result_df
        .groupby(["params", "n_features"], as_index=False)
        .agg(mean_inner_auc=("inner_auc", "mean"))
        .sort_values("mean_inner_auc", ascending=False)
    )

    best_row = summary.iloc[0]
    best_params = eval(best_row["params"])
    best_n_features = int(best_row["n_features"])

    return best_params, best_n_features, result_df, summary

def fit_final_pipeline_on_training_data(X, y, model_name, params, n_features):
    selected_features, en_params = fit_elasticnet_selector(X, y)

    if len(selected_features) == 0:
        raise RuntimeError("ElasticNet selector selected zero features.")

    ranked_features = rank_features_by_permutation_importance(
        model_name=model_name,
        params=params,
        X=X,
        y=y,
        features=selected_features
    )

    n_use = min(n_features, len(ranked_features))
    final_features = ranked_features[:n_use]

    clf = make_model(model_name, params)
    clf.fit(X[final_features], y)

    return clf, final_features, selected_features, en_params

def calculate_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "AUROC": roc_auc_score(y_true, y_prob),
        "AUPRC": average_precision_score(y_true, y_prob),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred),
        "Brier": brier_score_loss(y_true, y_prob)
    }

    return metrics

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

        m = calculate_metrics(y_b, p_b)
        boot_records.append(m)

    boot_df = pd.DataFrame(boot_records)

    ci_records = []
    for metric in boot_df.columns:
        ci_records.append({
            "metric": metric,
            "lower_95CI": np.percentile(boot_df[metric], 2.5),
            "upper_95CI": np.percentile(boot_df[metric], 97.5)
        })

    return pd.DataFrame(ci_records), boot_df

def save_calibration_plot(y_true, y_prob, title, out_path):
    prob_true, prob_pred = calibration_curve(
        y_true,
        y_prob,
        n_bins=10,
        strategy="quantile"
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

def evaluate_and_save(y_true, y_prob, model_name, dataset_name):
    metrics = calculate_metrics(y_true, y_prob)
    ci_df, boot_df = bootstrap_ci(y_true, y_prob)

    metrics_df = pd.DataFrame({
        "metric": list(metrics.keys()),
        "value": list(metrics.values())
    })

    metrics_df = metrics_df.merge(ci_df, on="metric", how="left")
    metrics_df["model"] = model_name
    metrics_df["dataset"] = dataset_name

    prefix = f"{model_name}_{dataset_name}".replace(" ", "_")

    metrics_df.to_csv(
        os.path.join(out_dir, f"{prefix}_metrics_bootstrap_ci.csv"),
        index=False
    )

    boot_df.to_csv(
        os.path.join(out_dir, f"{prefix}_bootstrap_distribution.csv"),
        index=False
    )

    save_calibration_plot(
        y_true,
        y_prob,
        title=f"{model_name} calibration: {dataset_name}",
        out_path=os.path.join(out_dir, f"{prefix}_calibration_curve.svg")
    )

    y_pred = (np.asarray(y_prob) >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=["true_0", "true_1"],
        columns=["pred_0", "pred_1"]
    )
    cm_df.to_csv(os.path.join(out_dir, f"{prefix}_confusion_matrix.csv"))

    return metrics_df

model_names = ["ElasticNet", "SVM", "RandomForest"]
if HAS_XGB:
    model_names.append("XGBoost")

all_metrics = []
all_outer_predictions = []
all_outer_configs = []
all_external_predictions = []
all_final_features = []

for model_name in model_names:
    print("\n===================================================")
    print(f"Running model: {model_name}")
    print("===================================================")

    outer_cv = StratifiedKFold(
        n_splits=N_OUTER,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    oof_prob = pd.Series(index=y_rosmap.index, dtype=float)
    oof_fold = pd.Series(index=y_rosmap.index, dtype=int)

    for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X_rosmap, y_rosmap), start=1):
        print(f"\nOuter fold {outer_fold}/{N_OUTER}: {model_name}")

        X_outer_train = X_rosmap.iloc[train_idx]
        y_outer_train = y_rosmap.iloc[train_idx]

        X_outer_test = X_rosmap.iloc[test_idx]
        y_outer_test = y_rosmap.iloc[test_idx]

        best_params, best_n_features, inner_records, inner_summary = select_config_inner_cv(
            X_outer_train,
            y_outer_train,
            model_name
        )

        inner_records.to_csv(
            os.path.join(out_dir, f"{model_name}_outer{outer_fold}_inner_records.csv"),
            index=False
        )
        inner_summary.to_csv(
            os.path.join(out_dir, f"{model_name}_outer{outer_fold}_inner_summary.csv"),
            index=False
        )

        clf, final_features, en_selected_features, en_params = fit_final_pipeline_on_training_data(
            X_outer_train,
            y_outer_train,
            model_name,
            best_params,
            best_n_features
        )

        prob = clf.predict_proba(X_outer_test[final_features])[:, 1]

        oof_prob.iloc[test_idx] = prob
        oof_fold.iloc[test_idx] = outer_fold

        all_outer_configs.append({
            "model": model_name,
            "outer_fold": outer_fold,
            "best_params": str(best_params),
            "best_n_features": best_n_features,
            "actual_n_features": len(final_features),
            "elasticnet_n_selected": len(en_selected_features),
            "elasticnet_params": str(en_params)
        })

        pd.DataFrame({
            "gene": final_features
        }).to_csv(
            os.path.join(out_dir, f"{model_name}_outer{outer_fold}_final_features.csv"),
            index=False
        )

    print(f"\nEvaluating ROSMAP nested-CV OOF performance: {model_name}")

    rosmap_metrics = evaluate_and_save(
        y_true=y_rosmap.values,
        y_prob=oof_prob.values,
        model_name=model_name,
        dataset_name="ROSMAP_nestedCV_OOF"
    )

    all_metrics.append(rosmap_metrics)

    outer_pred_df = pd.DataFrame({
        "specimenID": y_rosmap.index,
        "y_true": y_rosmap.values,
        "y_prob": oof_prob.values,
        "outer_fold": oof_fold.values,
        "model": model_name
    })

    outer_pred_df.to_csv(
        os.path.join(out_dir, f"{model_name}_ROSMAP_nestedCV_OOF_predictions.csv"),
        index=False
    )

    all_outer_predictions.append(outer_pred_df)

    print(f"\nSelecting final config on full ROSMAP only: {model_name}")

    final_params, final_n_features, final_inner_records, final_inner_summary = select_config_inner_cv(
        X_rosmap,
        y_rosmap,
        model_name
    )

    final_inner_records.to_csv(
        os.path.join(out_dir, f"{model_name}_final_ROSMAP_inner_records.csv"),
        index=False
    )
    final_inner_summary.to_csv(
        os.path.join(out_dir, f"{model_name}_final_ROSMAP_inner_summary.csv"),
        index=False
    )

    final_clf, final_features, en_selected_features, en_params = fit_final_pipeline_on_training_data(
        X_rosmap,
        y_rosmap,
        model_name,
        final_params,
        final_n_features
    )

    pd.DataFrame({
        "gene": final_features
    }).to_csv(
        os.path.join(out_dir, f"{model_name}_final_features_ROSMAP_trained.csv"),
        index=False
    )

    all_final_features.append(pd.DataFrame({
        "model": model_name,
        "gene": final_features
    }))

    print(f"\nExternal validation on untouched MSBB full cohort: {model_name}")

    msbb_prob = final_clf.predict_proba(X_msbb[final_features])[:, 1]

    msbb_metrics = evaluate_and_save(
        y_true=y_msbb.values,
        y_prob=msbb_prob,
        model_name=model_name,
        dataset_name="MSBB_full_external"
    )

    all_metrics.append(msbb_metrics)

    msbb_pred_df = pd.DataFrame({
        "specimenID": y_msbb.index,
        "y_true": y_msbb.values,
        "y_prob": msbb_prob,
        "model": model_name
    })

    msbb_pred_df.to_csv(
        os.path.join(out_dir, f"{model_name}_MSBB_full_external_predictions.csv"),
        index=False
    )

    all_external_predictions.append(msbb_pred_df)

    print(f"\nSecondary analysis on MSBB top/bottom quartile: {model_name}")

    msbb_meta = meta.loc[msbb_idx]
    upper_thres = msbb_meta["resilience_score"].quantile(0.75)
    lower_thres = msbb_meta["resilience_score"].quantile(0.25)

    sub_idx = msbb_meta[
        (msbb_meta["resilience_score"] >= upper_thres) |
        (msbb_meta["resilience_score"] <= lower_thres)
    ].index

    X_msbb_sub = X_msbb.loc[sub_idx, final_features]
    y_msbb_sub = y_msbb.loc[sub_idx]

    msbb_sub_prob = final_clf.predict_proba(X_msbb_sub)[:, 1]

    msbb_sub_metrics = evaluate_and_save(
        y_true=y_msbb_sub.values,
        y_prob=msbb_sub_prob,
        model_name=model_name,
        dataset_name="MSBB_top_bottom_quartile_secondary"
    )

    all_metrics.append(msbb_sub_metrics)

metrics_all_df = pd.concat(all_metrics, axis=0)
metrics_all_df.to_csv(
    os.path.join(out_dir, "all_model_metrics_with_bootstrap_ci.csv"),
    index=False
)

configs_df = pd.DataFrame(all_outer_configs)
configs_df.to_csv(
    os.path.join(out_dir, "outer_fold_selected_configs.csv"),
    index=False
)

pd.concat(all_outer_predictions, axis=0).to_csv(
    os.path.join(out_dir, "all_ROSMAP_nestedCV_OOF_predictions.csv"),
    index=False
)

pd.concat(all_external_predictions, axis=0).to_csv(
    os.path.join(out_dir, "all_MSBB_full_external_predictions.csv"),
    index=False
)

pd.concat(all_final_features, axis=0).to_csv(
    os.path.join(out_dir, "all_final_selected_features_ROSMAP_trained.csv"),
    index=False
)

print("\nDone.")
print(f"Results saved to: {out_dir}")
