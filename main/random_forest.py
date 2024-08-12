import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, GridSearchCV
import joblib
import json

# 設定を辞書で定義
config = {
    "result_dir": {
        "msbb": "/path/to/results/rf_regression_grid_search_cv_msbb_all/",
        "rosmap": "/path/to/results/rf_regression_grid_search_cv_rosmap_all/"
    },
    "model_dir": {
        "msbb": "/path/to/models/rf_regression_grid_search_cv_msbb_all/",
        "rosmap": "/path/to/models/rf_regression_grid_search_cv_rosmap_all/"
    },
    "random_state_out": 52,
    "random_state_in": 62,
    "n_jobs": 12
}

# 必要なディレクトリを作成
for key in config['result_dir']:
    os.makedirs(config['result_dir'][key], exist_ok=True)
    os.makedirs(config['model_dir'][key], exist_ok=True)

# CSVファイルのリスト
csv_files = {
    "msbb": [
        "/path/to/exp_msbb_with.resilience_cor_seqbatch_1000.csv",
        "/path/to/exp_msbb_with.resilience_cor_seqbatch_2000.csv",
        "/path/to/exp_msbb_with.resilience_cor_seqbatch_3000.csv",
        "/path/to/exp_msbb_with.resilience_cor_seqbatch_4000.csv",
        "/path/to/exp_msbb_with.resilience_cor_seqbatch_5000.csv"
    ],
    "rosmap": [
        "/path/to/exp_rosmap_with.resilience_cor_seqbatch_1000.csv",
        "/path/to/exp_rosmap_with.resilience_cor_seqbatch_2000.csv",
        "/path/to/exp_rosmap_with.resilience_cor_seqbatch_3000.csv",
        "/path/to/exp_rosmap_with.resilience_cor_seqbatch_4000.csv",
        "/path/to/exp_rosmap_with.resilience_cor_seqbatch_5000.csv"
    ]
}

# ハイパーパラメータの探索範囲を定義
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5, 10],
    'max_features': ['sqrt', 'log2', None]
}

# ランダムシードを固定
np.random.seed(config["random_state_out"])

def perform_grid_search_cv(file_path, short_name, dataset_type):
    # データの読み込み
    df = pd.read_csv(file_path, index_col=0)
    X = df.iloc[:, :-1].values  # 最後の列以外のすべての列を特徴量として使用
    y = df['norm_resilience_score'].values

    # データの前処理：正規化
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # 外部KFoldクロスバリデーションの設定
    outer_kf = KFold(n_splits=5, shuffle=True, random_state=config["random_state_out"])
    results_data = {
        'fold': [],
        'best_params': [],
        'train_r2': [],
        'test_r2': [],
        'train_rmse': [],
        'test_rmse': []
    }

    fold = 1
    for train_index, test_index in outer_kf.split(X_scaled):
        X_train, X_test = X_scaled[train_index], X_scaled[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # 内部KFoldクロスバリデーションの設定
        inner_kf = KFold(n_splits=5, shuffle=True, random_state=config["random_state_in"])

        # モデルの定義
        base_model = RandomForestRegressor(random_state=config["random_state_out"])

        # GridSearchCVを用いてハイパーパラメータを探索
        grid_search = GridSearchCV(estimator=base_model, param_grid=param_grid, cv=inner_kf, scoring='neg_mean_squared_error', n_jobs=config["n_jobs"])
        grid_search.fit(X_train, y_train)

        # グリッドサーチの結果を取得
        best_params = grid_search.best_params_

        # 最適なハイパーパラメータを表示
        print(f"[{dataset_type}] Fold {fold} - Best Parameters:", best_params)

        # 最適なモデルでの予測
        best_model = grid_search.best_estimator_
        best_model.fit(X_train, y_train)  # 最適なハイパーパラメータで再訓練

        y_train_pred_best = best_model.predict(X_train)
        y_test_pred_best = best_model.predict(X_test)

        # 最適なモデルでの評価
        train_r2_best = r2_score(y_train, y_train_pred_best)
        train_rmse_best = np.sqrt(mean_squared_error(y_train, y_train_pred_best))
        test_r2_best = r2_score(y_test, y_test_pred_best)
        test_rmse_best = np.sqrt(mean_squared_error(y_test, y_test_pred_best))

        results_data['fold'].append(fold)
        results_data['best_params'].append(best_params)
        results_data['train_r2'].append(train_r2_best)
        results_data['test_r2'].append(test_r2_best)
        results_data['train_rmse'].append(train_rmse_best)
        results_data['test_rmse'].append(test_rmse_best)

        # 結果の出力
        print(f"[{dataset_type}] Fold {fold} - Train R²: {train_r2_best:.4f}, Train RMSE: {train_rmse_best:.4f}")
        print(f"[{dataset_type}] Fold {fold} - Test R²: {test_r2_best:.4f}, Test RMSE: {test_rmse_best:.4f}")

        # モデルの保存
        joblib.dump(best_model, os.path.join(config["model_dir"][dataset_type], f"{short_name}_fold_{fold}.joblib"))

        fold += 1

    # 全体の結果をデータフレームに変換してCSV形式で保存
    results_df = pd.DataFrame(results_data)
    results_df['best_params'] = results_df['best_params'].apply(lambda x: json.dumps(x))
    results_df.to_csv(os.path.join(config['result_dir'][dataset_type], f"{short_name}_results.csv"), index=False)

    # 平均評価値の計算と表示
    mean_train_r2 = np.mean(results_data['train_r2'])
    mean_test_r2 = np.mean(results_data['test_r2'])
    mean_train_rmse = np.mean(results_data['train_rmse'])
    mean_test_rmse = np.mean(results_data['test_rmse'])
    print(f"[{dataset_type}] Average Train R²: {mean_train_r2:.4f}, Average Train RMSE: {mean_train_rmse:.4f}")
    print(f"[{dataset_type}] Average Test R²: {mean_test_r2:.4f}, Average Test RMSE: {mean_test_rmse:.4f}")

    # パフォーマンス指標の保存
    with open(os.path.join(config['result_dir'][dataset_type], f"{short_name}_metrics.txt"), 'w') as f:
        f.write(f"Average Train R²: {mean_train_r2:.4f}\n")
        f.write(f"Average Train RMSE: {mean_train_rmse:.4f}\n")
        f.write(f"Average Test R²: {mean_test_r2:.4f}\n")
        f.write(f"Average Test RMSE: {mean_test_rmse:.4f}\n")

# MSBBおよびROSMAPの各CSVファイルに対してグリッドサーチクロスバリデーションを実行
for dataset_type, files in csv_files.items():
    for file in files:
        short_name = os.path.basename(file).replace("exp_msbb_with.resilience_cor_seqbatch_", "").replace("exp_rosmap_with.resilience_cor_seqbatch_", "").replace(".csv", "")
        perform_grid_search_cv(file, short_name, dataset_type)
