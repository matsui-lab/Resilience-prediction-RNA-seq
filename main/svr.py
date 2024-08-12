import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import json

# Configuration dictionary
config = {
    "result_dir": {
        "msbb": "/path/to/results/svm_regression_grid_search_cv_msbb_all/",
        "rosmap": "/path/to/results/svm_regression_grid_search_cv_rosmap_all/"
    },
    "model_dir": {
        "msbb": "/path/to/models/svm_regression_grid_search_cv_msbb_all/",
        "rosmap": "/path/to/models/svm_regression_grid_search_cv_rosmap_all/"
    },
    "random_state_out": 52,
    "random_state_in": 62,
    "n_jobs": 16
}

# Create necessary directories
for key in config['result_dir']:
    os.makedirs(config['result_dir'][key], exist_ok=True)
    os.makedirs(config['model_dir'][key], exist_ok=True)

# List of CSV files for MSBB and ROSMAP datasets
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

# Define the hyperparameter search space for the SVM model
param_grid = {
    'C': [0.1, 1, 10],
    'epsilon': [0.01, 0.05, 0.1],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
}

# Set random seed for reproducibility
np.random.seed(config["random_state_out"])

def perform_grid_search_cv(file_path, short_name, dataset_type):
    # Load the dataset
    df = pd.read_csv(file_path, index_col=0)
    X = df.iloc[:, :-1].values  # Use all columns except the last one as features
    y = df['norm_resilience_score'].values  # Use the last column as the target variable

    # Data preprocessing: normalize the features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Configure the outer KFold cross-validation
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

        inner_kf = KFold(n_splits=5, shuffle=True, random_state=config["random_state_in"])

        # Define the base model
        base_model = SVR()

        # Perform grid search with cross-validation to find the best hyperparameters
        grid_search = GridSearchCV(estimator=base_model, param_grid=param_grid, cv=inner_kf, scoring='neg_mean_squared_error', n_jobs=config["n_jobs"])
        grid_search.fit(X_train, y_train)

        # Retrieve the best hyperparameters
        best_params = grid_search.best_params_

        # Print the best parameters for this fold
        print(f"[{dataset_type}] Fold {fold} - Best Parameters:", best_params)

        # Train the best model on the training set
        best_model = grid_search.best_estimator_
        best_model.fit(X_train, y_train)

        # Predict on both training and test sets
        y_train_pred_best = best_model.predict(X_train)
        y_test_pred_best = best_model.predict(X_test)

        # Evaluate the model
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred_best))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred_best))
        train_r2 = r2_score(y_train, y_train_pred_best)
        test_r2 = r2_score(y_test, y_test_pred_best)

        results_data['fold'].append(fold)
        results_data['best_params'].append(best_params)
        results_data['train_r2'].append(train_r2)
        results_data['test_r2'].append(test_r2)
        results_data['train_rmse'].append(train_rmse)
        results_data['test_rmse'].append(test_rmse)

        # Output the results for this fold
        print(f"[{dataset_type}] Fold {fold} - Train R²: {train_r2:.4f}, Train RMSE: {train_rmse:.4f}")
        print(f"[{dataset_type}] Fold {fold} - Test R²: {test_r2:.4f}, Test RMSE: {test_rmse:.4f}")

        # Save the best model for this fold
        joblib.dump(best_model, os.path.join(config["model_dir"][dataset_type], f"{short_name}_fold_{fold}.joblib"))

        fold += 1

    # Convert results to a DataFrame and save as CSV
    results_df = pd.DataFrame(results_data)
    results_df['best_params'] = results_df['best_params'].apply(lambda x: json.dumps(x))
    results_df.to_csv(os.path.join(config['result_dir'][dataset_type], f"{short_name}_results.csv"), index=False)

    # Calculate and print average performance metrics
    mean_train_rmse = np.mean(results_data['train_rmse'])
    mean_test_rmse = np.mean(results_data['test_rmse'])
    mean_train_r2 = np.mean(results_data['train_r2'])
    mean_test_r2 = np.mean(results_data['test_r2'])
    print(f"[{dataset_type}] Average Train R²: {mean_train_r2:.4f}, Average Train RMSE: {mean_train_rmse:.4f}")
    print(f"[{dataset_type}] Average Test R²: {mean_test_r2:.4f}, Average Test RMSE: {mean_test_rmse:.4f}")

    # Save the average performance metrics to a text file
    with open(os.path.join(config['result_dir'][dataset_type], f"{short_name}_metrics.txt"), 'w') as f:
        f.write(f"Average Train R²: {mean_train_r2:.4f}\n")
        f.write(f"Average Train RMSE: {mean_train_rmse:.4f}\n")
        f.write(f"Average Test R²: {mean_test_r2:.4f}\n")
        f.write(f"Average Test RMSE: {mean_test_rmse:.4f}\n")

# Run grid search cross-validation for each CSV file in both MSBB and ROSMAP datasets
for dataset_type, files in csv_files.items():
    for file in files:
        short_name = os.path.basename(file).replace("exp_msbb_with.resilience_cor_seqbatch_", "").replace("exp_rosmap_with.resilience_cor_seqbatch_", "").replace(".csv", "")
        perform_grid_search_cv(file, short_name, dataset_type)
