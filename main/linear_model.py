import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

# Configuration dictionary
config = {
    "learning_rate": 0.01,
    "epochs": 300,
    "batch_size": 16,
    "rand_seed": 52,
    "result_dir": {
        "msbb": "/path/to/results/linear_regression_cv_msbb_all/",
        "rosmap": "/path/to/results/linear_regression_cv_rosmap_all/"
    },
    "model_dir": {
        "msbb": "/path/to/models/linear_regression_cv_msbb_all/",
        "rosmap": "/path/to/models/linear_regression_cv_rosmap_all/"
    }
}

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(f"Using device: {device}")

# Create directories if they don't exist
for key in config['result_dir']:
    os.makedirs(config['result_dir'][key], exist_ok=True)
    os.makedirs(config['model_dir'][key], exist_ok=True)

# List of CSV files for both datasets
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

# Set random seed for reproducibility
torch.manual_seed(config["rand_seed"])
np.random.seed(config["rand_seed"])

# Define the Linear Regression model
class LinearRegressionModel(nn.Module):
    def __init__(self, input_size):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.linear(x)

# KFold Cross Validation function
def perform_kfold_cv(file_path, short_name, dataset_type):
    # Load data from CSV
    df = pd.read_csv(file_path, index_col=0)
    X = df.iloc[:, :-1].values  # All columns except the last one for features
    y = df['norm_resilience_score'].values  # Target variable

    # Normalize features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)  # Ensure y_tensor is of shape (n_samples, 1)

    # Initialize KFold Cross Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=config["rand_seed"])
    results_data = {
        'fold': [],
        'train_rmse': [],
        'test_rmse': [],
        'train_r2': [],
        'test_r2': []
    }

    def compute_metrics(y_true, y_pred):
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        return r2, rmse

    fold = 1
    for train_index, test_index in kf.split(X_tensor):
        X_train, X_test = X_tensor[train_index], X_tensor[test_index]
        y_train, y_test = y_tensor[train_index], y_tensor[test_index]

        # Data loaders for training and testing
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

        # Initialize model
        input_size = X_train.shape[1]
        model = LinearRegressionModel(input_size).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

        # Training loop
        for epoch in range(config["epochs"]):
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                output = model(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()

        # Evaluate model on training and testing data
        model.eval()
        train_preds, train_targets = [], []
        test_preds, test_targets = [], []

        with torch.no_grad():
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = model(X_batch).squeeze().unsqueeze(0)
                train_preds.extend(preds.cpu().numpy().flatten())
                train_targets.extend(y_batch.cpu().numpy())

            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = model(X_batch).squeeze().unsqueeze(0)
                test_preds.extend(preds.cpu().numpy().flatten())
                test_targets.extend(y_batch.cpu().numpy())

        # Compute metrics
        train_r2, train_rmse = compute_metrics(train_targets, train_preds)
        test_r2, test_rmse = compute_metrics(test_targets, test_preds)

        print(f"[{dataset_type}] Fold {fold} - Train R²: {train_r2:.4f}, Train RMSE: {train_rmse:.4f}")
        print(f"[{dataset_type}] Fold {fold} - Test R²: {test_r2:.4f}, Test RMSE: {test_rmse:.4f}")

        results_data['fold'].append(fold)
        results_data['train_rmse'].append(train_rmse)
        results_data['test_rmse'].append(test_rmse)
        results_data['train_r2'].append(train_r2)
        results_data['test_r2'].append(test_r2)

        fold += 1

    # Save fold results to CSV
    results_df = pd.DataFrame(results_data)
    results_file_name = f"{short_name}_cv_results.csv"
    results_df.to_csv(os.path.join(config['result_dir'][dataset_type], results_file_name), index=False)

    # Calculate and save average performance metrics
    mean_train_rmse = np.mean(results_data['train_rmse'])
    mean_test_rmse = np.mean(results_data['test_rmse'])
    mean_train_r2 = np.mean(results_data['train_r2'])
    mean_test_r2 = np.mean(results_data['test_r2'])

    print(f"[{dataset_type}] Average Train R²: {mean_train_r2:.4f}, Average Train RMSE: {mean_train_rmse:.4f}")
    print(f"[{dataset_type}] Average Test R²: {mean_test_r2:.4f}, Average Test RMSE: {mean_test_rmse:.4f}")

    metrics_file_name = f"{short_name}_average_performance_metrics.txt"
    with open(os.path.join(config['result_dir'][dataset_type], metrics_file_name), 'w') as f:
        f.write(f"Average Train R²: {mean_train_r2:.4f}\n")
        f.write(f"Average Train RMSE: {mean_train_rmse:.4f}\n")
        f.write(f"Average Test R²: {mean_test_r2:.4f}\n")
        f.write(f"Average Test RMSE: {mean_test_rmse:.4f}\n")

# Perform KFold CV for each CSV file in both datasets
for dataset_type, files in csv_files.items():
    for file in files:
        short_name = os.path.basename(file).replace("exp_msbb_with.resilience_cor_seqbatch_", "").replace("exp_rosmap_with.resilience_cor_seqbatch_", "").replace(".csv", "")
        perform_kfold_cv(file, short_name, dataset_type)
