import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import shap
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

# Configuration dictionary
config = {
    "result_dir": "/path/to/results/svm_regression_grid_search_cv_msbb_all/",
    "model_path": "/path/to/models/svm_regression_grid_search_cv_msbb_all/3000_fold_2.joblib",
    "data_file": "/path/to/exp_msbb_with.resilience_cor_seqbatch_3000.csv",
    "gene_mapping_file": "/path/to/human_ensembl.txt",
    "random_state": 52,
    "test_sample_size": 100  # Number of samples for SHAP analysis
}

# Set random seed for reproducibility
np.random.seed(config["random_state"])

# Create result directory if it doesn't exist
if not os.path.exists(config['result_dir']):
    os.makedirs(config['result_dir'])

# Load the dataset
df = pd.read_csv(config["data_file"], index_col=0)
X = df.iloc[:, :3000].values  # Use the first 3000 features
y = df['norm_resilience_score'].values

# Normalize the data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Extract feature names
feature_names = df.columns[:3000]

# Load gene mapping data
gene_mapping = pd.read_csv(config["gene_mapping_file"])

# Create a mapping dictionary for Ensembl ID to gene symbol, excluding missing values
gene_mapping = gene_mapping.dropna(subset=["Gene stable ID", "Gene name"])
ensembl_to_symbol = dict(zip(gene_mapping["Gene stable ID"], gene_mapping["Gene name"]))

# Convert feature names to gene symbols where possible
feature_names = [ensembl_to_symbol.get(name, name) for name in feature_names]

# Load the trained SVM model
best_model = joblib.load(config["model_path"])

# Set up KFold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=config["random_state"])

# Extract test indices for the selected fold (fold=2 in this case)
for fold, (train_index, test_index) in enumerate(kf.split(X_scaled)):
    if fold == 1:  # Corresponds to the 2nd fold (index 1)
        X_train, X_test = X_scaled[train_index], X_scaled[test_index]
        y_train, y_test = y[train_index], y[test_index]
        break

# Sample the test set for SHAP analysis
np.random.seed(config["random_state"])
if len(test_index) > config["test_sample_size"]:
    test_sample_indices = np.random.choice(test_index, config["test_sample_size"], replace=False)
else:
    test_sample_indices = test_index

X_test_sample = X_scaled[test_sample_indices]

# Function to generate SHAP summary plot and save SHAP values
def plot_shap_summary(model, X, feature_names, result_dir):
    explainer = shap.KernelExplainer(model.predict, X)
    shap_values = explainer.shap_values(X)
    
    # Generate and save SHAP summary plot
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    plt.savefig(os.path.join(result_dir, 'shap_summary_3000_fold_2.png'))
    plt.close()
    
    # Save SHAP values to CSV
    shap_values_df = pd.DataFrame(shap_values, columns=feature_names)
    shap_values_df.to_csv(os.path.join(result_dir, 'shap_values_3000_fold_2.csv'), index=False)

# Generate and save SHAP summary plot and SHAP values
plot_shap_summary(best_model, X_test_sample, feature_names, config['result_dir'])

print("SHAP summary plot and values saved.")
