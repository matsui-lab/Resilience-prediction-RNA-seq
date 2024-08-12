import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Configuration dictionary for setting model parameters
config = {
    "head_num": 4,
    "learning_rate": 0.0001,
    "dropout_rate": 0.4,
    "act_fun": 'gelu',
    "rand_seed": 52,
    "batch_size": 16,
    "epochs": 500,
    "result_dir_msbb": "/path/to/results_msbb/",
    "model_dir_msbb": "/path/to/models_msbb/",
    "result_dir_rosmap": "/path/to/results_rosmap/",
    "model_dir_rosmap": "/path/to/models_rosmap/"
}

# Ensure result and model directories exist
for dir_path in [config['result_dir_msbb'], config['model_dir_msbb'], config['result_dir_rosmap'], config['model_dir_rosmap']]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# List of CSV files for both MSBB and ROSMAP datasets
csv_files_msbb = [
    "/path/to/exp_msbb_with.resilience_cor_seqbatch_1000.csv",
    "/path/to/exp_msbb_with.resilience_cor_seqbatch_2000.csv",
    "/path/to/exp_msbb_with.resilience_cor_seqbatch_3000.csv",
    "/path/to/exp_msbb_with.resilience_cor_seqbatch_4000.csv",
    "/path/to/exp_msbb_with.resilience_cor_seqbatch_5000.csv"
]

csv_files_rosmap = [
    "/path/to/exp_rosmap_with.resilience_cor_seqbatch_1000.csv",
    "/path/to/exp_rosmap_with.resilience_cor_seqbatch_2000.csv",
    "/path/to/exp_rosmap_with.resilience_cor_seqbatch_3000.csv",
    "/path/to/exp_rosmap_with.resilience_cor_seqbatch_4000.csv",
    "/path/to/exp_rosmap_with.resilience_cor_seqbatch_5000.csv"
]

# Set random seed for reproducibility
torch.manual_seed(config["rand_seed"])
np.random.seed(config["rand_seed"])

# Set the device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the multi-head attention layer
class MultiAttention(nn.Module):
    def __init__(self, batch_size, n_head, n_gene, n_feature, query_gene, mode):
        super(MultiAttention, self).__init__()
        self.n_head = n_head
        self.n_gene = n_gene
        self.batch_size = batch_size
        self.n_feature = n_feature
        self.mode = mode
        self.query_gene = query_gene

        self.WQ = nn.Parameter(torch.Tensor(self.n_head, self.n_feature, 1))
        self.WK = nn.Parameter(torch.Tensor(self.n_head, self.n_feature, 1))
        self.WV = nn.Parameter(torch.Tensor(self.n_head, self.n_feature, 1))
        torch.nn.init.xavier_normal_(self.WQ)
        torch.nn.init.xavier_normal_(self.WK)
        torch.nn.init.xavier_normal_(self.WV)
        self.W_0 = nn.Parameter(torch.Tensor(self.n_head * [0.001]))

    def QK_diff(self, Q_seq, K_seq):
        QK_dif = -1 * torch.pow((Q_seq - K_seq), 2)
        return torch.nn.Softmax(dim=2)(QK_dif)

    def mask_softmax_self(self, x):
        d = x.shape[1]
        x = x * ((1 - torch.eye(d, d)).to(device))
        return x

    def attention(self, x, Q_seq, WK, WV):
        if self.mode == 0:
            K_seq = x * WK
            K_seq = K_seq.expand(K_seq.shape[0], K_seq.shape[1], self.n_gene)
            K_seq = K_seq.permute(0, 2, 1)
            V_seq = x * WV
            QK_product = Q_seq * K_seq
            z = torch.nn.Softmax(dim=2)(QK_product)
            z = self.mask_softmax_self(z)
            out_seq = torch.matmul(z, V_seq)

        elif self.mode == 1:
            zz_list = []
            for q in range(self.n_gene // self.query_gene):
                K_seq = x * WK
                V_seq = x * WV
                Q_seq_x = x[:, (q * self.query_gene):((q + 1) * self.query_gene), :]
                Q_seq = Q_seq_x.expand(Q_seq_x.shape[0], Q_seq_x.shape[1], self.n_gene)
                K_seq = K_seq.expand(K_seq.shape[0], K_seq.shape[1], self.query_gene)
                K_seq = K_seq.permute(0, 2, 1)
                QK_diff = self.QK_diff(Q_seq, K_seq)
                z = torch.nn.Softmax(dim=2)(QK_diff)
                z = torch.matmul(z, V_seq)
                zz_list.append(z)
            out_seq = torch.cat(zz_list, dim=1)
        return out_seq

    def forward(self, x):
        x = torch.reshape(x, (x.shape[0], x.shape[1], 1))
        out_h = []
        for h in range(self.n_head):
            Q_seq = x * self.WQ[h, :, :]
            Q_seq = Q_seq.expand(Q_seq.shape[0], Q_seq.shape[1], self.n_gene)
            attention_out = self.attention(x, Q_seq, self.WK[h, :, :], self.WV[h, :, :])
            out_h.append(attention_out)
        out_seq = torch.cat(out_h, dim=2)
        out_seq = torch.matmul(out_seq, self.W_0)
        return out_seq

# Define the layer normalization layer
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

# Define the residual connection layer
class ResConnect(nn.Module):
    def __init__(self, size, dropout):
        super(ResConnect, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, out):
        return x + self.norm(self.dropout(out))

# Define the main network architecture
class MyNet(nn.Module):
    def __init__(self, batch_size, n_head, n_gene, n_feature, query_gene, d_ff, dropout_rate, mode, act_fun):
        super(MyNet, self).__init__()
        self.n_head = n_head
        self.n_gene = n_gene
        self.batch_size = batch_size
        self.n_feature = n_feature
        self.d_ff = d_ff
        self.act_fun = act_fun
        self.mulitiattention1 = MultiAttention(self.batch_size, self.n_head, self.n_gene, self.n_feature, query_gene, mode)
        self.mulitiattention2 = MultiAttention(self.batch_size, self.n_head, self.n_gene, self.n_feature, query_gene, mode)
        self.fc = nn.Linear(self.n_gene, 1)
        torch.nn.init.xavier_uniform_(self.fc.weight)
        self.ffn1 = nn.Linear(self.n_gene, self.d_ff)
        self.ffn2 = nn.Linear(self.d_ff, self.n_gene)
        self.dropout = nn.Dropout(dropout_rate)
        self.sublayer = ResConnect(n_gene, dropout_rate)

    def feedforward(self, x):
        out = F.relu(self.ffn1(x))
        out = self.ffn2(self.dropout(out))
        return out

    def forward(self, x):
        out_attn = self.mulitiattention1(x)
        out_attn_1 = self.sublayer(x, out_attn)
        out_attn_2 = self.mulitiattention2(out_attn_1)
        out_attn_2 = self.sublayer(out_attn_1, out_attn_2)
        if self.act_fun == 'relu':
            out_attn_2 = F.relu(out_attn_2)
        elif self.act_fun == 'leakyrelu':
            out_attn_2 = torch.nn.LeakyReLU(0.1)(out_attn_2)
        elif self.act_fun == 'gelu':
            out_attn_2 = torch.nn.GELU()(out_attn_2)
        y_pred = self.fc(out_attn_2)
        return y_pred

# Define a function to perform KFold Cross Validation
def perform_kfold_cv(file_path, short_name, result_dir, model_dir):
    df = pd.read_csv(file_path, index_col=0)
    X = df.iloc[:, :-1].values
    y = df['norm_resilience_score'].values

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    n_gene = X_tensor.shape[1]
    query_gene = 64

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

        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

        model = MyNet(batch_size=config["batch_size"], n_head=config["head_num"], n_gene=n_gene,
                      n_feature=n_gene, query_gene=query_gene, d_ff=256, dropout_rate=config["dropout_rate"], mode=0, act_fun=config["act_fun"]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
        criterion = nn.MSELoss()

        train_loss_list = []
        test_loss_list = []

        for epoch in range(config["epochs"]):
            model.train()
            train_loss = 0

            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                output = model(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)
            train_loss_list.append(train_loss)

            model.eval()
            test_loss = 0

            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    output = model(X_batch)
                    loss = criterion(output, y_batch)
                    test_loss += loss.item()

            test_loss /= len(test_loader)
            test_loss_list.append(test_loss)

        # Making predictions
        model.eval()
        train_preds = []
        train_targets = []
        test_preds = []
        test_targets = []
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

        train_r2, train_rmse = compute_metrics(train_targets, train_preds)
        test_r2, test_rmse = compute_metrics(test_targets, test_preds)

        print(f"Fold {fold} - Train R²: {train_r2:.4f}, Train RMSE: {train_rmse:.4f}")
        print(f"Fold {fold} - Test R²: {test_r2:.4f}, Test RMSE: {test_rmse:.4f}")

        results_data['fold'].append(fold)
        results_data['train_rmse'].append(train_rmse)
        results_data['test_rmse'].append(test_rmse)
        results_data['train_r2'].append(train_r2)
        results_data['test_r2'].append(test_r2)

        # Save model for each fold
        torch.save(model.state_dict(), os.path.join(model_dir, f"{short_name}_fold_{fold}.pt"))

        fold += 1

    # Save results to DataFrame and CSV
    results_df = pd.DataFrame(results_data)
    results_file_name = f"{short_name}_cv_results.csv"
    results_df.to_csv(os.path.join(result_dir, results_file_name), index=False)

    # Save average performance metrics
    mean_train_rmse = np.mean(results_data['train_rmse'])
    mean_test_rmse = np.mean(results_data['test_rmse'])
    mean_train_r2 = np.mean(results_data['train_r2'])
    mean_test_r2 = np.mean(results_data['test_r2'])

    print(f"Average Train R²: {mean_train_r2:.4f}, Average Train RMSE: {mean_train_rmse:.4f}")
    print(f"Average Test R²: {mean_test_r2:.4f}, Average Test RMSE: {mean_test_rmse:.4f}")

    metrics_file_name = f"{short_name}_average_performance_metrics.txt"
    with open(os.path.join(result_dir, metrics_file_name), 'w') as f:
        f.write(f"Average Train R²: {mean_train_r2:.4f}\n")
        f.write(f"Average Train RMSE: {mean_train_rmse:.4f}\n")
        f.write(f"Average Test R²: {mean_test_r2:.4f}\n")
        f.write(f"Average Test RMSE: {mean_test_rmse:.4f}\n")

    # Save learning curve
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, config["epochs"] + 1), train_loss_list, 'b', label='Training loss')
    plt.plot(range(1, config["epochs"] + 1), test_loss_list, 'k', label='Test loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Learning Curve (Loss)')
    plt.legend()
    plt.savefig(os.path.join(result_dir, f"{short_name}_learning_curve_loss_cv.png"))
    plt.close()

    # Save scatter plot of true vs predicted values
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(train_targets, train_preds, alpha=0.5)
    plt.title('Train Data: True vs Predicted')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.subplot(1, 2, 2)
    plt.scatter(test_targets, test_preds, alpha=0.5)
    plt.title('Test Data: True vs Predicted')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, f"{short_name}_scatter_true_vs_predicted_cv.png"))
    plt.close()

# Run KFold CV for each CSV file in the MSBB dataset
for file in csv_files_msbb:
    short_name = os.path.basename(file).replace("exp_msbb_with.resilience_cor_seqbatch_", "").replace(".csv", "")
    perform_kfold_cv(file, short_name, config['result_dir_msbb'], config['model_dir_msbb'])

# Run KFold CV for each CSV file in the ROSMAP dataset
for file in csv_files_rosmap:
    short_name = os.path.basename(file).replace("exp_rosmap_with.resilience_cor_seqbatch_", "").replace(".csv", "")
    perform_kfold_cv(file, short_name, config['result_dir_rosmap'], config['model_dir_rosmap'])
