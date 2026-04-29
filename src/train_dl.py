import os
import json
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

import sys
sys.path.append('..')
from src.preprocess import preprocess, TARGET_ENCODE_COLS
from src.features import add_features

N_SPLITS    = 10
RANDOM_SEED = 42
MODEL_DIR   = '../saved_models'
DATA_PATH   = '../data/raw/'

DROP_COLS = ['ID', '시술 당시 나이', '임신 성공 여부']


def load_data():
    train = pd.read_csv(DATA_PATH + 'train.csv')
    test  = pd.read_csv(DATA_PATH + 'test.csv')
    return train, test


def prepare(train, test, use_features=True):
    train, test = preprocess(train, test)
    if use_features:
        train = add_features(train)
        test  = add_features(test)
    y      = train['임신 성공 여부']
    X      = train.drop(columns=[c for c in DROP_COLS if c in train.columns])
    X_test = test.drop(columns=[c for c in DROP_COLS if c in test.columns])
    return X, y, X_test


def encode_for_nn(X_train, X_test):
    X_train = X_train.copy()
    X_test  = X_test.copy()
    obj_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in obj_cols:
        le = LabelEncoder()
        train_vals = X_train[col].fillna('__nan__').astype(str)
        test_vals  = X_test[col].fillna('__nan__').astype(str)
        le.fit(pd.concat([train_vals, test_vals], axis=0))
        X_train[col] = le.transform(train_vals)
        X_test[col]  = test_vals.map(lambda x, le=le: le.transform([x])[0] if x in le.classes_ else -1)
    for col in X_train.select_dtypes(include='number').columns:
        median_val = X_train[col].median()
        X_train[col] = X_train[col].fillna(median_val)
        X_test[col]  = X_test[col].fillna(median_val)
    return X_train, X_test


# ────────────────────────────────────────────
# 모델 정의
# ────────────────────────────────────────────

class FTTransformer(nn.Module):
    def __init__(self, input_dim, d_model=128, n_heads=8, n_layers=3, dropout=0.1):
        super().__init__()
        self.tokenizer   = nn.Linear(1, d_model)
        encoder_layer    = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.classifier  = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.tokenizer(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.classifier(x).squeeze(1)


class SAINTLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.feat_attn    = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.feat_norm    = nn.LayerNorm(d_model)
        self.feat_ff      = nn.Sequential(nn.Linear(d_model, d_model*4), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model*4, d_model))
        self.feat_ff_norm = nn.LayerNorm(d_model)
        self.samp_attn    = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.samp_norm    = nn.LayerNorm(d_model)
        self.samp_ff      = nn.Sequential(nn.Linear(d_model, d_model*4), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model*4, d_model))
        self.samp_ff_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out, _ = self.feat_attn(x, x, x)
        x = self.feat_norm(x + attn_out)
        x = self.feat_ff_norm(x + self.feat_ff(x))
        x_t = x.permute(1, 0, 2)
        attn_out, _ = self.samp_attn(x_t, x_t, x_t)
        x_t = self.samp_norm(x_t + attn_out)
        x_t = self.samp_ff_norm(x_t + self.samp_ff(x_t))
        return x_t.permute(1, 0, 2)


class SAINT(nn.Module):
    def __init__(self, input_dim, d_model=128, n_heads=8, n_layers=3, dropout=0.1):
        super().__init__()
        self.tokenizer  = nn.Linear(1, d_model)
        self.layers     = nn.ModuleList([SAINTLayer(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.classifier = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.tokenizer(x)
        for layer in self.layers:
            x = layer(x)
        x = x.mean(dim=1)
        return self.classifier(x).squeeze(1)


class ResNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, n_layers=4, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            ) for _ in range(n_layers)
        ])
        self.classifier = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))

    def forward(self, x):
        x = self.input_proj(x)
        for layer in self.layers:
            x = x + layer(x)
        return self.classifier(x).squeeze(1)


class NODELayer(nn.Module):
    def __init__(self, input_dim, n_trees=128, tree_depth=6):
        super().__init__()
        self.n_trees    = n_trees
        self.tree_depth = tree_depth
        n_leaves = 2 ** tree_depth
        self.feat_selector  = nn.Linear(input_dim, n_trees * tree_depth)
        self.threshold      = nn.Parameter(torch.zeros(n_trees, tree_depth))
        self.leaf_responses = nn.Parameter(torch.randn(n_trees, n_leaves))

    def forward(self, x):
        B = x.size(0)
        feat = self.feat_selector(x).view(B, self.n_trees, self.tree_depth)
        feat = feat - self.threshold.unsqueeze(0)
        decisions = torch.sigmoid(feat * 10)
        leaf_probs = torch.ones(B, self.n_trees, 1, device=x.device)
        for d in range(self.tree_depth):
            left  = decisions[:, :, d:d+1]
            right = 1 - left
            leaf_probs = torch.cat([leaf_probs * left, leaf_probs * right], dim=2)
        out = (leaf_probs * self.leaf_responses.unsqueeze(0)).sum(dim=2)
        return out.mean(dim=1)


class NODE(nn.Module):
    def __init__(self, input_dim, n_trees=128, tree_depth=6, n_layers=2):
        super().__init__()
        self.layers     = nn.ModuleList([NODELayer(input_dim, n_trees, tree_depth) for _ in range(n_layers)])
        self.classifier = nn.Linear(1, 1)

    def forward(self, x):
        out = sum(layer(x) for layer in self.layers) / len(self.layers)
        return self.classifier(out.unsqueeze(1)).squeeze(1)


# ────────────────────────────────────────────
# 공통 학습 함수
# ────────────────────────────────────────────

def _train_dl_model(model, X_arr, Xt_arr, y_arr, name, seed,
                    lr, epochs, batch_size, patience, pos_weight, device, start_fold=1):
    skf        = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    oof        = np.zeros(len(X_arr))
    test_preds = np.zeros(len(Xt_arr))
    fold_roc_aucs = []
    fold_pr_aucs  = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_arr, y_arr), start=1):

        if fold < start_fold:
            saved_model = joblib.load(f'{MODEL_DIR}/{name}_fold{fold}.pkl')
            saved_model.to(device)
            saved_model.eval()
            with torch.no_grad():
                val_logits  = saved_model(torch.tensor(X_arr[val_idx]).to(device)).cpu().numpy()
                test_logits = saved_model(torch.tensor(Xt_arr).to(device)).cpu().numpy()
            val_pred     = 1 / (1 + np.exp(-val_logits))
            oof[val_idx] = val_pred
            fold_roc = roc_auc_score(y_arr[val_idx], val_pred)
            fold_pr  = average_precision_score(y_arr[val_idx], val_pred)
            fold_roc_aucs.append(fold_roc)
            fold_pr_aucs.append(fold_pr)
            test_preds += (1 / (1 + np.exp(-test_logits))) / N_SPLITS
            print(f'Fold {fold} | ROC-AUC: {fold_roc:.4f} | PR-AUC: {fold_pr:.4f} (이어서)')
            continue

        X_tr, X_val = X_arr[tr_idx], X_arr[val_idx]
        y_tr, y_val = y_arr[tr_idx], y_arr[val_idx]

        train_ds = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr))
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        pw        = torch.tensor([pos_weight]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

        best_val_roc = 0
        patience_cnt = 0
        best_state   = None

        for epoch in range(epochs):
            model.train()
            for xb, yb in train_dl:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_logits = model(torch.tensor(X_val).to(device)).cpu().numpy()
                val_pred   = 1 / (1 + np.exp(-val_logits))
                val_roc    = roc_auc_score(y_val, val_pred)

            if val_roc > best_val_roc:
                best_val_roc = val_roc
                best_state   = {k: v.clone() for k, v in model.state_dict().items()}
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= patience:
                    break

        model.load_state_dict(best_state)
        model.eval()

        with torch.no_grad():
            val_logits  = model(torch.tensor(X_val).to(device)).cpu().numpy()
            test_logits = model(torch.tensor(Xt_arr).to(device)).cpu().numpy()

        val_pred     = 1 / (1 + np.exp(-val_logits))
        oof[val_idx] = val_pred

        fold_roc = roc_auc_score(y_val, val_pred)
        fold_pr  = average_precision_score(y_val, val_pred)
        fold_roc_aucs.append(fold_roc)
        fold_pr_aucs.append(fold_pr)

        test_preds += (1 / (1 + np.exp(-test_logits))) / N_SPLITS

        joblib.dump(model, f'{MODEL_DIR}/{name}_fold{fold}.pkl')
        print(f'Fold {fold} | ROC-AUC: {fold_roc:.4f} | PR-AUC: {fold_pr:.4f}')

    mean_roc = np.mean(fold_roc_aucs)
    mean_pr  = np.mean(fold_pr_aucs)
    print(f'\nCV Mean | ROC-AUC: {mean_roc:.4f} | PR-AUC: {mean_pr:.4f}')

    np.save(f'{MODEL_DIR}/{name}_oof.npy',  oof)
    np.save(f'{MODEL_DIR}/{name}_test.npy', test_preds)

    return fold_roc_aucs, fold_pr_aucs, mean_roc, mean_pr


def _save_result(name, model_type, fold_roc_aucs, fold_pr_aucs, mean_roc, mean_pr, extra={}):
    result = {
        'name':         name,
        'model_type':   model_type,
        'cv_roc_auc':   [round(v, 4) for v in fold_roc_aucs],
        'mean_roc_auc': round(mean_roc, 4),
        'cv_pr_auc':    [round(v, 4) for v in fold_pr_aucs],
        'mean_pr_auc':  round(mean_pr, 4),
        **extra
    }
    with open(f'{MODEL_DIR}/{name}_result.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


# ────────────────────────────────────────────
# 실험 함수들
# ────────────────────────────────────────────

def run_experiment_fttransformer(
    name, use_features=True, d_model=128, n_heads=8, n_layers=3,
    dropout=0.1, lr=1e-4, epochs=100, batch_size=512, patience=10,
    pos_weight=3.0, seed=RANDOM_SEED, start_fold=1,
):
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_raw, test_raw = load_data()
    X, y, X_test = prepare(train_raw, test_raw, use_features=use_features)
    X, X_test    = encode_for_nn(X, X_test)

    scaler = StandardScaler()
    X_arr  = scaler.fit_transform(X).astype(np.float32)
    Xt_arr = scaler.transform(X_test).astype(np.float32)
    y_arr  = y.values.astype(np.float32)

    model = FTTransformer(X_arr.shape[1], d_model, n_heads, n_layers, dropout).to(device)
    fold_roc_aucs, fold_pr_aucs, mean_roc, mean_pr = _train_dl_model(
        model, X_arr, Xt_arr, y_arr, name, seed, lr, epochs, batch_size, patience, pos_weight, device, start_fold
    )
    return _save_result(name, 'fttransformer', fold_roc_aucs, fold_pr_aucs, mean_roc, mean_pr)


def run_experiment_saint(
    name, use_features=True, d_model=128, n_heads=8, n_layers=3,
    dropout=0.1, lr=1e-4, epochs=100, batch_size=512, patience=10,
    pos_weight=3.0, seed=RANDOM_SEED, start_fold=1,
):
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_raw, test_raw = load_data()
    X, y, X_test = prepare(train_raw, test_raw, use_features=use_features)
    X, X_test    = encode_for_nn(X, X_test)

    scaler = StandardScaler()
    X_arr  = scaler.fit_transform(X).astype(np.float32)
    Xt_arr = scaler.transform(X_test).astype(np.float32)
    y_arr  = y.values.astype(np.float32)

    model = SAINT(X_arr.shape[1], d_model, n_heads, n_layers, dropout).to(device)
    fold_roc_aucs, fold_pr_aucs, mean_roc, mean_pr = _train_dl_model(
        model, X_arr, Xt_arr, y_arr, name, seed, lr, epochs, batch_size, patience, pos_weight, device, start_fold
    )
    return _save_result(name, 'saint', fold_roc_aucs, fold_pr_aucs, mean_roc, mean_pr)


def run_experiment_resnet(
    name, use_features=True, hidden_dim=256, n_layers=4,
    dropout=0.1, lr=1e-3, epochs=100, batch_size=512, patience=10,
    pos_weight=3.0, seed=RANDOM_SEED, start_fold=1,
):
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_raw, test_raw = load_data()
    X, y, X_test = prepare(train_raw, test_raw, use_features=use_features)
    X, X_test    = encode_for_nn(X, X_test)

    scaler = StandardScaler()
    X_arr  = scaler.fit_transform(X).astype(np.float32)
    Xt_arr = scaler.transform(X_test).astype(np.float32)
    y_arr  = y.values.astype(np.float32)

    model = ResNet(X_arr.shape[1], hidden_dim, n_layers, dropout).to(device)
    fold_roc_aucs, fold_pr_aucs, mean_roc, mean_pr = _train_dl_model(
        model, X_arr, Xt_arr, y_arr, name, seed, lr, epochs, batch_size, patience, pos_weight, device, start_fold
    )
    return _save_result(name, 'resnet', fold_roc_aucs, fold_pr_aucs, mean_roc, mean_pr)


def run_experiment_node(
    name, use_features=True, n_trees=128, tree_depth=6, n_layers=2,
    lr=1e-3, epochs=100, batch_size=512, patience=10,
    pos_weight=3.0, seed=RANDOM_SEED, start_fold=1,
):
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_raw, test_raw = load_data()
    X, y, X_test = prepare(train_raw, test_raw, use_features=use_features)
    X, X_test    = encode_for_nn(X, X_test)

    scaler = StandardScaler()
    X_arr  = scaler.fit_transform(X).astype(np.float32)
    Xt_arr = scaler.transform(X_test).astype(np.float32)
    y_arr  = y.values.astype(np.float32)

    model = NODE(X_arr.shape[1], n_trees, tree_depth, n_layers).to(device)
    fold_roc_aucs, fold_pr_aucs, mean_roc, mean_pr = _train_dl_model(
        model, X_arr, Xt_arr, y_arr, name, seed, lr, epochs, batch_size, patience, pos_weight, device, start_fold
    )
    return _save_result(name, 'node', fold_roc_aucs, fold_pr_aucs, mean_roc, mean_pr)