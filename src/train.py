import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from category_encoders import TargetEncoder
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pytorch_tabnet.tab_model import TabNetClassifier

import sys
sys.path.append('..')
from src.preprocess import preprocess, TARGET_ENCODE_COLS
from src.features import add_features


N_SPLITS    = 10
RANDOM_SEED = 42
MODEL_DIR   = '../saved_models'
DATA_PATH   = '../data/raw/'

DEFAULT_PARAMS_LGBM = {
    'objective':             'binary',
    'metric':                'auc',
    'learning_rate':         0.05,
    'num_leaves':            127,
    'min_child_samples':     50,
    'subsample':             0.8,
    'colsample_bytree':      0.8,
    'n_estimators':          1000,
    'early_stopping_rounds': 50,
    'verbose':               -1,
    'random_state':          RANDOM_SEED,
}

DEFAULT_PARAMS_XGB = {
    'objective':             'binary:logistic',
    'eval_metric':           'auc',
    'learning_rate':         0.05,
    'max_depth':             6,
    'min_child_weight':      5,
    'subsample':             0.8,
    'colsample_bytree':      0.8,
    'n_estimators':          1000,
    'early_stopping_rounds': 50,
    'verbosity':             0,
    'random_state':          RANDOM_SEED,
}

DEFAULT_PARAMS_CAT = {
    'loss_function':         'Logloss',
    'eval_metric':           'AUC',
    'learning_rate':         0.05,
    'depth':                 6,
    'n_estimators':          1000,
    'early_stopping_rounds': 50,
    'verbose':               0,
    'random_seed':           RANDOM_SEED,
}

DEFAULT_PARAMS_RF = {
    'n_estimators': 500,
    'max_depth': None,
    'min_samples_leaf': 20,
    'max_features': 'sqrt',
    'class_weight': 'balanced',
    'n_jobs': 8,
    'random_state': RANDOM_SEED,
}

DEFAULT_PARAMS_ET = {
    'n_estimators': 500,
    'max_depth': None,
    'min_samples_leaf': 20,
    'max_features': 'sqrt',
    'class_weight': 'balanced',
    'n_jobs': 8,
    'random_state': RANDOM_SEED,
}

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


def encode_for_nn(X_train: pd.DataFrame, X_test: pd.DataFrame):
    X_train = X_train.copy()
    X_test  = X_test.copy()

    obj_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    for col in obj_cols:
        le = LabelEncoder()
        train_vals = X_train[col].fillna('__nan__').astype(str)
        test_vals  = X_test[col].fillna('__nan__').astype(str)
        le.fit(pd.concat([train_vals, test_vals], axis=0))
        X_train[col] = le.transform(train_vals)
        X_test[col]  = test_vals.map(
            lambda x, le=le: le.transform([x])[0] if x in le.classes_ else -1
        )

    for col in X_train.select_dtypes(include='number').columns:
        median_val = X_train[col].median()
        X_train[col] = X_train[col].fillna(median_val)
        X_test[col]  = X_test[col].fillna(median_val)

    return X_train, X_test

def run_experiment(
    name,
    use_features        = True,
    class_weight        = None,
    params              = None,
    model_type          = 'lgbm',
    use_target_encoding = False,
    seed                = RANDOM_SEED,
):
    os.makedirs(MODEL_DIR, exist_ok=True)

    if params is None:
        if model_type == 'lgbm':
            params = DEFAULT_PARAMS_LGBM.copy()
            params['random_state'] = seed
        elif model_type == 'xgb':
            params = DEFAULT_PARAMS_XGB.copy()
            params['random_state'] = seed
        elif model_type == 'catboost':
            params = DEFAULT_PARAMS_CAT.copy()
            params['random_seed'] = seed
        elif model_type == 'rf':
            params = DEFAULT_PARAMS_RF.copy()
            params['random_state'] = seed
        elif model_type == 'et':
            params = DEFAULT_PARAMS_ET.copy()
            params['random_state'] = seed
    else:
        params = params.copy()
        if model_type == 'catboost':
            params['random_seed'] = seed
        else:
            params['random_state'] = seed

    if class_weight == 'balanced':
        if model_type == 'lgbm':
            params['is_unbalance'] = True
        elif model_type == 'catboost':
            params['auto_class_weights'] = 'Balanced'

    train_raw, test_raw = load_data()
    X, y, X_test = prepare(train_raw, test_raw, use_features=use_features)

    skf        = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    oof        = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))

    fold_roc_aucs = []
    fold_pr_aucs  = []

    te_cols = [c for c in TARGET_ENCODE_COLS if c in X.columns]

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_tr,  X_val  = X.iloc[tr_idx].copy(),  X.iloc[val_idx].copy()
        y_tr,  y_val  = y.iloc[tr_idx],          y.iloc[val_idx]
        X_test_fold   = X_test.copy()

        if use_target_encoding and te_cols:
            te = TargetEncoder(cols=te_cols)
            X_tr        = te.fit_transform(X_tr, y_tr)
            X_val       = te.transform(X_val)
            X_test_fold = te.transform(X_test_fold)
        else:
            X_test_fold = X_test.copy()

        clean_params = {k: v for k, v in params.items() if k != 'model_type'}

        if model_type == 'lgbm':
            model = lgb.LGBMClassifier(**clean_params)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])

        elif model_type == 'xgb':
            import xgboost as xgb
            fit_params = {'verbose': False}
            if class_weight == 'balanced':
                from sklearn.utils.class_weight import compute_sample_weight
                fit_params['sample_weight'] = compute_sample_weight('balanced', y_tr)
            model = xgb.XGBClassifier(**clean_params)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], **fit_params)

        elif model_type == 'catboost':
            from catboost import CatBoostClassifier
            model = CatBoostClassifier(**clean_params)
            model.fit(X_tr, y_tr, eval_set=(X_val, y_val))

        elif model_type == 'rf':
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(**clean_params)
            model.fit(X_tr, y_tr)

        elif model_type == 'et':
            from sklearn.ensemble import ExtraTreesClassifier
            model = ExtraTreesClassifier(**clean_params)
            model.fit(X_tr, y_tr)

        val_pred     = model.predict_proba(X_val)[:, 1]
        oof[val_idx] = val_pred

        fold_roc = roc_auc_score(y_val, val_pred)
        fold_pr  = average_precision_score(y_val, val_pred)
        fold_roc_aucs.append(fold_roc)
        fold_pr_aucs.append(fold_pr)

        test_preds += model.predict_proba(X_test_fold)[:, 1] / N_SPLITS

        joblib.dump(model, f'{MODEL_DIR}/{name}_fold{fold}.pkl')

        print(f'Fold {fold} | ROC-AUC: {fold_roc:.4f} | PR-AUC: {fold_pr:.4f}')

    mean_roc = np.mean(fold_roc_aucs)
    mean_pr  = np.mean(fold_pr_aucs)

    print(f'\nCV Mean | ROC-AUC: {mean_roc:.4f} | PR-AUC: {mean_pr:.4f}')

    np.save(f'{MODEL_DIR}/{name}_oof.npy',  oof)
    np.save(f'{MODEL_DIR}/{name}_test.npy', test_preds)

    result = {
        'name':                name,
        'model_type':          model_type,
        'use_features':        use_features,
        'use_target_encoding': use_target_encoding,
        'class_weight':        class_weight,
        'seed':                seed,
        'cv_roc_auc':          [round(v, 4) for v in fold_roc_aucs],
        'mean_roc_auc':        round(mean_roc, 4),
        'cv_pr_auc':           [round(v, 4) for v in fold_pr_aucs],
        'mean_pr_auc':         round(mean_pr, 4),
    }

    with open(f'{MODEL_DIR}/{name}_result.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result


def tune_experiment(
    name                = 'tuned',
    use_features        = True,
    n_trials            = 50,
    model_type          = 'lgbm',
    use_target_encoding = False,
    class_weight        = None,
    seed                = RANDOM_SEED,
):
    os.makedirs(MODEL_DIR, exist_ok=True)

    train_raw, test_raw = load_data()
    X, y, _ = prepare(train_raw, test_raw, use_features=use_features)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    te_cols = [c for c in TARGET_ENCODE_COLS if c in X.columns]

    def objective(trial):
        if model_type == 'lgbm':
            params = {
                'objective':             'binary',
                'metric':                'auc',
                'verbose':               -1,
                'random_state':          seed,
                'learning_rate':         trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'num_leaves':            trial.suggest_int('num_leaves', 31, 255),
                'min_child_samples':     trial.suggest_int('min_child_samples', 20, 200),
                'subsample':             trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree':      trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha':             trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
                'reg_lambda':            trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
                'n_estimators':          1000,
                'early_stopping_rounds': 50,
            }
            if class_weight == 'balanced':
                params['is_unbalance'] = True
            model = lgb.LGBMClassifier(**params)

        elif model_type == 'xgb':
            import xgboost as xgb
            params = {
                'objective':             'binary:logistic',
                'eval_metric':           'auc',
                'verbosity':             0,
                'random_state':          seed,
                'learning_rate':         trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'max_depth':             trial.suggest_int('max_depth', 3, 10),
                'min_child_weight':      trial.suggest_int('min_child_weight', 1, 20),
                'subsample':             trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree':      trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha':             trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
                'reg_lambda':            trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
                'n_estimators':          1000,
                'early_stopping_rounds': 50,
            }
            model = xgb.XGBClassifier(**params)

        elif model_type == 'catboost':
            from catboost import CatBoostClassifier
            params = {
                'loss_function':         'Logloss',
                'eval_metric':           'AUC',
                'verbose':               0,
                'random_seed':           seed,
                'learning_rate':         trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'depth':                 trial.suggest_int('depth', 4, 10),
                'min_data_in_leaf':      trial.suggest_int('min_data_in_leaf', 20, 200),
                'subsample':             trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bylevel':     trial.suggest_float('colsample_bylevel', 0.5, 1.0),
                'reg_lambda':            trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
                'n_estimators':          1000,
                'early_stopping_rounds': 50,
            }
            if class_weight == 'balanced':
                params['auto_class_weights'] = 'Balanced'
            model = CatBoostClassifier(**params)

        fold_scores = []
        for tr_idx, val_idx in skf.split(X, y):
            X_tr,  X_val = X.iloc[tr_idx].copy(), X.iloc[val_idx].copy()
            y_tr,  y_val = y.iloc[tr_idx],         y.iloc[val_idx]

            if use_target_encoding and te_cols:
                te = TargetEncoder(cols=te_cols)
                X_tr  = te.fit_transform(X_tr, y_tr)
                X_val = te.transform(X_val)

            if model_type == 'xgb':
                fit_params = {'verbose': False}
                if class_weight == 'balanced':
                    from sklearn.utils.class_weight import compute_sample_weight
                    fit_params['sample_weight'] = compute_sample_weight('balanced', y_tr)
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], **fit_params)
            else:
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])

            val_pred = model.predict_proba(X_val)[:, 1]
            fold_scores.append(roc_auc_score(y_val, val_pred))

        return np.mean(fold_scores)

    sampler = TPESampler(seed=seed)
    study   = optuna.create_study(
        study_name     = f'{name}_seed{seed}',
        storage        = f'sqlite:///{MODEL_DIR}/optuna.db',
        load_if_exists = True,
        direction      = 'maximize',
        sampler        = sampler,
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_params.update({
        'model_type':            model_type,
        'n_estimators':          1000,
        'early_stopping_rounds': 50,
    })

    if model_type == 'lgbm':
        best_params.update({'objective': 'binary', 'metric': 'auc', 'verbose': -1, 'random_state': seed})
        if class_weight == 'balanced':
            best_params['is_unbalance'] = True
    elif model_type == 'xgb':
        best_params.update({'objective': 'binary:logistic', 'eval_metric': 'auc', 'verbosity': 0, 'random_state': seed})
    elif model_type == 'catboost':
        best_params.update({'loss_function': 'Logloss', 'eval_metric': 'AUC', 'verbose': 0, 'random_seed': seed})
        if class_weight == 'balanced':
            best_params['auto_class_weights'] = 'Balanced'

    print(f'\nBest ROC-AUC : {study.best_value:.4f}')
    print(f'Best Params  : {best_params}')

    with open(f'{MODEL_DIR}/{name}_seed{seed}_best_params.json', 'w', encoding='utf-8') as f:
        json.dump(best_params, f, ensure_ascii=False, indent=2)

    return best_params


def run_experiment_mlp(
    name,
    use_features = True,
    hidden_dims  = [256, 128, 64],
    dropout      = 0.3,
    lr           = 1e-3,
    epochs       = 100,
    batch_size   = 512,
    patience     = 10,
    seed         = RANDOM_SEED,
    start_fold   = 1,
    pos_weight   = 3.0,
):
    os.makedirs(MODEL_DIR, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_raw, test_raw = load_data()
    X, y, X_test = prepare(train_raw, test_raw, use_features=use_features)

    X, X_test = encode_for_nn(X, X_test)

    scaler  = StandardScaler()
    X_arr   = scaler.fit_transform(X).astype(np.float32)
    Xt_arr  = scaler.transform(X_test).astype(np.float32)
    y_arr   = y.values.astype(np.float32)

    skf        = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    oof        = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))

    fold_roc_aucs = []
    fold_pr_aucs  = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_arr, y_arr), start=1):
        if fold < start_fold:
            model = joblib.load(f'{MODEL_DIR}/{name}_fold{fold}.pkl')
            model.to(device)
            model.eval()
            with torch.no_grad():
                val_logits  = model(torch.tensor(X_arr[val_idx]).to(device)).cpu().numpy()
                test_logits = model(torch.tensor(Xt_arr).to(device)).cpu().numpy()
            val_pred     = 1 / (1 + np.exp(-val_logits))
            oof[val_idx] = val_pred
            fold_roc = roc_auc_score(y_arr[val_idx], val_pred)
            fold_pr  = average_precision_score(y_arr[val_idx], val_pred)
            fold_roc_aucs.append(fold_roc)
            fold_pr_aucs.append(fold_pr)
            test_preds += (1 / (1 + np.exp(-test_logits))) / N_SPLITS
            print(f'Fold {fold} | ROC-AUC: {fold_roc:.4f} | PR-AUC: {fold_pr:.4f} (이어서)')
            continue

        X_tr,  X_val = X_arr[tr_idx], X_arr[val_idx]
        y_tr,  y_val = y_arr[tr_idx], y_arr[val_idx]

        train_ds = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr))
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        model     = MLP(X_arr.shape[1], hidden_dims, dropout).to(device)
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

    result = {
        'name':         name,
        'model_type':   'mlp',
        'use_features': use_features,
        'hidden_dims':  hidden_dims,
        'dropout':      dropout,
        'seed':         seed,
        'cv_roc_auc':   [round(v, 4) for v in fold_roc_aucs],
        'mean_roc_auc': round(mean_roc, 4),
        'cv_pr_auc':    [round(v, 4) for v in fold_pr_aucs],
        'mean_pr_auc':  round(mean_pr, 4),
    }

    with open(f'{MODEL_DIR}/{name}_result.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers += [
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)
    
def run_experiment_tabnet(
    name,
    use_features = True,
    n_d          = 64,
    n_a          = 64,
    n_steps      = 5,
    gamma        = 1.5,
    lr           = 2e-2,
    batch_size   = 1024,
    max_epochs   = 200,
    patience     = 20,
    seed         = RANDOM_SEED,
):
    os.makedirs(MODEL_DIR, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_raw, test_raw = load_data()
    X, y, X_test = prepare(train_raw, test_raw, use_features=use_features)

    X, X_test = encode_for_nn(X, X_test)

    scaler = StandardScaler()
    X_arr  = scaler.fit_transform(X).astype(np.float32)
    Xt_arr = scaler.transform(X_test).astype(np.float32)
    y_arr  = y.values

    skf        = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    oof        = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))

    fold_roc_aucs = []
    fold_pr_aucs  = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_arr, y_arr), start=1):
        X_tr, X_val = X_arr[tr_idx], X_arr[val_idx]
        y_tr, y_val = y_arr[tr_idx], y_arr[val_idx]

        model = TabNetClassifier(
            n_d              = n_d,
            n_a              = n_a,
            n_steps          = n_steps,
            gamma            = gamma,
            optimizer_fn     = torch.optim.Adam,
            optimizer_params = {'lr': lr},
            scheduler_fn     = torch.optim.lr_scheduler.StepLR,
            scheduler_params = {'step_size': 50, 'gamma': 0.9},
            verbose          = 0,
            seed             = seed,
        )

        model.fit(
            X_tr, y_tr,
            eval_set           = [(X_val, y_val)],
            eval_metric        = ['auc'],
            max_epochs         = max_epochs,
            patience           = patience,
            batch_size         = batch_size,
            virtual_batch_size = 256,
            weights            = 1,
        )

        val_pred     = model.predict_proba(X_val)[:, 1]
        oof[val_idx] = val_pred

        fold_roc = roc_auc_score(y_val, val_pred)
        fold_pr  = average_precision_score(y_val, val_pred)
        fold_roc_aucs.append(fold_roc)
        fold_pr_aucs.append(fold_pr)

        test_preds += model.predict_proba(Xt_arr)[:, 1] / N_SPLITS

        joblib.dump(model, f'{MODEL_DIR}/{name}_fold{fold}.pkl')

        print(f'Fold {fold} | ROC-AUC: {fold_roc:.4f} | PR-AUC: {fold_pr:.4f}')

    mean_roc = np.mean(fold_roc_aucs)
    mean_pr  = np.mean(fold_pr_aucs)

    print(f'\nCV Mean | ROC-AUC: {mean_roc:.4f} | PR-AUC: {mean_pr:.4f}')

    np.save(f'{MODEL_DIR}/{name}_oof.npy',  oof)
    np.save(f'{MODEL_DIR}/{name}_test.npy', test_preds)

    result = {
        'name':         name,
        'model_type':   'tabnet',
        'use_features': use_features,
        'n_d':          n_d,
        'n_a':          n_a,
        'n_steps':      n_steps,
        'seed':         seed,
        'cv_roc_auc':   [round(v, 4) for v in fold_roc_aucs],
        'mean_roc_auc': round(mean_roc, 4),
        'cv_pr_auc':    [round(v, 4) for v in fold_pr_aucs],
        'mean_pr_auc':  round(mean_pr, 4),
    }

    with open(f'{MODEL_DIR}/{name}_result.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result