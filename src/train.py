import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from category_encoders import TargetEncoder
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler

import sys
sys.path.append('..')
from src.preprocess import preprocess, TARGET_ENCODE_COLS
from src.features import add_features


N_SPLITS    = 5
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

DROP_COLS = ['ID', '시술 당시 나이', '임신 성공 여부']


def load_data():
    train = pd.read_csv(DATA_PATH + 'train.csv')
    test  = pd.read_csv(DATA_PATH + 'test.csv')
    return train, test


def prepare(train, test, use_features=True):
    train = preprocess(train)
    test  = preprocess(test)

    if use_features:
        train = add_features(train)
        test  = add_features(test)

    y      = train['임신 성공 여부']
    X      = train.drop(columns=[c for c in DROP_COLS if c in train.columns])
    X_test = test.drop(columns=[c for c in DROP_COLS if c in test.columns])

    return X, y, X_test


def run_experiment(
    name,
    use_features        = True,
    class_weight        = None,
    params              = None,
    model_type          = 'lgbm',
    use_target_encoding = False,
):
    os.makedirs(MODEL_DIR, exist_ok=True)

    if params is None:
        if model_type == 'lgbm':
            params = DEFAULT_PARAMS_LGBM.copy()
        elif model_type == 'xgb':
            params = DEFAULT_PARAMS_XGB.copy()
        elif model_type == 'catboost':
            params = DEFAULT_PARAMS_CAT.copy()

    if class_weight == 'balanced':
        if model_type == 'lgbm':
            params['is_unbalance'] = True
        elif model_type == 'catboost':
            params['auto_class_weights'] = 'Balanced'

    train_raw, test_raw = load_data()
    X, y, X_test = prepare(train_raw, test_raw, use_features=use_features)

    skf        = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
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
):
    os.makedirs(MODEL_DIR, exist_ok=True)

    train_raw, test_raw = load_data()
    X, y, _ = prepare(train_raw, test_raw, use_features=use_features)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    te_cols = [c for c in TARGET_ENCODE_COLS if c in X.columns]

    def objective(trial):
        if model_type == 'lgbm':
            params = {
                'objective':             'binary',
                'metric':                'auc',
                'verbose':               -1,
                'random_state':          RANDOM_SEED,
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
            model = lgb.LGBMClassifier(**params)

        elif model_type == 'xgb':
            import xgboost as xgb
            params = {
                'objective':             'binary:logistic',
                'eval_metric':           'auc',
                'verbosity':             0,
                'random_state':          RANDOM_SEED,
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
                'random_seed':           RANDOM_SEED,
                'learning_rate':         trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'depth':                 trial.suggest_int('depth', 4, 10),
                'min_data_in_leaf':      trial.suggest_int('min_data_in_leaf', 20, 200),
                'subsample':             trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bylevel':     trial.suggest_float('colsample_bylevel', 0.5, 1.0),
                'reg_lambda':            trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
                'n_estimators':          1000,
                'early_stopping_rounds': 50,
            }
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
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            else:
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])

            val_pred = model.predict_proba(X_val)[:, 1]
            fold_scores.append(roc_auc_score(y_val, val_pred))

        return np.mean(fold_scores)

    sampler = TPESampler(seed=RANDOM_SEED)
    study   = optuna.create_study(
        study_name     = name,
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
        best_params.update({'objective': 'binary', 'metric': 'auc', 'verbose': -1, 'random_state': RANDOM_SEED})
    elif model_type == 'xgb':
        best_params.update({'objective': 'binary:logistic', 'eval_metric': 'auc', 'verbosity': 0, 'random_state': RANDOM_SEED})
    elif model_type == 'catboost':
        best_params.update({'loss_function': 'Logloss', 'eval_metric': 'AUC', 'verbose': 0, 'random_seed': RANDOM_SEED})

    print(f'\nBest ROC-AUC : {study.best_value:.4f}')
    print(f'Best Params  : {best_params}')

    with open(f'{MODEL_DIR}/{name}_best_params.json', 'w', encoding='utf-8') as f:
        json.dump(best_params, f, ensure_ascii=False, indent=2)

    return best_params