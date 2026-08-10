# GNN vs XGBoost — Fraud Detection

Baseline experiments for detecting credit-card fraud on the [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) dataset, using classical ML baselines (Random Forest, XGBoost). A graph-neural-network baseline (GNN) is intended as the comparison point.

## Dataset

Downloaded automatically via [`kagglehub`](https://github.com/Kaggle/kagglehub) (`kagglehub.competition_download('ieee-fraud-detection')`).

- `train_transaction`: 590,540 rows × 393 cols
- `train_identity`: 144,233 rows × 40 cols
- Joined (left join on `TransactionID`): 590,540 rows × 433 cols
- **Fraud rate: 3.50%** (20,663 fraud vs 569,877 legitimate) — ~27.6:1 imbalance

## Notebooks

| Notebook | Contents |
|---|---|
| [`data.ipynb`](data.ipynb) | Dataset download, loading, and EDA: class balance, transaction-amount distribution (log scale), fraud rate by card type, top email domains (`P_emaildomain`, `R_emaildomain`) |
| [`random_forest.ipynb`](random_forest.ipynb) | Random Forest baseline: preprocessing, training, evaluation, feature importances |
| [`xgboost.ipynb`](xgboost.ipynb) | XGBoost baseline: preprocessing, training, evaluation, feature importances |

## Feature preprocessing (shared across models)

- Subset of 49 features: `TransactionAmt`, `ProductCD`, card/address fields, email domains, and the `C*`, `D*`, `M*` masked feature groups
- 12 categorical columns label-encoded (`LabelEncoder`, NaN → `"MISSING"`)
- Numeric columns median-imputed (train median carried to test)
- Train/test split: 80/20 stratified by `isFraud` (random_state 42)

## Results (held-out 20%, 118,108 transactions)

| Model | PR-AUC | ROC-AUC | Fraud precision | Fraud recall | Fraud F1 | Train time | Inference time |
|---|---|---|---|---|---|---|---|
| **XGBoost** | **0.7582** | **0.9608** | 0.39 | **0.83** | **0.53** | 9.6 s | 0.167 s |
| Random Forest | 0.5645 | 0.9115 | 0.21 | 0.77 | 0.32 | 50.2 s | 1.728 s (50k rows) |

XGBoost (`max_depth=8`, `learning_rate=0.1`, `subsample=0.8`, `colsample_bytree=0.8`, `scale_pos_weight≈27.6`, 300 boosting rounds) clearly outperforms the Random Forest (`n_estimators=200`, `max_depth=12`, `min_samples_leaf=50`, `class_weight='balanced_subsample'`) on every metric and is roughly 5× faster to train.

## Dependencies

Python 3.12, with `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `xgboost`, and `kagglehub` (managed via the `.venv` virtual environment).
