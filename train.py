"""
Train a LightGBM regression model on the tabular dataset.

Usage:
    python train.py --train_path train.csv --model_path model.txt

Outputs:
    - Saved LightGBM model file
    - Prints cross-validation RMSE
"""

import argparse
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


def load_data(path):
    """Load dataset and split into features and target."""
    df = pd.read_csv(path)
    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y


def train_model(X, y, params=None, n_folds=5):
    """
    Train LightGBM with cross-validation to estimate performance,
    then train final model on full data.

    Returns:
        model: trained LightGBM Booster
        rmse_scores: list of per-fold RMSE scores
    """
    if params is None:
        params = {
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": 0.05,
            "num_leaves": 63,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
        }

    # Cross-validation to estimate RMSE
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    rmse_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        train_set = lgb.Dataset(X_train, label=y_train)
        val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

        model = lgb.train(
            params,
            train_set,
            num_boost_round=1000,
            valid_sets=[val_set],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )

        preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        rmse_scores.append(rmse)
        print(f"Fold {fold + 1} RMSE: {rmse:.4f}")

    print(f"\nMean CV RMSE: {np.mean(rmse_scores):.4f} (+/- {np.std(rmse_scores):.4f})")

    # Train final model on full data
    full_train_set = lgb.Dataset(X, label=y)
    final_model = lgb.train(
        params,
        full_train_set,
        num_boost_round=1000,
    )

    return final_model, rmse_scores


def main():
    parser = argparse.ArgumentParser(description="Train regression model")
    parser.add_argument("--train_path", type=str, default="train.csv", help="Path to training CSV")
    parser.add_argument("--model_path", type=str, default="model.txt", help="Path to save the model")
    args = parser.parse_args()

    print("Loading data...")
    X, y = load_data(args.train_path)
    print(f"Dataset shape: {X.shape}")

    print("\nTraining with cross-validation...")
    model, scores = train_model(X, y)

    model.save_model(args.model_path)
    print(f"\nModel saved to {args.model_path}")


if __name__ == "__main__":
    main()
