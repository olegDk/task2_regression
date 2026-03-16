"""
Generate predictions using a trained LightGBM model.

Usage:
    python predict.py --test_path hidden_test.csv --model_path model.txt --output_path predictions.csv

Outputs:
    - CSV file with predictions
"""

import argparse
import pandas as pd
import lightgbm as lgb


def main():
    parser = argparse.ArgumentParser(description="Generate predictions")
    parser.add_argument("--test_path", type=str, default="hidden_test.csv", help="Path to test CSV")
    parser.add_argument("--model_path", type=str, default="model.txt", help="Path to trained model")
    parser.add_argument("--output_path", type=str, default="predictions.csv", help="Path to save predictions")
    args = parser.parse_args()

    print("Loading model...")
    model = lgb.Booster(model_file=args.model_path)

    print("Loading test data...")
    X_test = pd.read_csv(args.test_path)
    print(f"Test set shape: {X_test.shape}")

    print("Generating predictions...")
    predictions = model.predict(X_test)

    result = pd.DataFrame({"target": predictions})
    result.to_csv(args.output_path, index=False)
    print(f"Predictions saved to {args.output_path} ({len(predictions)} rows)")


if __name__ == "__main__":
    main()
