# Tabular Regression

## Task
Build a regression model for 53 anonymized features. Target metric: RMSE.

## Solution
LightGBM gradient boosting regressor with 5-fold cross-validation for model selection.

**CV RMSE: 0.164** — LightGBM handles heterogeneous feature types (int/float) natively and performs well on tabular data without heavy feature engineering.

## Project Structure
- `eda.ipynb` — exploratory data analysis
- `train.py` — model training script
- `predict.py` — inference script
- `predictions.csv` — predictions for hidden_test.csv
- `requirements.txt` — dependencies

## Setup
```bash
pip install -r requirements.txt
```

## Usage
```bash
# Train model
python train.py --train_path train.csv --model_path model.txt

# Generate predictions
python predict.py --test_path hidden_test.csv --model_path model.txt --output_path predictions.csv
```
