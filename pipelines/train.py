import pandas as pd
import xgboost as xgb
from pathlib import Path
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # <--- Using Seaborn for better plots

# --- Configuration ---
DATA_PROCESSED_DIR = Path("data_processed")
SUBMISSIONS_DIR = Path("submissions")

def run_validation(df_train: pd.DataFrame, features: list):
    print("\n--- Starting Model Validation ---")
    
    train_data, val_data = train_test_split(df_train, test_size=0.2, random_state=42)
    X_train_val = train_data[features]
    y_train_val = train_data['Weekly_Sales']
    X_val = val_data[features]
    y_val = val_data['Weekly_Sales']
    
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5, n_jobs=-1, random_state=42)
    model.fit(X_train_val, y_train_val)
    val_predictions = model.predict(X_val)
    
    rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
    print(f"\n  🎉 MODEL VALIDATION RMSE: {rmse:.4f}\n")

    # --- INSIGHT 1: Residual Plot (Are errors random or biased?) ---
    # If the blob is centered on 0, we are good. If it's skewed, we have bias.
    residuals = y_val - val_predictions
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True, bins=50, color='purple')
    plt.title(f'Residual Distribution (RMSE: {rmse:.0f})\nCheck for Normal Distribution (Bell Curve)')
    plt.xlabel('Prediction Error ($)')
    plt.axvline(x=0, color='r', linestyle='--')
    plt.tight_layout()
    plt.savefig(SUBMISSIONS_DIR / 'model_residuals.png')
    plt.close()
    
    # --- INSIGHT 2: Actual vs Predicted (Accuracy Check) ---
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_val, y=val_predictions, alpha=0.3)
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
    plt.xlabel('Actual Sales')
    plt.ylabel('Predicted Sales')
    plt.title('Actual vs Predicted Sales')
    plt.tight_layout()
    plt.savefig(SUBMISSIONS_DIR / 'actual_vs_predicted.png')
    plt.close()
    
    return rmse

def train_and_predict():
    print("\nStarting model training script...")
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        df_train = pd.read_csv(DATA_PROCESSED_DIR / "train_processed.csv", parse_dates=["Date"])
        df_test = pd.read_csv(DATA_PROCESSED_DIR / "test_processed.csv", parse_dates=["Date"])
    except FileNotFoundError:
        print(f"❌ Error: Data not found.")
        sys.exit(1)

    df_train_clean = df_train.dropna(subset=['Weekly_Sales_Lag_1', 'Weekly_Sales_4_Week_Avg'])

    features = [
        'Store', 'Dept', 'IsHoliday', 'Type', 'Size',
        'Temperature', 'Fuel_Price', 'CPI', 'Unemployment',
        'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5',
        'Year', 'Month', 'WeekOfYear', 'Day',
        'Weekly_Sales_Lag_1', 'Weekly_Sales_Lag_52', 'Weekly_Sales_4_Week_Avg'
    ]
    
    # Validate
    valid_features = [col for col in features if col in df_train_clean.columns]
    run_validation(df_train_clean, valid_features)
    
    # Final Train
    print("--- Final Training ---")
    final_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5, n_jobs=-1, random_state=42)
    final_model.fit(df_train_clean[valid_features], df_train_clean['Weekly_Sales'])
    
    # --- INSIGHT 3: Feature Importance (Did DE work?) ---
    # This proves if your Lags/Rolling Avgs are actually being used by the model.
    plt.figure(figsize=(12, 8))
    xgb.plot_importance(final_model, max_num_features=20, height=0.5, importance_type='weight')
    plt.title('Feature Importance (Validation of Feature Engineering)')
    plt.tight_layout()
    plt.savefig(SUBMISSIONS_DIR / 'feature_importance.png')
    plt.close()

    # Predict
    predictions = final_model.predict(df_test[valid_features])
    df_test['id'] = df_test['Store'].astype(str) + '_' + df_test['Dept'].astype(str) + '_' + df_test['Date'].dt.strftime('%Y-%m-%d')
    submission = pd.DataFrame({"id": df_test['id'], "Weekly_Sales": predictions})
    submission['Weekly_Sales'] = submission['Weekly_Sales'].clip(lower=0)

    submission.to_csv(SUBMISSIONS_DIR / "submission.csv", index=False)
    print(f"✔ Saved submission and insight plots to {SUBMISSIONS_DIR}")

if __name__ == "__main__":
    train_and_predict()