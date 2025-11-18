import pandas as pd
import xgboost as xgb
from pathlib import Path
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# --- Configuration (Root-Relative) ---
DATA_PROCESSED_DIR = Path("data_processed")
SUBMISSIONS_DIR = Path("submissions")

# --- 1. NEW VALIDATION FUNCTION ---
def run_validation(df_train: pd.DataFrame, features: list):
    """
    Splits the training data to train a temporary model and
    prints its RMSE score for performance evaluation.
    """
    print("\n--- Starting Model Validation ---")
    
    # Split data into training (80%) and validation (20%)
    # We use random_state=42 for reproducible results
    train_data, val_data = train_test_split(df_train, test_size=0.2, random_state=42)
    
    X_train_val = train_data[features]
    y_train_val = train_data['Weekly_Sales']
    
    X_val = val_data[features]
    y_val = val_data['Weekly_Sales']
    
    # Create and train a temporary XGBoost model
    print("  - Training temporary model on 80% of data...")
    temp_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        n_jobs=-1,
        random_state=42,
        early_stopping_rounds=10 # Stop if validation score doesn't improve
    )
    
    temp_model.fit(X_train_val, y_train_val,
                   eval_set=[(X_val, y_val)],
                   verbose=False)
    
    # Make predictions on the 20% validation set
    print("  - Making predictions on 20% validation data...")
    val_predictions = temp_model.predict(X_val)
    
    # Calculate and print the RMSE
    # RMSE = Root Mean Squared Error. Lower is better.
    # This shows, on average, how many $ off the predictions are.
    rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
    
    print(f"\n  ========================================")
    print(f"  🎉 MODEL VALIDATION RMSE: {rmse:.4f}")
    print(f"  ========================================\n")
    
    return rmse # Return the score if needed


# --- 2. MAIN TRAINING & PREDICTION FUNCTION ---
def train_and_predict():
    print("\nStarting model training script...")

    # Create submissions folder at the root
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

    # Load processed data
    try:
        df_train = pd.read_csv(DATA_PROCESSED_DIR / "train_processed.csv", parse_dates=["Date"])
        df_test = pd.read_csv(DATA_PROCESSED_DIR / "test_processed.csv", parse_dates=["Date"])
        print(f"  - Successfully loaded {DATA_PROCESSED_DIR / 'train_processed.csv'}")
        print(f"  - Successfully loaded {DATA_PROCESSED_DIR / 'test_processed.csv'}")
    except FileNotFoundError:
        print(f"❌ Error: Processed data not found in {DATA_PROCESSED_DIR}.")
        print("Please run pipeline.py first.")
        sys.exit(1) # Exit with an error code to fail the GitHub Action

    # --- Define Features (X) and Target (y) ---
    print("  - Defining features and target...")
    
    # Drop NAs created by lag/roll features *before* validation
    df_train_clean = df_train.dropna(subset=['Weekly_Sales_Lag_1', 'Weekly_Sales_4_Week_Avg'])

    # Define all feature columns
    features = [
        'Store', 'Dept', 'IsHoliday', 'Type', 'Size',
        'Temperature', 'Fuel_Price', 'CPI', 'Unemployment',
        'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5',
        'Year', 'Month', 'WeekOfYear', 'Day',
        'Weekly_Sales_Lag_1', 'Weekly_Sales_Lag_52', 'Weekly_Sales_4_Week_Avg'
    ]
    
    # Check that all features exist in both dataframes
    valid_features = []
    for col in features:
        if col not in df_train_clean.columns:
            print(f"  - Warning: Feature column '{col}' not found in train data. Skipping.")
        elif col not in df_test.columns:
            print(f"  - Warning: Feature column '{col}' not found in test data. Skipping.")
        else:
            valid_features.append(col)
    
    print(f"  - Using {len(valid_features)} features for training.")

    # --- Run Validation Step ---
    # This will train a temp model and print its RMSE score
    run_validation(df_train_clean, valid_features)
    
    # --- Train FINAL Model ---
    print("--- Starting Final Model Training ---")
    print("  - Training final model on 100% of clean data...")
    
    # Use the full, clean training dataset
    X_train_final = df_train_clean[valid_features]
    y_train_final = df_train_clean['Weekly_Sales']
    
    # Use the test set (which may have NAs in some columns)
    X_test_final = df_test[valid_features]

    # Create and train the final XGBoost model
    final_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        n_jobs=-1,
        random_state=42
    )
    
    final_model.fit(X_train_final, y_train_final)
    print("  - Final model training complete.")

    # --- Generate Predictions ---
    print("  - Generating final predictions...")
    predictions = final_model.predict(X_test_final)
    
    # Format for submission
    df_test['id'] = df_test['Store'].astype(str) + '_' + \
                    df_test['Dept'].astype(str) + '_' + \
                    df_test['Date'].dt.strftime('%Y-%m-%d')

    submission = pd.DataFrame({
        "id": df_test['id'],
        "Weekly_Sales": predictions
    })
    
    # Kaggle requires non-negative sales
    submission['Weekly_Sales'] = submission['Weekly_Sales'].clip(lower=0)
    print("  - Predictions generated.")

    # --- Save Submission File ---
    output_path = SUBMISSIONS_DIR / "submission.csv"
    submission.to_csv(output_path, index=False)
    
    print(f"✔ Model training complete. Submission file saved to {output_path}")

if __name__ == "__main__":
    train_and_predict()