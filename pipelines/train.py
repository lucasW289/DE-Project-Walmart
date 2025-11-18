import pandas as pd
import xgboost as xgb
from pathlib import Path
import sys

# --- Configuration (Root-Relative) ---
DATA_PROCESSED_DIR = Path("data_processed")
SUBMISSIONS_DIR = Path("submissions")

# --- Main Training Function ---
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
    
    # Drop NAs created by lag/roll features from the training data
    df_train = df_train.dropna(subset=['Weekly_Sales_Lag_1', 'Weekly_Sales_4_Week_Avg'])

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
        if col not in df_train.columns:
            print(f"  - Warning: Feature column '{col}' not found in train data. Skipping.")
        elif col not in df_test.columns:
            print(f"  - Warning: Feature column '{col}' not found in test data. Skipping.")
        else:
            valid_features.append(col)
    
    print(f"  - Using {len(valid_features)} features for training.")

    X_train = df_train[valid_features]
    y_train = df_train['Weekly_Sales']
    
    X_test = df_test[valid_features]

    # --- Train XGBoost Model ---
    print("  - Training XGBoost model...")
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        n_jobs=-1,  # Use all cores
        random_state=42
    )
    
    model.fit(X_train, y_train)
    print("  - Model training complete.")

    # --- Generate Predictions ---
    print("  - Generating predictions...")
    predictions = model.predict(X_test)
    
    # Format for submission
    # The 'id' is a combination of Store_Dept_Date
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