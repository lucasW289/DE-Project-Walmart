"""
ETL Pipeline for Walmart Sales Dataset (Production-Ready)
---------------------------------------------------------

This script performs a full ETL process:
1. Extract   - Read raw CSV files (train, test, stores, features).
2. Transform - Combine datasets, clean data, and engineer 
               time-series features (lags, rolling avgs).
3. Load      - Save the processed, model-ready 'train' and 'test'
               DataFrames to the /data_processed folder.

This version FIXES the rolling/lag error by removing invalid index reset.
"""

import pandas as pd
from pathlib import Path

# ====================================================
# 1. CONFIGURATION
# ====================================================

BASE_DIR = Path("..")
DATA_RAW_DIR = BASE_DIR / "data_raw"
DATA_PROCESSED_DIR = BASE_DIR / "data_processed"  # <-- fixed typo

CONFIG = {
    "paths": {
        "train": DATA_RAW_DIR / "train.csv",
        "test": DATA_RAW_DIR / "test.csv",
        "features": DATA_RAW_DIR / "features.csv",
        "stores": DATA_RAW_DIR / "stores.csv",
        "output_train": DATA_PROCESSED_DIR / "train_processed.csv",
        "output_test": DATA_PROCESSED_DIR / "test_processed.csv",
    }
}

# ====================================================
# 2. EXTRACT
# ====================================================

def extract_data(paths: dict) -> tuple:
    print("STEP 1/3: Extracting raw CSV files...")

    stores = pd.read_csv(paths["stores"])
    features = pd.read_csv(paths["features"], parse_dates=["Date"])
    train = pd.read_csv(paths["train"], parse_dates=["Date"])
    test = pd.read_csv(paths["test"], parse_dates=["Date"])

    train["dataset"] = "train"
    test["dataset"] = "test"

    df_combined = pd.concat([train, test], ignore_index=True)

    return df_combined, features, stores


# ====================================================
# 3. TRANSFORM
# ====================================================

def transform_data(df, features, stores):
    print("STEP 2/3: Transforming data...")
    print("  - Merging datasets...")

    df = (
        df.merge(stores, on="Store", how="left")
          .merge(features, on=["Store", "Date", "IsHoliday"], how="left")
    )

    print("  - Cleaning missing values...")

    markdown_cols = [f"MarkDown{i}" for i in range(1, 6)]
    df[markdown_cols] = df[markdown_cols].fillna(0)

    df = df.sort_values(by=["Store", "Date"])
    df["CPI"] = df.groupby("Store")["CPI"].transform(lambda x: x.interpolate())
    df["Unemployment"] = df.groupby("Store")["Unemployment"].transform(lambda x: x.interpolate())

    print("  - Feature engineering basic date features...")

    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week
    df["Day"] = df["Date"].dt.day
    df["IsHoliday"] = df["IsHoliday"].astype(int)
    df["Type"] = df["Type"].map({"A": 1, "B": 2, "C": 3})

    print("  - Engineering time-series features (lags + rolling)...")

    df = df.sort_values(by=["Store", "Dept", "Date"])
    grouped = df.groupby(["Store", "Dept"])

    df["Weekly_Sales_Lag_1"] = grouped["Weekly_Sales"].shift(1)
    df["Weekly_Sales_Lag_52"] = grouped["Weekly_Sales"].shift(52)

    # Fixed rolling feature — removed invalid reset_index()
    df["Weekly_Sales_4_Week_Avg"] = (
        grouped["Weekly_Sales"]
        .shift(1)
        .rolling(4)
        .mean()
    )

    print("  ✔ Transformation complete.\n")
    return df


# ====================================================
# 4. LOAD
# ====================================================

def load_data(df, out_train, out_test):
    print("STEP 3/3: Loading processed data...")

    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df_train = df[df["dataset"] == "train"].drop(columns=["dataset"])
    df_test = df[df["dataset"] == "test"].drop(columns=["dataset"])

    df_train.to_csv(out_train, index=False)
    df_test.to_csv(out_test, index=False)

    print(f"  ✔ Saved: {out_train}")
    print(f"  ✔ Saved: {out_test}\n")


# ====================================================
# 5. MAIN
# ====================================================

def run_pipeline():
    print("\n🚀 Running Walmart ETL Pipeline...\n")

    try:
        df_combined, features, stores = extract_data(CONFIG["paths"])
        df_processed = transform_data(df_combined, features, stores)
        load_data(
            df_processed,
            CONFIG["paths"]["output_train"],
            CONFIG["paths"]["output_test"],
        )

        print("🎉 ETL Pipeline Completed Successfully!\n")

    except Exception as e:
        print("\n❌ ETL Pipeline FAILED")
        print("Error:", e)


if __name__ == "__main__":
    run_pipeline()
