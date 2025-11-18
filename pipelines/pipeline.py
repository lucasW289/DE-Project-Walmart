"""
ETL Pipeline for Walmart Sales Dataset (Production-Ready)
---------------------------------------------------------
Updates:
- Generates Data Engineering Validation Plots
- Saves plots to 'submissions/' for GitHub Artifacts
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# ====================================================
# 1. CONFIGURATION
# ====================================================

DATA_RAW_DIR = Path("data_raw")
DATA_PROCESSED_DIR = Path("data_processed")
SUBMISSIONS_DIR = Path("submissions") # Plots will go here

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
    try:
        stores = pd.read_csv(paths["stores"])
        features = pd.read_csv(paths["features"], parse_dates=["Date"])
        train = pd.read_csv(paths["train"], parse_dates=["Date"])
        test = pd.read_csv(paths["test"], parse_dates=["Date"])
    except FileNotFoundError as e:
        print(f"❌ Error: {e}.")
        sys.exit(1)

    train["dataset"] = "train"
    test["dataset"] = "test"
    df_combined = pd.concat([train, test], ignore_index=True)

    return df_combined, features, stores


# ====================================================
# 3. TRANSFORM
# ====================================================

def transform_data(df, features, stores):
    print("STEP 2/3: Transforming data...")
    
    # Merge
    df = df.merge(stores, on="Store", how="left").merge(features, on=["Store", "Date", "IsHoliday"], how="left")

    # Clean
    markdown_cols = [f"MarkDown{i}" for i in range(1, 6)]
    df[markdown_cols] = df[markdown_cols].fillna(0)
    df = df.sort_values(by=["Store", "Date"])
    df["CPI"] = df.groupby("Store")["CPI"].transform(lambda x: x.interpolate())
    df["Unemployment"] = df.groupby("Store")["Unemployment"].transform(lambda x: x.interpolate())

    # Feature Engineering
    print("  - Engineering features...")
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week
    df["Day"] = df["Date"].dt.day
    df["IsHoliday"] = df["IsHoliday"].astype(int)
    df["Type"] = df["Type"].map({"A": 1, "B": 2, "C": 3})

    # Time Series Features
    df = df.sort_values(by=["Store", "Dept", "Date"])
    grouped = df.groupby(["Store", "Dept"])
    df["Weekly_Sales_Lag_1"] = grouped["Weekly_Sales"].shift(1)
    df["Weekly_Sales_Lag_52"] = grouped["Weekly_Sales"].shift(52)
    df["Weekly_Sales_4_Week_Avg"] = grouped["Weekly_Sales"].shift(1).rolling(4).mean()

    return df

# ====================================================
# 4. VISUALIZATION (NEW)
# ====================================================
def generate_pipeline_plots(df):
    """
    Generates insights about the processed data.
    Insight 1: Correlation Matrix (Do our new features actually matter?)
    """
    print("  - Generating Pipeline Data Insights...")
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Filter only numeric columns for correlation
    numeric_df = df.select_dtypes(include=['float64', 'int64', 'int32'])
    
    # Focus on the correlation with Weekly_Sales
    if 'Weekly_Sales' in numeric_df.columns:
        plt.figure(figsize=(12, 10))
        correlation = numeric_df.corr()
        
        # Plot heatmap
        sns.heatmap(correlation[['Weekly_Sales']].sort_values(by='Weekly_Sales', ascending=False), 
                    annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        
        plt.title('Feature Correlation with Weekly Sales (Data Engineering Check)')
        plt.tight_layout()
        
        plot_path = SUBMISSIONS_DIR / 'data_correlation_heatmap.png'
        plt.savefig(plot_path)
        print(f"  ✔ Saved Data Engineering Insight: {plot_path}")
        plt.close()

# ====================================================
# 5. LOAD
# ====================================================

def load_data(df, out_train, out_test):
    print("STEP 3/3: Loading processed data...")
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df_train = df[df["dataset"] == "train"].drop(columns=["dataset"])
    df_test = df[df["dataset"] == "test"].drop(columns=["dataset"])

    # Generate plots on the training data before saving
    generate_pipeline_plots(df_train)

    df_train.to_csv(out_train, index=False)
    df_test.to_csv(out_test, index=False)
    print(f"  ✔ Saved: {out_train}")
    print(f"  ✔ Saved: {out_test}\n")


def run_pipeline():
    print("\n🚀 Running Walmart ETL Pipeline...\n")
    try:
        df_combined, features, stores = extract_data(CONFIG["paths"])
        df_processed = transform_data(df_combined, features, stores)
        load_data(df_processed, CONFIG["paths"]["output_train"], CONFIG["paths"]["output_test"])
        print("🎉 ETL Pipeline Completed Successfully!\n")
    except Exception as e:
        print("\n❌ ETL Pipeline FAILED")
        print("Error:", e)
        sys.exit(1)

if __name__ == "__main__":
    run_pipeline()