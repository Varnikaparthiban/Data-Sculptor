import pandas as pd

def clean_data(df):
    # 1. Capture stats before cleaning
    missing_values = df.isnull().sum().to_dict()
    missing_percent = (df.isnull().mean() * 100).round(2).to_dict()
    duplicates = int(df.duplicated().sum())

    # 2. Actual Cleaning (Innovation: Auto-Correction)
    # Remove duplicates
    df = df.drop_duplicates()

    # Date conversion
    date_cols = [c for c in df.columns if 'date' in c.lower()]
    invalid_dates = 0
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')
        invalid_dates += int(df[col].isna().sum())

    # Impute missing values (Accuracy: Preserving data structure)
    # Numeric -> Median (Robust to outliers)
    for col in df.select_dtypes(include=['number']).columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    # Categorical -> Mode
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].isnull().any():
            if not df[col].mode().empty:
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna("Unknown")

    return df, missing_values, missing_percent, duplicates, invalid_dates
