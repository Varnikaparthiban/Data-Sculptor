import numpy as np
import pandas as pd

def detect_outliers(df):
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    outlier_counts = {}
    total_outliers = 0
    if numeric_cols:
        # Accuracy Upgrade: Use IQR (Interquartile Range) instead of Z-Score
        # IQR is robust against skewed distributions, whereas Z-Score assumes normality.
        Q1 = df[numeric_cols].quantile(0.25)
        Q3 = df[numeric_cols].quantile(0.75)
        IQR = Q3 - Q1

        # Define outliers as points outside 1.5 * IQR
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        mask = (df[numeric_cols] < lower_bound) | (df[numeric_cols] > upper_bound)
        
        total_outliers = int(mask.any(axis=1).sum())
        for i, col in enumerate(numeric_cols):
            outlier_counts[col] = int(mask[col].sum())
    return total_outliers, outlier_counts
