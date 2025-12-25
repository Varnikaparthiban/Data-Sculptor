# c:\Users\admin\OneDrive\Desktop\data_sculptor_web\utils\data_quality.py
import pandas as pd

def calculate_quality_score(df):
    """
    Calculate a data quality score (0-100) based on missing values, duplicates, and dataset size.
    Returns: (score, grade)
    """
    score = 100
    
    # 1. Completeness (Missing Values)
    total_cells = df.size
    missing_cells = df.isnull().sum().sum()
    missing_ratio = missing_cells / total_cells if total_cells > 0 else 0
    score -= (missing_ratio * 100) * 1.5  # Penalize missing data heavily
    
    # 2. Uniqueness (Duplicates)
    total_rows = len(df)
    duplicates = df.duplicated().sum()
    duplicate_ratio = duplicates / total_rows if total_rows > 0 else 0
    score -= (duplicate_ratio * 100) * 1.0 # Penalize duplicates
    
    # 3. Volume Bonus (Small datasets are harder to trust for ML)
    if total_rows < 50:
        score -= 10
    
    # Cap score between 0 and 100
    score = max(0, min(100, score))
    
    # Determine Grade
    if score >= 90: grade = "A (Excellent)"
    elif score >= 80: grade = "B (Good)"
    elif score >= 70: grade = "C (Fair)"
    elif score >= 60: grade = "D (Poor)"
    else: grade = "F (Critical)"
    
    return round(score, 1), grade
