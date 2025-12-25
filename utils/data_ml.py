import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

def train_model(df, target_col):
    """
    Automatically detects the task type (Classification/Regression),
    preprocesses data, and trains a Random Forest model.
    Returns: (task_type, score)
    """
    # 1. Prepare Data
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Handle categorical features in X
    X = pd.get_dummies(X, drop_first=True)

    # 2. Determine Task Type
    # If target is numeric and has many unique values, assume Regression
    # Otherwise, assume Classification
    if pd.api.types.is_numeric_dtype(y) and y.nunique() > 10:
        task_type = "Regression"
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        task_type = "Classification"
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Encode target if it's categorical
        if not pd.api.types.is_numeric_dtype(y):
            le = LabelEncoder()
            y = le.fit_transform(y.astype(str))

    # 3. Split Data
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 4. Train
        model.fit(X_train, y_train)

        # 5. Score
        # Returns Accuracy for Classification, R^2 for Regression
        score = model.score(X_test, y_test)
        
        return task_type, score

    except Exception as e:
        # Fallback for very small datasets or errors
        print(f"ML Training Error: {e}")
        return task_type, 0.0