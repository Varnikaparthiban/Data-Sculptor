import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

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

def generate_forecast(df, date_col, value_col, periods=10):
    """Simple linear trend forecasting for time-series data."""
    df_temp = df[[date_col, value_col]].copy()
    df_temp[date_col] = pd.to_datetime(df_temp[date_col])
    df_temp = df_temp.sort_values(date_col)
    
    # Convert dates to ordinal for regression
    X = np.array(range(len(df_temp))).reshape(-1, 1)
    y = df_temp[value_col].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict future
    future_X = np.array(range(len(df_temp), len(df_temp) + periods)).reshape(-1, 1)
    future_y = model.predict(future_X)
    
    last_date = df_temp[date_col].max()
    future_dates = pd.date_range(start=last_date, periods=periods + 1, freq='D')[1:]
    return pd.DataFrame({date_col: future_dates, value_col: future_y})