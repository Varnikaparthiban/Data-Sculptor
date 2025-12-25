import numpy as np

def generate_insights(df):
    insights = []
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if numeric_cols:
        for col in numeric_cols:
            insights.append(f"{col} -> Max: {df[col].max()}, Min: {df[col].min()}, Mean: {df[col].mean():.2f}")
        if len(numeric_cols) > 1:
            corr = df[numeric_cols].corr().abs()
            top_pair = corr.unstack().sort_values(ascending=False)
            for (a,b), val in top_pair.items():
                if a != b:
                    insights.append(f"Strongest correlation: {a} ↔ {b} = {val:.2f}")
                    break
    
    if categorical_cols:
        for col in categorical_cols:
            most_freq = df[col].mode()[0] if not df[col].mode().empty else "N/A"
            insights.append(f"{col} -> Most frequent: {most_freq}")
    
    return insights
