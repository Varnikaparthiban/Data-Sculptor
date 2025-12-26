import streamlit as st
from utils.data_preprocessing import clean_data
from utils.data_insights import generate_insights
from utils.data_outliers import detect_outliers
from utils.data_quality import calculate_quality_score

@st.cache_data(show_spinner=False)
def perform_ai_sculpting(df):
    """Orchestrates the modular utility functions with caching for performance."""
    # Preprocessing & Cleaning
    df_cleaned, missing, missing_percent, duplicates, invalid_dates = clean_data(df)
    
    # AI Insights & Quality Scoring
    quality_score, quality_grade = calculate_quality_score(df_cleaned)
    total_outliers, outlier_counts = detect_outliers(df_cleaned)
    insights = generate_insights(df_cleaned)
    
    return {
        "df_cleaned": df_cleaned,
        "missing": missing,
        "duplicates": duplicates,
        "invalid_dates": invalid_dates,
        "quality_score": quality_score,
        "quality_grade": quality_grade,
        "total_outliers": total_outliers,
        "insights": insights
    }