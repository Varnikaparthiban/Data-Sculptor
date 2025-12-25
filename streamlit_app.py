import streamlit as st
import pandas as pd
import os

# Import your existing modular AI-driven functions
from utils.data_preprocessing import clean_data
from utils.data_insights import generate_insights
from utils.data_visualization import save_heatmap, save_trend_plot, save_distribution_plot
from utils.data_outliers import detect_outliers
from utils.data_quality import calculate_quality_score
from utils.data_ml import train_model

# --- Page Configuration ---
st.set_page_config(
    page_title="Data Sculptor",
    page_icon="🗿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ensure required folders exist for the visualization functions
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(BASE_DIR, 'static/images'), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'uploads'), exist_ok=True)

# --- Streamlit Cloud Caching Method ---
@st.cache_data(show_spinner=False)
def perform_ai_sculpting(df):
    """Orchestrates the modular utility functions with caching for performance."""
    df_cleaned, missing, missing_percent, duplicates, invalid_dates = clean_data(df)
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

# --- Custom CSS for "Shock" Factor ---
st.markdown("""
    <style>
    /* Main background and font */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    /* Professional Card Styling */
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        color: #1E3A8A;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.7);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.1);
        backdrop-filter: blur(4px);
    }
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #1E293B;
    }
    /* Success message styling */
    .stSuccess {
        border-left: 5px solid #10B981;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Header ---
st.title("🗿 DATA SCULPTOR")
st.markdown("**AI-Driven Data Cleaning & Analysis Tool**")
st.markdown("---")

# --- Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=100)
    st.header("Control Panel")
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    st.markdown("---")
    st.info("💡 **Pro Tip:** Upload datasets with missing values or outliers to see the AI's full potential.")
    
# --- Main Application Logic ---
if uploaded_file is not None:
    try:
        # Load Data
        df = pd.read_csv(uploaded_file)

        # --- Step 1: AI Processing ---
        with st.status("🤖 AI Sculptor at work...", expanded=True) as status:
            st.write("🔍 Analyzing data structures...")
            # Use the cached "Cloud Method" for processing
            results = perform_ai_sculpting(df)
            
            df_cleaned = results["df_cleaned"]
            quality_score = results["quality_score"]
            quality_grade = results["quality_grade"]
            duplicates = results["duplicates"]
            total_outliers = results["total_outliers"]
            insights = results["insights"]
            invalid_dates = results["invalid_dates"]
            missing = results["missing"]

            status.update(label="✅ Sculpting Complete!", state="complete", expanded=False)

        # --- Step 2: Dashboard Metrics ---
        st.subheader("🎯 Executive Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Quality Score", f"{quality_score}/100", delta=quality_grade)
        with col2:
            st.metric("Data Volume", f"{df_cleaned.shape[0]} rows", delta=f"-{duplicates} dups")
        with col3:
            st.metric("Dimensions", f"{df_cleaned.shape[1]} cols")
        with col4:
            st.metric("Anomalies", total_outliers, delta="Detected", delta_color="inverse")

        # --- Export Section (Moved to main area) ---
        st.markdown("### 🚀 Export Sculpted Data")
        csv = df_cleaned.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Cleaned CSV",
            data=csv,
            file_name=f"sculpted_{uploaded_file.name}",
            mime='text/csv',
            type='primary',
            use_container_width=False
        )

        # --- Step 3: Cleaned Data & Download ---
        st.markdown("---")
        tab_preview, tab_compare = st.tabs(["💾 Sculpted Dataset", "⚖️ Before vs After"])
        
        with tab_preview:
            st.dataframe(df_cleaned.head(20), use_container_width=True)
        
        with tab_compare:
            col_a, col_b = st.columns(2)
            col_a.markdown("#### 🔴 Original (Raw)")
            col_a.write(df.describe())
            col_b.markdown("#### 🟢 Sculpted (Clean)")
            col_b.write(df_cleaned.describe())

        # --- Step 4: Insights & Issues ---
        row2_1, row2_2 = st.columns(2)
        
        with row2_1:
            st.subheader("🧠 AI Intelligence Report")
            for insight in insights:
                st.info(f"✨ {insight}")

        with row2_2:
            st.subheader("🛠️ Transformation Log")
            st.json({
                "Missing Values Imputed": sum(missing.values()),
                "Duplicates Purged": duplicates,
                "Date Formats Standardized": invalid_dates,
                "Outliers Isolated": total_outliers
            })

        # --- Step 5: Visualizations ---
        st.markdown("---")
        st.subheader("🎨 Visualizations")
        
        tab1, tab2, tab3, tab4 = st.tabs(["🔥 Correlation Heatmap", "📈 Trends", "📊 Distributions", "🔮 AutoML Prediction"])
        
        with tab1:
            heatmap_file = save_heatmap(df_cleaned)
            if heatmap_file:
                img_path = os.path.join(BASE_DIR, 'static/images', heatmap_file)
                if os.path.exists(img_path):
                    st.image(img_path, caption="Feature Correlation Heatmap")
            else:
                st.info("Not enough numeric data for a heatmap.")

        numeric_cols = df_cleaned.select_dtypes(include=['number']).columns
        
        with tab2:
            if len(numeric_cols) > 0:
                col_trend = st.selectbox("Select Column for Trend", numeric_cols, key='trend')
                trend_file = save_trend_plot(df_cleaned, col_trend)
                img_path = os.path.join(BASE_DIR, 'static/images', trend_file)
                if os.path.exists(img_path):
                    st.image(img_path)
        
        with tab3:
            if len(numeric_cols) > 0:
                col_dist = st.selectbox("Select Column for Distribution", numeric_cols, key='dist')
                dist_file = save_distribution_plot(df_cleaned, col_dist)
                img_path = os.path.join(BASE_DIR, 'static/images', dist_file)
                if os.path.exists(img_path):
                    st.image(img_path)

        with tab4:
            st.write("### 🤖 Automated Machine Learning")
            st.info("Select a target column you want to predict. The AI will automatically choose the best model.")
            target_col = st.selectbox("Select Target Column", df_cleaned.columns)
            if st.button("Train AI Model"):
                with st.spinner("Training model..."):
                    task_type, score = train_model(df_cleaned, target_col)
                    st.success(f"Model Trained Successfully! (Task: {task_type})")
                    st.metric("Model Accuracy / Score", f"{score:.2%}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.write("👈 Please upload a CSV file from the sidebar to begin.")