import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats

# Import your existing modular AI-driven functions
from utils.data_visualization import get_heatmap, get_trend_plot, get_distribution_plot, get_regression_plot, get_forecast_plot
from utils.ai_engine import perform_ai_sculpting
from utils.data_ml import train_model, generate_forecast
from sklearn.ensemble import IsolationForest

# --- Helper Functions ---
def nav_to(page_name):
    st.session_state["nav_page"] = page_name

# --- Page Configuration ---
st.set_page_config(
    page_title="Data Sculptor",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for "Shock" Factor ---
st.markdown("""
    <style>
    /* Main background and font */
    .main {
        background-color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }
    /* Professional Card Styling */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        color: #1E3A8A;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.7);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stSuccess {
        border-left: 5px solid #10B981;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Authentication Logic ---
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

def login_page():
    """Displays a centered login form."""
    st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(-45deg, #0f172a, #1e293b, #334155, #1e1b4b);
            background-size: 400% 400%;
            animation: gradient 15s ease infinite;
        }
        @keyframes gradient {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        @keyframes float {
            0% { transform: translateY(0px); }
            50% { transform: translateY(-15px); }
            100% { transform: translateY(0px); }
        }
        .floating-icon {
            animation: float 4s ease-in-out infinite;
            text-align: center;
        }
        .stForm {
            background: rgba(255, 255, 255, 0.05) !important;
            backdrop-filter: blur(16px) !important;
            -webkit-backdrop-filter: blur(16px) !important;
            border-radius: 24px !important;
            padding: 3rem !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5) !important;
        }
        .stTextInput > div > div > input {
            background-color: rgba(255, 255, 255, 0.05) !important;
            color: white !important;
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
            border-radius: 12px !important;
        }
        .stButton > button {
            background: linear-gradient(90deg, #10B981 0%, #3B82F6 100%) !important;
            color: white !important;
            border: none !important;
            font-weight: 700 !important;
            letter-spacing: 1px !important;
            border-radius: 12px !important;
            transition: all 0.3s ease !important;
            margin-top: 1rem !important;
        }
        .stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 10px 15px -3px rgba(16, 185, 129, 0.4) !important;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<br><br><br>", unsafe_allow_html=True)
    _, col, _ = st.columns([1, 1.2, 1])
    with col:
        st.markdown("<div class='floating-icon'><h1 style='font-size: 5rem; margin-bottom: 0;'>🧮</h1></div>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center; color: white; font-weight: 800; letter-spacing: -1px; margin-bottom: 2rem;'>DATA SCULPTOR</h2>", unsafe_allow_html=True)
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="")
            password = st.text_input("Password", type="password", placeholder="")
            submit = st.form_submit_button("UNLOCK WORKSPACE", use_container_width=True)
            
            if submit:
                if username == "admin" and password == "admin5":
                    st.session_state["authenticated"] = True
                    st.rerun()
                else:
                    st.error("🚫 Access Denied: Invalid Credentials")

if not st.session_state["authenticated"]:
    login_page()
    st.stop()

# --- Header ---
st.markdown("<h1 style='text-align: center; font-size: 3.5rem; font-weight: 800; color: #1E40AF; margin-bottom: 0;'>🧮 DATA SCULPTOR</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'><b>AI-Driven Data Cleaning & Analysis Tool</b></p>", unsafe_allow_html=True)
st.markdown("---")

# --- Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=100)
    st.header("Control Panel")
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    # Navigation Menu
    if "nav_page" not in st.session_state:
        st.session_state["nav_page"] = "Data Studio"

    page = st.radio("🚀 Navigation", ["Data Studio", "Visual Gallery", "Advanced Analytics", "AI Predictions"], key="nav_page")
    st.markdown("---")
    st.subheader("🛠️ Quick Actions")
    
    st.info("💡 **Pro Tip:** Upload datasets with missing values or outliers to see the AI's full potential.")
    
    if st.button("Logout", use_container_width=True, type="secondary"):
        st.session_state["authenticated"] = False
        st.rerun()
    
# --- Main Application Logic ---
if uploaded_file is not None:
    # Reset ML results if a new file is uploaded to prevent data leakage between sessions
    if "current_file" not in st.session_state or st.session_state["current_file"] != uploaded_file.name:
        st.session_state["current_file"] = uploaded_file.name
        if 'ml_results' in st.session_state:
            del st.session_state['ml_results']

    try:
        # Load Data
        df = pd.read_csv(uploaded_file)

        # --- Innovation: Workflow Navigator (Redirection) ---
        st.markdown("<br>", unsafe_allow_html=True)
        steps = ["Data Studio", "Visual Gallery", "Advanced Analytics", "AI Predictions"]
        current_idx = steps.index(page)
        
        cols = st.columns(len(steps))
        for i, step in enumerate(steps):
            is_active = i <= current_idx
            color = "#10B981" if is_active else "#E2E8F0"
            cols[i].markdown(f"<div style='text-align:center; font-weight:bold; color:{color};'>{step}</div>", unsafe_allow_html=True)
            cols[i].markdown(f"<div style='height:5px; background-color:{color}; border-radius:5px; margin-top:5px;'></div>", unsafe_allow_html=True)

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

        if page == "Data Studio":
            # --- Step 2: Dashboard Metrics ---
            st.subheader("🎯 Executive Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Quality Score", f"{quality_score}/100", delta=quality_grade)
                st.progress(quality_score / 100)
            with col2:
                st.metric("Data Volume", f"{df_cleaned.shape[0]} rows", delta=f"-{duplicates} dups")
            with col3:
                st.metric("Dimensions", f"{df_cleaned.shape[1]} cols")
            with col4:
                st.metric("Anomalies", total_outliers, delta="Detected", delta_color="inverse")

            # --- Export Section ---
            st.markdown("### 🚀 Export Sculpted Data")
            _, col_btn, _ = st.columns([1, 1, 1])
            with col_btn:
                csv = df_cleaned.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Cleaned CSV",
                    data=csv,
                    file_name=f"sculpted_{uploaded_file.name}",
                    mime='text/csv',
                    type='primary',
                    use_container_width=True
                )

            # --- Step 3: Cleaned Data Preview ---
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
                
                st.markdown("#### 🤖 Automated Recommendations")
                if quality_score > 80:
                    st.success("✅ **Action:** High data quality detected. The dataset is ready for advanced modeling and predictive analytics.")
                else:
                    st.info("ℹ️ **Action:** Fair data quality. We recommend reviewing the transformation logs and insights before proceeding to AI predictions.")
                
                if total_outliers > 0:
                    st.warning(f"⚠️ **Note:** {total_outliers} outliers were detected. Consider applying robust scaling if you plan to perform linear regression.")

            with row2_2:
                st.subheader("🛠️ Transformation Log")
                c1, c2 = st.columns(2)
                c1.write(f"🩹 **Imputed:** {sum(missing.values())} values")
                c1.write(f"♻️ **Purged:** {duplicates} duplicates")
                c2.write(f"📅 **Fixed:** {invalid_dates} dates")
                c2.write(f"🔍 **Isolated:** {total_outliers} outliers")

            # --- Innovation: Automation Decision Making ---
            st.markdown("### 🧠 Smart Decision Engine")
            with st.container(border=True):
                if df_cleaned.isnull().sum().sum() == 0:
                    st.success("✅ **Decision:** Data is fully imputed. Recommended for Machine Learning.")
                else:
                    st.error("❌ **Decision:** Missing values remain in the dataset. Further imputation is required for most ML tasks.")

                if total_outliers > (len(df_cleaned) * 0.05):
                    st.warning("⚠️ **Decision:** High anomaly rate detected (>5%). Consider using robust models or specialized outlier treatment.")
                if len(df_cleaned.columns) > 15:
                    st.info("💡 **Decision:** High dimensionality detected. Feature selection is advised.")
                
                # Innovation: Automated Decision for Multicollinearity
                numeric_cols_only = df_cleaned.select_dtypes(include=[np.number])
                if len(numeric_cols_only.columns) > 1:
                    corr_matrix = numeric_cols_only.corr().abs()
                    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                    stacked_corr = upper.stack()
                    if not stacked_corr.empty and any(stacked_corr > 0.9):
                        st.warning("⚠️ **Decision:** High multicollinearity (>0.9) detected. Consider removing redundant features to avoid overfitting.")

            st.markdown("---")
            c_back, c_next = st.columns(2)
            c_back.button("🔄 Reset Analysis", use_container_width=True, on_click=nav_to, args=("Data Studio",))
            c_next.button("Explore Visual Gallery ➡️", use_container_width=True, type="primary", on_click=nav_to, args=("Visual Gallery",))

        elif page == "Visual Gallery":
            # --- Step 5: Visualizations ---
            st.subheader("🎨 Visualizations")
            
            tab1, tab2, tab3 = st.tabs(["🔥 Correlation Heatmap", "📈 Trends", "📊 Distributions"])
            
            with tab1:
                fig_heatmap = get_heatmap(df_cleaned)
                if fig_heatmap:
                    st.plotly_chart(fig_heatmap, use_container_width=True)
                else:
                    st.info("Not enough numeric data for a heatmap.")

            numeric_cols = df_cleaned.select_dtypes(include=['number']).columns
            
            with tab2:
                if len(numeric_cols) > 0:
                    col_trend = st.selectbox("Select Column for Trend", numeric_cols, key='trend')
                    fig_trend = get_trend_plot(df_cleaned, col_trend)
                    st.plotly_chart(fig_trend, use_container_width=True)
            
            with tab3:
                if len(numeric_cols) > 0:
                    col_dist = st.selectbox("Select Column for Distribution", numeric_cols, key='dist')
                    fig_dist = get_distribution_plot(df_cleaned, col_dist)
                    st.plotly_chart(fig_dist, use_container_width=True)

            st.markdown("---")
            c_back, c_next = st.columns(2)
            c_back.button("⬅️ Back to Data Studio", use_container_width=True, on_click=nav_to, args=("Data Studio",))
            c_next.button("Run Advanced Analytics ➡️", use_container_width=True, type="primary", on_click=nav_to, args=("Advanced Analytics",))

        elif page == "Advanced Analytics":
            st.subheader("🧪 Advanced Statistical Analysis")
            
            # Innovation: Statistical Deep Dive (Mean/Mode/Median)
            with st.container(border=True):
                st.markdown("#### 📊 Column-Specific Deep Dive")
                selected_col = st.selectbox("Select Column to Analyze", df_cleaned.columns)
                s_col1, s_col2, s_col3, s_col4 = st.columns(4)
                
                mode_val = df_cleaned[selected_col].mode()
                s_col1.metric("Mode", str(mode_val[0]) if not mode_val.empty else "N/A")

                if pd.api.types.is_numeric_dtype(df_cleaned[selected_col]):
                    s_col2.metric("Mean", f"{df_cleaned[selected_col].mean():.2f}")
                    s_col3.metric("Median", f"{df_cleaned[selected_col].median():.2f}")
                    s_col4.metric("Std Dev", f"{df_cleaned[selected_col].std():.2f}")
                    st.write(f"**Variance:** {df_cleaned[selected_col].var():.2f} | **Skewness:** {df_cleaned[selected_col].skew():.2f}")
                else:
                    s_col2.write("**Categorical Data**")

            # Anomaly Detection
            st.markdown("---")
            st.subheader("🚨 Anomaly Detection")
            numeric_df = df_cleaned.select_dtypes(include=['number'])
            if not numeric_df.empty:
                iso = IsolationForest(contamination=0.05, random_state=42)
                preds = iso.fit_predict(numeric_df.fillna(0))
                anomalies = df_cleaned[preds == -1]
                
                if not anomalies.empty:
                    st.warning(f"AI Isolation Forest detected {len(anomalies)} complex anomalies.")
                    # Innovation: Anomaly Visualization
                    if len(numeric_df.columns) >= 2:
                        fig_anom = px.scatter(df_cleaned, x=numeric_df.columns[0], y=numeric_df.columns[1], color=preds.astype(str), title="Anomaly Map (Outliers in Gold)")
                        st.plotly_chart(fig_anom, use_container_width=True)
                    st.dataframe(anomalies, use_container_width=True)
                else:
                    st.success("No multi-dimensional anomalies detected.")
            else:
                st.success("No significant anomalies detected in the current view.")

            # Regression Analysis Preview
            st.markdown("---")
            st.subheader("📉 Regression Analysis")
            num_cols = df_cleaned.select_dtypes(include=['number']).columns.tolist()
            if len(num_cols) >= 2:
                reg_x = st.selectbox("Independent Variable (X)", num_cols, index=0)
                reg_y = st.selectbox("Dependent Variable (Y)", num_cols, index=1)
                
                # Innovation: Detailed Regression Statistics
                slope, intercept, r_value, p_value, std_err = stats.linregress(df_cleaned[reg_x], df_cleaned[reg_y])
                
                col_reg1, col_reg2 = st.columns(2)
                with col_reg1:
                    st.info(f"**Pearson Correlation:** {r_value:.4f}")
                    st.write(f"**Equation:** y = {slope:.4f}x + {intercept:.4f}")
                with col_reg2:
                    st.write(f"**R-squared:** {r_value**2:.4f}")
                    st.write(f"**P-value:** {p_value:.4f}")
                
                fig_reg = get_regression_plot(df_cleaned, reg_x, reg_y)
                st.plotly_chart(fig_reg, use_container_width=True)

                if abs(r_value) > 0.7:
                    st.success("Strong linear relationship detected. Ideal for regression.")
                else:
                    st.warning("Weak linear relationship. Consider non-linear models.")

            st.markdown("---")
            c_back, c_next = st.columns(2)
            c_back.button("⬅️ Back to Visual Gallery", use_container_width=True, on_click=nav_to, args=("Visual Gallery",))
            c_next.button("Unlock AI Predictions ➡️", use_container_width=True, type="primary", on_click=nav_to, args=("AI Predictions",))

        elif page == "AI Predictions":
            st.subheader("🔮 AI Prediction Hub")
            
            tab_ml, tab_forecast = st.tabs(["🤖 AutoML Prediction", "📈 Predictive Forecasting"])
            
            with tab_forecast:
                st.write("### ⏳ Time-Series Forecasting")
                # Innovation: Robust Date Detection
                date_cols = []
                for col in df_cleaned.columns:
                    try:
                        pd.to_datetime(df_cleaned[col].head(5), errors='raise')
                        date_cols.append(col)
                    except:
                        continue
                
                val_cols = df_cleaned.select_dtypes(include=['number']).columns
                
                if len(date_cols) > 0 and len(val_cols) > 0:
                    c_date = st.selectbox("Select Date Column", date_cols)
                    c_val = st.selectbox("Select Value to Forecast", val_cols)
                    horizon = st.slider("Forecast Horizon (Days)", 5, 30, 10)
                    
                    if st.button("Generate AI Forecast"):
                        with st.spinner("Calculating trends..."):
                            f_df = generate_forecast(df_cleaned, c_date, c_val, periods=horizon)
                            fig_f = get_forecast_plot(df_cleaned, c_date, c_val, f_df)
                            st.plotly_chart(fig_f, use_container_width=True)
                            st.write("#### 📋 Forecasted Values")
                            st.dataframe(f_df, use_container_width=True)
                else:
                    st.error("Forecasting requires at least one valid Date column and one numeric column.")

            with tab_ml:
                st.write("### 🤖 Automated Machine Learning")
                st.info("Select a target column you want to predict. The AI will automatically choose the best model.")
                target_col = st.selectbox("Select Target Column", df_cleaned.columns)
                if st.button("Train AI Model"):
                    with st.spinner("Training model..."):
                        results = train_model(df_cleaned, target_col)
                        st.session_state['ml_results'] = results
                
                # Persist results using session_state
                if 'ml_results' in st.session_state:
                    task_type, score = st.session_state['ml_results']
                    st.success(f"Model Trained Successfully! (Task: {task_type})")
                    st.metric("Model Accuracy / Score", f"{score:.2%}")
                    st.info("💡 Note: Changing the target column will require a new training session.")

            st.markdown("---")
            c_back, c_next = st.columns(2)
            c_back.button("⬅️ Back to Advanced Analytics", use_container_width=True, on_click=nav_to, args=("Advanced Analytics",))
            c_next.button("Start New Analysis ➡️", use_container_width=True, type="primary", on_click=nav_to, args=("Data Studio",))

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.write("👈 Please upload a CSV file from the sidebar to begin.")