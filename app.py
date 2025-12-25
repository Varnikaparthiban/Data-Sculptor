from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
import pandas as pd
from werkzeug.utils import secure_filename
import os

# Import modular AI-driven preprocessing and analysis functions
from utils.data_preprocessing import clean_data
from utils.data_insights import generate_insights
from utils.data_visualization import save_heatmap, save_trend_plot, save_distribution_plot
from utils.data_outliers import detect_outliers
from utils.data_quality import calculate_quality_score


# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['IMAGE_FOLDER'] = 'static/images'

# Ensure required folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['IMAGE_FOLDER'], exist_ok=True)


@app.route('/')
def index():
    """Render homepage for Datasculptor"""
    return render_template('index.html', title="DATA SCULPTOR – AI Driven Tool")


@app.route('/analyze', methods=['POST'])
def analyze():
    """Process uploaded file and analyze using AI-driven modules"""
    if 'file' not in request.files or request.files['file'].filename == '':
        return "⚠️ Please upload a valid dataset file.", 400

    file = request.files['file']
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        df = pd.read_csv(filepath)
    except Exception:
        return "Error reading CSV. Ensure your file contains valid data.", 500

    # Preprocessing & Cleaning
    df, missing, missing_percent, duplicates, invalid_dates = clean_data(df)

    # Save cleaned data for download
    cleaned_filename = f"cleaned_{filename}"
    df.to_csv(os.path.join(app.config['UPLOAD_FOLDER'], cleaned_filename), index=False)

    # AI Insights
    insights = generate_insights(df)

    # Outliers
    total_outliers, outlier_counts = detect_outliers(df)

    # Innovation: Data Quality Score
    quality_score, quality_grade = calculate_quality_score(df)

    # Visualization
    heatmap_file = save_heatmap(df)
    trend_plots = {}
    dist_plots = {}
    for col in df.select_dtypes(include=['number']).columns:
        trend_plots[col] = save_trend_plot(df, col)
        dist_plots[col] = save_distribution_plot(df, col)

    # API Response Support (Innovation: Headless capability)
    if request.args.get('format') == 'json':
        return jsonify({
            "filename": filename,
            "shape": df.shape,
            "duplicates": int(duplicates) if isinstance(duplicates, (int, float)) else duplicates,
            "total_outliers": int(total_outliers) if isinstance(total_outliers, (int, float)) else total_outliers,
            "quality_score": quality_score,
            "quality_grade": quality_grade,
            "insights": insights,
            "download_url": url_for('download_file', filename=cleaned_filename, _external=True)
        })

    return render_template(
        'analyze.html',
        title="Analysis Results – DATA SCULPTOR",
        shape=df.shape,
        columns=df.columns.tolist(),
        missing=missing,
        missing_percent=missing_percent,
        duplicates=duplicates,
        invalid_dates=invalid_dates,
        insights=insights,
        total_outliers=total_outliers,
        outlier_counts=outlier_counts,
        heatmap_file=heatmap_file,
        trend_plots=trend_plots,
        dist_plots=dist_plots,
        quality_score=quality_score,
        quality_grade=quality_grade,
        cleaned_filename=cleaned_filename
    )

@app.route('/download/<filename>')
def download_file(filename):
    """Serve the cleaned file to the user."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    # Use the PORT environment variable if available (required for cloud hosting)
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)