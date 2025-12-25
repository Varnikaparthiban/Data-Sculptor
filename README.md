
# DATA-SCULPTOR-AI-Driven-Preprocessing-and-Insight-Extraction-Tool
## 1. Introduction

DATA SCULPTOR is an intelligent, end-to-end AI-driven data preprocessing and insight extraction web application developed using Flask. The tool is designed to automate critical data preparation tasks and provide meaningful insights without requiring extensive programming knowledge from the user. By simplifying the preprocessing pipeline, DATA SCULPTOR enables users to focus on decision-making and analytical interpretation rather than manual data handling.
The application integrates data cleaning, outlier detection, statistical insight generation, and visual analytics into a unified workflow. Its modular and scalable architecture ensures that the system is not limited to exploratory analysis but is also ready for future expansion into machine learning, predictive analytics, and intelligent decision support systems.
Built using Python, Pandas, NumPy, Matplotlib, and Flask, DATA SCULPTOR emphasizes accuracy, usability, and scalability, making it suitable for academic projects, data analytics coursework, research experimentation, and business intelligence applications.

## 2. Statement of the Problem

### Raw datasets often contain:

Most real-world datasets suffer from several quality issues such as:

Missing or null values

Duplicate records

Outliers and anomalies

Inconsistent data formats

Invalid or corrupted entries

These issues significantly affect the reliability of statistical analysis and machine learning models. Manual preprocessing requires technical expertise, significant time investment, and careful validation, which may not be feasible for non-technical users or time-sensitive projects.

There is a need for an automated, user-friendly, and reliable system that can preprocess datasets, extract insights, detect anomalies, and visualize trends efficiently.

## 3. Purpose of the Project

The primary purpose of DATA SCULPTOR is to:

Automate essential data preprocessing tasks

Reduce manual effort in data cleaning

Provide quick statistical insights into datasets

Identify outliers that impact data quality

Present results through intuitive visualizations

Support non-technical users in data analysis

The system aims to bridge the gap between raw data and actionable intelligence.

## 4. Features Overview
### Home Page
The home page provides a clean and intuitive interface where users can:

Upload CSV datasets

Initiate the preprocessing and analysis pipeline

Navigate seamlessly to insight and visualization outputs

The design prioritizes clarity, simplicity, and ease of use, ensuring accessibility for users with minimal technical background.

## Automated Data Preprocessing
This module automatically prepares the uploaded dataset for analysis.

Key Capabilities

Handling missing and null values

Removing duplicate records

Standardizing column formats

Identifying invalid date entries

The preprocessing pipeline ensures that the dataset is clean, structured, consistent, and analysis-ready, forming a reliable foundation for further analytics.


## AI-Driven Insight Extraction

After preprocessing, the system extracts meaningful statistical insights.

Insights Generated

Dataset dimensions (rows and columns)

Summary statistics such as mean, median, minimum, and maximum

Column-wise numeric distributions

Correlation indicators among numerical features

These insights are presented in a human-readable format, allowing users to understand the dataset’s behavior quickly and accurately.
## Outlier Detection System


Identifies anomalous values that may affect data quality.

Techniques Used

Statistical thresholding

Interquartile Range (IQR) method

Numeric feature scanning

## Data Visualization Engine

Automatically generates visual analytics.

Visual Outputs

Correlation heatmaps

Time-based trend plots

Numeric feature trends

## 5. AI-Ready Architecture

DATA SCULPTOR is designed for scalability.

Modular utility files (utils/)

Independent preprocessing and analysis functions

Ready for ML model integration:

Classification

Regression

Clustering
# Installation & Setup:
1️⃣ Clone the Repository
```
git clone https://github.com/ABINAYA-27-76/DATA-SCULPTOR.git
cd DATA-SCULPTOR

```
2️⃣ Create a Virtual Environment
```
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows
```
3️⃣ Install Dependencies

Ensure you have file containing:

flask
pandas
numpy
matplotlib
seaborn
#### Install with:
```
pip install -r requirements.txt
```
4️⃣ Run the Application
```
python app.py
```
# program:
```
app py

from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from werkzeug.utils import secure_filename
import os
import sys

# Extend sys path to access custom utils
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

# Import modular AI-driven preprocessing and analysis functions
from utils.data_preprocessing import clean_data
from utils.data_insights import generate_insights
from utils.data_visualization import save_heatmap, save_trend_plot
from utils.data_outliers import detect_outliers


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

    # AI Insights
    insights = generate_insights(df)

    # Outliers
    total_outliers, outlier_counts = detect_outliers(df)

    # Visualization
    heatmap_file = save_heatmap(df)
    trend_plots = {}
    for col in df.select_dtypes(include=['number']).columns:
        trend_plots[col] = save_trend_plot(df, col)

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
    )

if __name__ == '__main__':
    app.run(debug=True)
```
## UI Highlights

Clean, minimal web-based interface

Simple dataset upload workflow

Automated analytics without coding

Visual outputs rendered instantly

Modular backend for easy upgrades
## Interface Preview:

### dataset information:
![WhatsApp Image 2025-12-23 at 9 41 41 PM](https://github.com/user-attachments/assets/d8799cd1-865e-4311-8a3a-aa08a1375181)
### AI-driven insigths:
![WhatsApp Image 2025-12-23 at 9 43 05 PM](https://github.com/user-attachments/assets/8de3d62e-1203-4985-a730-f7d4de1fc649)
### Correlation heatmap:
![WhatsApp Image 2025-12-23 at 9 43 12 PM](https://github.com/user-attachments/assets/f3b41be6-90fb-4aa5-a618-91d3c255d2a9)
### Trend plots:
![WhatsApp Image 2025-12-23 at 9 43 15 PM](https://github.com/user-attachments/assets/370f00a4-ee55-4884-a73a-82d3747e62d2)


## Future Enhancements:

 Machine Learning model integration

 Predictive analytics dashboards

 Auto feature engineering

 Cloud deployment (AWS / IBM Cloud)

 Downloadable insight reports (PDF)
 ## References:
 
 Software engineering / modelling: “Object-modeling technique (OMT)” —
Wikipedia: https://en.wikipedia.org/wiki/Object-modeling_technique (Wikipedia)


●Victor K.F., Michael I.Z., “Intelligent data analysis and machine learning: Are they really equivalent concepts?,” in the 2017 Second Russia and Pacific Conference on Computer Technology and Applications (RPC), 2017, pp. 59-
63. https://www.proceedings.com/content/037/037218webtoc.pdf (Proceedings)


●Uppala Sai Sudeep, Kandra Narasimha Naidu, Pulagam Sai Girish, Tatineni Naga Nikesh, Ch Sunanda. “Brain Tumor Classification using a Support Vector Machine.” International Journal of Computer Applications, vol. 184, no. 28 (Sep 2022), pp. 15-17. DOI:10.5120/ijca2022922347. Available at: https://ijcaonline.org/archives/volume184/number28/32492-2022922347/ (ijcaonline.org)



