# DATA SCULPTOR-AI DRIVEN PREPROCESSING AND INSIGHT EXTRACTION TOOL
#  BY: DHIVYA DHARSHINI B,VARNIKA P, ABINAYA A
## INTRODUCTION :

In today’s data-driven world, organizations and individuals generate large volumes of data from various sources. However, raw data is often incomplete, inconsistent, and unstructured, making it difficult to analyze directly. 
Data preprocessing plays a crucial role in transforming raw data into a clean and usable format, but it is usually a time-consuming and complex task that requires technical expertise.


Data Sculptor – AI Driven Preprocessing and Insight Extraction Tool is designed to address these challenges by providing an automated and intelligent solution for data cleaning, preprocessing, and analysis.
The system leverages artificial intelligence and machine learning techniques to handle missing values, duplicates, outliers, and inconsistencies while automatically extracting meaningful insights such as trends, correlations, and feature importance. 
By offering a user-friendly, web-based interface with visual outputs, Data Sculptor simplifies the data analytics process and makes it accessible to both technical and non-technical users.
The tool aims to improve efficiency, accuracy, and decision-making by bridging the gap between raw data and actionable intelligence.

## FEATURES OVERVIEW :
1) Automated Data Acquisition
Supports importing datasets from multiple sources such as CSV files, Excel sheets, databases, and APIs.

2) Intelligent Data Preprocessing
Automatically handles missing values, removes duplicates, detects outliers, and fixes inconsistencies to improve data quality.

3) Data Transformation & Feature Engineering
Performs normalization, encoding, scaling, and feature selection to prepare data for accurate analysis.

4) AI-Driven Insight Extraction
Applies machine learning techniques to identify trends, correlations, regression patterns, and feature importance.

5) Recommendation System
Suggests optimal preprocessing and transformation strategies based on dataset characteristics.

6) Interactive Data Visualization
Presents insights through easy-to-understand charts, graphs, and dashboards for better interpretation.

7) User-Friendly Web Interface
Simple and intuitive UI that allows users to upload datasets, trigger analysis, and view results without coding.

8) Real-Time Processing Support
Enables fast and efficient analysis for moderately sized datasets.

9) Secure Data Handling
Incorporates data anonymization and access control to protect sensitive information.

10) Scalable & Modular Architecture
Designed for easy expansion and seamless integration with machine learning pipelines.

## PROJECT STRUCTURE :

<img width="732" height="551" alt="image" src="https://github.com/user-attachments/assets/594869ee-f916-4243-ba7c-c8bfd556900d" />


## INSTALLATION AND SETUP :

### Clone or Download the Repository

```bash
git clone https://github.com/dhivyadharshini2006/Data-Sculptor
cd Data-Sculptor
```

---

### Create a Virtual Environment (Recommended)

```bash
python -m venv venv
```

Activate the virtual environment:

* **Windows**

```bash
venv\Scripts\activate
```

* **macOS / Linux**

```bash
source venv/bin/activate
```

---

### Install Dependencies

Ensure a `requirements.txt` file is present with the following libraries:

```
flask
pandas
numpy
scikit-learn
matplotlib
seaborn
```

### Install all dependencies using:

```bash
pip install -r requirements.txt
```

---

### Run the Application

```bash
python app.py
```

Once the server starts, open your browser and navigate to:

```
http://127.0.0.1:5000/
```

## PROGRAM :
```python

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

## CORE CALCULATIONS :

| **Process**            | **Technique Used**                 | **Purpose**                                         |
| ---------------------- | ---------------------------------- | --------------------------------------------------- |
| Missing Value Handling | Mean / Median / Mode Imputation    | Fills missing data to maintain dataset completeness |
| Duplicate Removal      | Row Similarity Check               | Eliminates redundant records to reduce bias         |
| Outlier Detection      | IQR / Z-Score Method               | Identifies and handles extreme values               |
| Data Normalization     | Min–Max Scaling / Standardization  | Brings numerical features to a common scale         |
| Categorical Encoding   | Label Encoding / One-Hot Encoding  | Converts categorical data into numerical form       |
| Correlation Analysis   | Pearson Correlation Coefficient    | Identifies relationships between variables          |
| Regression Analysis    | Linear Regression                  | Analyzes dependency between variables               |
| Feature Importance     | ML-based Importance Scores         | Determines influential features                     |
| Trend Detection        | Time-Series / Statistical Analysis | Identifies patterns and trends in data              |
| Data Visualization     | Charts and Graphs                  | Presents insights in an interpretable format        |

## OUTPUT :
### UPLOADING DATASET :

<img width="1600" height="908" alt="image" src="https://github.com/user-attachments/assets/ce439db4-ecb5-43f6-a2d0-be5ff6345538" />

### HEATMAP :

<img width="942" height="827" alt="image" src="https://github.com/user-attachments/assets/352db3ce-fd79-47e3-ad01-76a38f110e5d" />

### TRENDS :

<img width="611" height="670" alt="image" src="https://github.com/user-attachments/assets/d7a7ab43-009e-4b89-a555-f71ad6d53cec" />

### DISTRIBUTIONS :


<img width="613" height="614" alt="image" src="https://github.com/user-attachments/assets/c560aa8d-e5ba-4bfe-95be-b6d7facba805" />


## IMPROVEMENTS OVER EXISISTING WEBSITES :

| **Aspect**         | **Existing Websites**  | **Data Sculptor**                   |
| ------------------ | ---------------------- | ----------------------------------- |
| Ease of Use        | Requires coding skills | No coding required                  |
| Preprocessing      | Mostly manual          | Fully automated                     |
| Insight Generation | Limited analysis       | AI-driven insights                  |
| Visualization      | Basic charts           | Clear visual dashboards             |
| Workflow           | Multiple tools needed  | Single integrated platform          |
| Security           | Minimal protection     | Data anonymization & access control |

## CONCLUSION :

In this project, Data Sculptor – AI Driven Preprocessing and Insight Extraction Tool successfully addresses the challenges of manual and complex data preprocessing.
By automating data cleaning, transformation, and insight extraction, the system reduces human effort and improves analytical accuracy. 
Its AI-driven approach, user-friendly web interface, and modular architecture make data analytics accessible to both technical and non-technical users. 
Overall, Data Sculptor provides an efficient, scalable, and reliable solution for transforming raw data into meaningful insights and supports informed decision-making across various application domains.

## WEBSITE : https://data-sculptorgit-xrou23exga6gqqnvwvvvrd.streamlit.app/



