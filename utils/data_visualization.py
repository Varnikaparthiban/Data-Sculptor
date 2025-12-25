import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import os
import time
import pandas as pd

matplotlib.use('Agg')  # Required for headless server environments

def save_heatmap(df):
    """Save correlation heatmap to static/images with a unique filename."""
    # Select only numeric columns
    numeric_df = df.select_dtypes(include='number')
    
    if numeric_df.empty:
        return None  # No numeric columns to plot

    plt.figure(figsize=(8,6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
    
    timestamp = int(time.time() * 1000)  # unique timestamp
    filename = f"heatmap_{timestamp}.png"
    filepath = os.path.join("static", "images", filename)
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()
    
    return filename

def save_trend_plot(df, column):
    """Save trend plot of a column with improved style."""
    plt.figure(figsize=(6,4))
    
    # Set a light background style
    sns.set_style("whitegrid")
    
    # Create a colorful line plot with points
    plt.plot(df[column], marker='o', linestyle='-', markersize=6, color="#1abc9c", markerfacecolor="#e74c3c")
    
    plt.title(f'{column} Trend', fontsize=14, color="#2c3e50")
    plt.xlabel('Index', fontsize=12, color="#2c3e50")
    plt.ylabel(column, fontsize=12, color="#2c3e50")
    
    # Light background for plot
    plt.gca().set_facecolor('#f5f7fa')
    plt.grid(True, color='white', linestyle='-', linewidth=1, alpha=0.8)
    
    timestamp = int(time.time() * 1000)
    safe_column = column.replace(" ", "_")
    filename = f"{safe_column}_trend_{timestamp}.png"
    filepath = os.path.join("static", "images", filename)
    
    plt.savefig(filepath, bbox_inches='tight', facecolor='#f5f7fa')
    plt.close()
    
    return filename

def save_distribution_plot(df, column):
    """Save distribution plot (histogram + KDE) of a column."""
    plt.figure(figsize=(6,4))
    sns.set_style("whitegrid")
    
    # Histogram with KDE
    sns.histplot(df[column], kde=True, color="#3498db", edgecolor="#2980b9", alpha=0.6)
    
    plt.title(f'{column} Distribution', fontsize=14, color="#2c3e50")
    plt.xlabel(column, fontsize=12, color="#2c3e50")
    plt.ylabel('Frequency', fontsize=12, color="#2c3e50")
    
    plt.gca().set_facecolor('#f5f7fa')
    
    timestamp = int(time.time() * 1000)
    safe_column = column.replace(" ", "_")
    filename = f"{safe_column}_dist_{timestamp}.png"
    filepath = os.path.join("static", "images", filename)
    
    plt.savefig(filepath, bbox_inches='tight', facecolor='#f5f7fa')
    plt.close()
    
    return filename
