import plotly.express as px
import plotly.graph_objects as go

def get_heatmap(df):
    """Returns an interactive Plotly heatmap."""
    # Select only numeric columns
    numeric_df = df.select_dtypes(include='number')
    
    if numeric_df.empty:
        return None  # No numeric columns to plot

    corr = numeric_df.corr()
    fig = px.imshow(
        corr, 
        text_auto=True if len(corr.columns) <= 10 else False,
        color_continuous_scale='Viridis',
        aspect="auto",
        title="Feature Correlation"
    )
    fig.update_layout(height=350, margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig

def get_trend_plot(df, column):
    """Returns an interactive Plotly line chart."""
    fig = px.line(
        df, 
        y=column, 
        markers=True, 
        template="plotly_white",
        title=f"{column} Trend Analysis"
    )
    fig.update_traces(line_color='#0F172A', marker=dict(color='#10B981', size=4))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10), font=dict(size=10))
    return fig

def get_distribution_plot(df, column):
    """Returns an interactive Plotly histogram with a marginal box plot."""
    fig = px.histogram(
        df, 
        x=column, 
        marginal="box", # Adds an innovative box plot on top
        template="plotly_white",
        title=f"{column} Distribution",
        color_discrete_sequence=['#10B981']
    )
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10), font=dict(size=10))
    return fig

def get_regression_plot(df, x_col, y_col):
    """Returns a scatter plot with a linear regression trend line."""
    fig = px.scatter(
        df, x=x_col, y=y_col, 
        trendline="ols",
        template="plotly_white",
        title=f"Regression: {x_col} vs {y_col}",
        color_discrete_sequence=['#3B82F6']
    )
    fig.update_layout(height=400)
    return fig

def get_forecast_plot(df, date_col, value_col, forecast_df):
    """Returns a line chart showing historical data and future predictions."""
    fig = go.Figure()
    # Historical
    fig.add_trace(go.Scatter(x=df[date_col], y=df[value_col], name="Historical", line=dict(color='#0F172A')))
    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast_df[date_col], 
        y=forecast_df[value_col], 
        name="AI Forecast", 
        line=dict(color='#10B981', dash='dash')
    ))
    fig.update_layout(
        title=f"Predictive Forecast: {value_col}",
        template="plotly_white",
        height=400
    )
    return fig
