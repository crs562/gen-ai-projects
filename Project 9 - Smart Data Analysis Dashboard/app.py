"""
Smart Data Analytics Dashboard
============================
A simplified, deployable version of the advanced BI dashboard
Optimized for Render deployment with essential features
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import hashlib
import seaborn as sns
from PIL import Image
import chardet

# LLM Integration (simplified)
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    st.warning("Groq SDK not installed. AI features will be limited.")

# Configuration
SAMPLE_DATASETS = ["Titanic", "Iris", "Tips", "Car Crashes", "Flights"]
CHART_TYPES = ["Bar", "Line", "Scatter", "Histogram", "Box", "Pie", "Heatmap"]
AGGREGATION_FUNCTIONS = ["None", "Sum", "Mean", "Count", "Min", "Max", "Median"]

# App Configuration
st.set_page_config(
    page_title="Smart Data Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
    .chart-container {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        background: white;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    defaults = {
        'df': None,
        'dashboards': {},
        'current_dashboard': None,
        'analysis_history': [],
        'groq_client': None
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    # Initialize Groq client if available
    if GROQ_AVAILABLE and not st.session_state.groq_client:
        groq_api_key = os.environ.get("GROQ_API_KEY")
        if groq_api_key:
            try:
                st.session_state.groq_client = Groq(api_key=groq_api_key)
            except Exception as e:
                st.error(f"Failed to initialize Groq client: {e}")

def load_sample_data(dataset_name):
    """Load sample datasets"""
    try:
        if dataset_name == "Titanic":
            df = sns.load_dataset('titanic')
        elif dataset_name == "Iris":
            df = sns.load_dataset('iris')
        elif dataset_name == "Tips":
            df = sns.load_dataset('tips')
        elif dataset_name == "Car Crashes":
            df = sns.load_dataset('car_crashes')
        elif dataset_name == "Flights":
            df = sns.load_dataset('flights')
        else:
            return None
        
        return clean_data(df)
    except Exception as e:
        st.error(f"Error loading {dataset_name}: {e}")
        return None

def clean_data(df):
    """Clean and prepare data"""
    if df is None:
        return None
    
    # Clean column names
    df.columns = [str(col).strip().replace(' ', '_').replace('(', '').replace(')', '') 
                  for col in df.columns]
    
    # Convert date columns
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_datetime(df[col], errors='ignore')
            except:
                pass
    
    return df

def load_csv_file(uploaded_file):
    """Load CSV file with encoding detection"""
    try:
        # Detect encoding
        raw_data = uploaded_file.read(10000)
        uploaded_file.seek(0)
        detected = chardet.detect(raw_data)
        encoding = detected['encoding']
        
        encodings_to_try = [encoding, 'utf-8', 'latin1', 'iso-8859-1']
        
        for enc in encodings_to_try:
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding=enc)
                st.success(f"CSV loaded successfully with {enc} encoding!")
                return clean_data(df)
            except:
                continue
        
        st.error("Failed to load CSV file")
        return None
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def create_visualization(df, chart_type, x_col, y_col, color_col=None, agg_func="None"):
    """Create visualizations based on parameters"""
    try:
        plot_df = df.copy()
        
        # Apply aggregation if needed
        if agg_func != "None" and x_col and y_col:
            numeric_cols = df.select_dtypes(include=np.number).columns
            if y_col in numeric_cols:
                agg_map = {
                    'Sum': 'sum', 'Mean': 'mean', 'Count': 'count',
                    'Min': 'min', 'Max': 'max', 'Median': 'median'
                }
                if agg_func in agg_map:
                    plot_df = df.groupby(x_col)[y_col].agg(agg_map[agg_func]).reset_index()
        
        # Create charts
        fig = None
        title = f"{chart_type} Chart"
        
        if chart_type == "Bar":
            fig = px.bar(plot_df, x=x_col, y=y_col, color=color_col, title=title)
        elif chart_type == "Line":
            fig = px.line(plot_df, x=x_col, y=y_col, color=color_col, title=title)
        elif chart_type == "Scatter":
            fig = px.scatter(plot_df, x=x_col, y=y_col, color=color_col, title=title)
        elif chart_type == "Histogram":
            fig = px.histogram(plot_df, x=x_col, color=color_col, title=title)
        elif chart_type == "Box":
            fig = px.box(plot_df, x=x_col, y=y_col, color=color_col, title=title)
        elif chart_type == "Pie":
            if y_col:
                fig = px.pie(plot_df, names=x_col, values=y_col, title=title)
            else:
                value_counts = plot_df[x_col].value_counts()
                fig = px.pie(values=value_counts.values, names=value_counts.index, title=title)
        elif chart_type == "Heatmap":
            numeric_df = plot_df.select_dtypes(include=[np.number])
            if len(numeric_df.columns) > 1:
                corr_matrix = numeric_df.corr()
                fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", title="Correlation Heatmap")
        
        return fig
    except Exception as e:
        st.error(f"Error creating chart: {e}")
        return None

def analyze_with_ai(prompt, df_context):
    """Analyze data using Groq AI"""
    if not st.session_state.groq_client:
        return "AI analysis not available. Please set GROQ_API_KEY environment variable."
    
    try:
        # Prepare data context
        df_info = f"""
Dataset Shape: {df_context.shape}
Columns: {', '.join(df_context.columns[:10])}
Sample Data:
{df_context.head(3).to_string()}
        """
        
        full_prompt = f"""
You are a data analyst. Analyze the following dataset and answer the user's question.

{df_info}

User Question: {prompt}

Provide insights, patterns, and actionable recommendations in a clear, structured format.
"""
        
        response = st.session_state.groq_client.chat.completions.create(
            messages=[{"role": "user", "content": full_prompt}],
            model="llama3-70b-8192",
            temperature=0.3,
            max_tokens=1000
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"AI analysis error: {e}"

def render_sidebar():
    """Render sidebar for data loading and configuration"""
    with st.sidebar:
        st.header("📁 Data Source")
        
        # Data source selection
        data_source = st.radio(
            "Choose data source:",
            ["Sample Dataset", "Upload CSV", "Manual Entry"]
        )
        
        if data_source == "Sample Dataset":
            dataset = st.selectbox("Select dataset:", SAMPLE_DATASETS)
            if st.button("Load Dataset", type="primary"):
                df = load_sample_data(dataset)
                if df is not None:
                    st.session_state.df = df
                    st.success(f"Loaded {dataset} dataset!")
                    st.rerun()
        
        elif data_source == "Upload CSV":
            uploaded_file = st.file_uploader("Choose CSV file", type=["csv"])
            if uploaded_file:
                df = load_csv_file(uploaded_file)
                if df is not None:
                    st.session_state.df = df
                    st.rerun()
        
        elif data_source == "Manual Entry":
            st.subheader("Create Sample Data")
            if st.button("Generate Sample Sales Data"):
                np.random.seed(42)
                dates = pd.date_range('2023-01-01', periods=100)
                data = {
                    'Date': dates,
                    'Sales': np.random.normal(1000, 200, 100),
                    'Category': np.random.choice(['A', 'B', 'C'], 100),
                    'Region': np.random.choice(['North', 'South', 'East', 'West'], 100)
                }
                st.session_state.df = pd.DataFrame(data)
                st.success("Sample data generated!")
                st.rerun()
        
        # Dashboard management
        st.markdown("---")
        st.header("📊 Dashboard")
        
        if st.session_state.df is not None:
            new_dashboard = st.text_input("New dashboard name:")
            if st.button("Create Dashboard") and new_dashboard:
                st.session_state.dashboards[new_dashboard] = {"charts": [], "created": datetime.now()}
                st.session_state.current_dashboard = new_dashboard
                st.success(f"Created dashboard: {new_dashboard}")
                st.rerun()
            
            if st.session_state.dashboards:
                dashboard_names = list(st.session_state.dashboards.keys())
                selected = st.selectbox("Select dashboard:", dashboard_names)
                if selected != st.session_state.current_dashboard:
                    st.session_state.current_dashboard = selected
                    st.rerun()

def render_data_overview():
    """Render data overview section"""
    if st.session_state.df is None:
        st.info("👋 Welcome! Please load a dataset from the sidebar to get started.")
        return
    
    df = st.session_state.df
    
    st.markdown('<div class="main-header"><h1>📊 Smart Data Analytics Dashboard</h1></div>', 
                unsafe_allow_html=True)
    
    # Data overview metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Rows", f"{len(df):,}")
    with col2:
        st.metric("Total Columns", len(df.columns))
    with col3:
        st.metric("Memory Usage", f"{df.memory_usage().sum() / 1024:.1f} KB")
    with col4:
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        st.metric("Missing Data", f"{missing_pct:.1f}%")
    
    # Data preview
    with st.expander("📋 Data Preview", expanded=True):
        st.dataframe(df.head(10), use_container_width=True)
    
    # Data types and info
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📈 Data Types")
        dtype_df = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes.astype(str),
            'Non-Null': df.count(),
            'Null Count': df.isnull().sum()
        })
        st.dataframe(dtype_df, use_container_width=True)
    
    with col2:
        st.subheader("📊 Numeric Summary")
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            st.dataframe(numeric_df.describe(), use_container_width=True)
        else:
            st.info("No numeric columns found")

def render_visualization_section():
    """Render visualization creation section"""
    if st.session_state.df is None:
        return
    
    df = st.session_state.df
    
    st.header("📈 Create Visualizations")
    
    # Chart configuration
    col1, col2, col3 = st.columns(3)
    with col1:
        chart_type = st.selectbox("Chart Type:", CHART_TYPES)
    with col2:
        x_column = st.selectbox("X-axis:", [None] + list(df.columns))
    with col3:
        y_column = st.selectbox("Y-axis:", [None] + list(df.columns))
    
    col1, col2 = st.columns(2)
    with col1:
        color_column = st.selectbox("Color by:", [None] + list(df.columns))
    with col2:
        if y_column and y_column in df.select_dtypes(include=[np.number]).columns:
            agg_func = st.selectbox("Aggregation:", AGGREGATION_FUNCTIONS)
        else:
            agg_func = "None"
    
    # Generate chart
    if st.button("Create Chart", type="primary"):
        if chart_type == "Heatmap" or (x_column and (y_column or chart_type in ["Histogram", "Pie"])):
            fig = create_visualization(df, chart_type, x_column, y_column, color_column, agg_func)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                
                # Save to dashboard
                if st.session_state.current_dashboard:
                    chart_config = {
                        "type": chart_type,
                        "x_axis": x_column,
                        "y_axis": y_column,
                        "color": color_column,
                        "aggregation": agg_func,
                        "created": datetime.now().isoformat()
                    }
                    st.session_state.dashboards[st.session_state.current_dashboard]["charts"].append(chart_config)
                    st.success("Chart added to dashboard!")
        else:
            st.warning("Please select appropriate columns for the chart type.")

def render_ai_analysis():
    """Render AI analysis section"""
    if st.session_state.df is None:
        return
    
    st.header("🤖 AI-Powered Analysis")
    
    # Quick analysis buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📊 Dataset Overview"):
            prompt = "Provide a comprehensive overview of this dataset including key statistics, patterns, and insights."
            analysis = analyze_with_ai(prompt, st.session_state.df)
            st.markdown(analysis)
    
    with col2:
        if st.button("🔍 Data Quality Check"):
            prompt = "Analyze the data quality including missing values, outliers, and potential data issues."
            analysis = analyze_with_ai(prompt, st.session_state.df)
            st.markdown(analysis)
    
    with col3:
        if st.button("💡 Business Insights"):
            prompt = "Identify key business insights and actionable recommendations from this data."
            analysis = analyze_with_ai(prompt, st.session_state.df)
            st.markdown(analysis)
    
    # Custom analysis
    st.subheader("Custom Analysis")
    custom_question = st.text_area("Ask a question about your data:", 
                                   placeholder="e.g., What are the main trends in this dataset?")
    
    if st.button("Analyze", type="primary") and custom_question:
        with st.spinner("🧠 AI is thinking..."):
            analysis = analyze_with_ai(custom_question, st.session_state.df)
            st.markdown("### Analysis Results:")
            st.markdown(analysis)
            
            # Save to history
            st.session_state.analysis_history.append({
                "question": custom_question,
                "answer": analysis,
                "timestamp": datetime.now()
            })

def render_dashboard():
    """Render saved dashboard"""
    if not st.session_state.current_dashboard:
        return
    
    dashboard = st.session_state.dashboards[st.session_state.current_dashboard]
    
    st.header(f"📊 Dashboard: {st.session_state.current_dashboard}")
    
    if not dashboard["charts"]:
        st.info("No charts in this dashboard yet. Create some visualizations to add them!")
        return
    
    # Display charts in grid
    charts_per_row = 2
    charts = dashboard["charts"]
    
    for i in range(0, len(charts), charts_per_row):
        cols = st.columns(charts_per_row)
        for j, col in enumerate(cols):
            if i + j < len(charts):
                chart_config = charts[i + j]
                with col:
                    fig = create_visualization(
                        st.session_state.df,
                        chart_config["type"],
                        chart_config["x_axis"],
                        chart_config["y_axis"],
                        chart_config["color"],
                        chart_config["aggregation"]
                    )
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

def main():
    """Main application function"""
    initialize_session_state()
    
    # Render sidebar
    render_sidebar()
    
    # Main content tabs
    if st.session_state.df is not None:
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Overview", "📈 Visualizations", "🤖 AI Analysis", "📱 Dashboard"])
        
        with tab1:
            render_data_overview()
        
        with tab2:
            render_visualization_section()
        
        with tab3:
            render_ai_analysis()
        
        with tab4:
            render_dashboard()
    else:
        render_data_overview()
    
    # Footer
    st.markdown("---")
    st.markdown("**Smart Data Analytics Dashboard** | Built with Streamlit & Plotly")

if __name__ == "__main__":
    main()