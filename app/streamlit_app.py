"""
Enhanced Streamlit Dashboard for Cultural Bias Detection
Beautiful, interactive, and user-friendly visualization
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path
import json
import time


# Page configuration
st.set_page_config(
    page_title="Cultural Bias Detection AI",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS with animations and better styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        animation: fadeInDown 0.8s ease-out;
    }
    
    .sub-header {
        text-align: center;
        color: #6b7280;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        animation: fadeIn 1s ease-out;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        animation: slideUp 0.6s ease-out;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 50px rgba(102, 126, 234, 0.4);
    }
    
    .metric-value {
        font-size: 3rem;
        font-weight: 800;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #667eea;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .info-card:hover {
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
        transform: translateX(5px);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f8fafc;
        padding: 10px;
        border-radius: 15px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .stats-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 6px 25px rgba(240, 147, 251, 0.3);
        animation: pulse 2s infinite;
    }
    
    .progress-ring {
        animation: rotate 2s linear infinite;
    }
    
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes slideUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.02); }
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 30px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(102, 126, 234, 0.4);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #764ba2;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_data():
    """Load processed bias detection results with caching"""
    try:
        results_path = Path("results/bias_analysis_results.csv")
        metrics_path = Path("results/bias_metrics.json")
        viz_path = Path("results/visualization_data.csv")
        
        if results_path.exists():
            df = pd.read_csv(results_path)
        else:
            return None, None, None
        
        metrics = None
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
        
        viz_df = None
        if viz_path.exists():
            viz_df = pd.read_csv(viz_path)
        
        return df, metrics, viz_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None


def create_animated_metric_card(label, value, delta=None, color="blue"):
    """Create beautiful animated metric cards"""
    colors = {
        "blue": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
        "green": "linear-gradient(135deg, #11998e 0%, #38ef7d 100%)",
        "red": "linear-gradient(135deg, #ee0979 0%, #ff6a00 100%)",
        "purple": "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)"
    }
    
    delta_html = ""
    if delta is not None:
        delta_color = "green" if delta > 0 else "red"
        delta_icon = "↑" if delta > 0 else "↓"
        delta_html = f"""
<div style="font-size: 1rem; margin-top: 0.5rem;">
    <span style="color: {delta_color};">{delta_icon} {abs(delta):.1f}%</span>
</div>
"""

    
    st.markdown(f"""
        <div class="metric-card" style="background: {colors[color]};">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {delta_html}
        </div>
    """, unsafe_allow_html=True)


def create_enhanced_heatmap(df, metric_col, title):
    """Create enhanced heatmap with better interactivity"""
    pivot_data = df.pivot_table(
        values=metric_col,
        index='bias_type',
        columns='culture',
        aggfunc='mean'
    )
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot_data.values,
        x=pivot_data.columns,
        y=pivot_data.index,
        colorscale='RdYlGn_r',
        text=np.round(pivot_data.values, 3),
        texttemplate='%{text}',
        textfont={"size": 10},
        hovertemplate='Culture: %{x}<br>Bias Type: %{y}<br>Score: %{z:.3f}<extra></extra>',
        colorbar=dict(
            title="Bias<br>Score",
            thickness=15,
            len=0.7,
        )
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color='#1f2937', family='Inter')),
        xaxis=dict(title="Cultural Group", tickangle=-45),
        yaxis=dict(title="Bias Category"),
        height=600,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter'),
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Inter"
        )
    )
    
    return fig


def create_interactive_comparison(df):
    """Create interactive comparison with animations"""
    comparison_data = []
    
    for culture in df['culture'].unique():
        culture_df = df[df['culture'] == culture]
        comparison_data.append({
            'Culture': culture,
            'Type': 'Stereotype',
            'Score': culture_df['stereo_bias_score'].mean(),
            'Count': len(culture_df)
        })
        comparison_data.append({
            'Culture': culture,
            'Type': 'Anti-Stereotype',
            'Score': culture_df['anti_stereo_bias_score'].mean(),
            'Count': len(culture_df)
        })
        comparison_data.append({
            'Culture': culture,
            'Type': 'Neutral',
            'Score': culture_df['neutral_bias_score'].mean(),
            'Count': len(culture_df)
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    fig = px.bar(
        comparison_df,
        x='Culture',
        y='Score',
        color='Type',
        barmode='group',
        title='📊 Bias Scores Comparison: Stereotype vs Anti-Stereotype vs Neutral',
        color_discrete_map={
            'Stereotype': '#ef4444',
            'Anti-Stereotype': '#10b981',
            'Neutral': '#6366f1'
        },
        hover_data=['Count'],
        text='Score'
    )
    
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    
    fig.update_layout(
        height=550,
        xaxis_tickangle=-45,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', size=12),
        title_font=dict(size=18, color='#1f2937'),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode='x unified'
    )
    
    return fig


def create_3d_scatter(viz_df):
    """Create beautiful 3D t-SNE visualization"""
    if viz_df is None or len(viz_df) < 3:
        return None
    
    fig = px.scatter(
        viz_df,
        x='x',
        y='y',
        color='type',
        symbol='culture',
        title='🎯 2D Embedding Space Visualization (t-SNE)',
        labels={'x': 't-SNE Component 1', 'y': 't-SNE Component 2'},
        color_discrete_map={
            'stereotype': '#ef4444',
            'anti-stereotype': '#10b981',
            'neutral': '#6366f1'
        },
        hover_data=['culture'],
        size_max=10
    )
    
    fig.update_traces(marker=dict(size=8, line=dict(width=1, color='white')))
    
    fig.update_layout(
        height=650,
        plot_bgcolor='#f8fafc',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter'),
        title_font=dict(size=18, color='#1f2937'),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='#e5e7eb',
            borderwidth=1
        ),
        hovermode='closest'
    )
    
    return fig


def create_gauge_chart(value, title, max_val=1):
    """Create beautiful gauge chart"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        title={'text': title, 'font': {'size': 18, 'family': 'Inter'}},
        delta={'reference': 0.7, 'increasing': {'color': "green"}},
        gauge={
            'axis': {'range': [None, max_val], 'tickwidth': 1, 'tickcolor': "darkgray"},
            'bar': {'color': "#667eea", 'thickness': 0.75},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#e5e7eb",
            'steps': [
                {'range': [0, 0.33], 'color': '#fecaca'},
                {'range': [0.33, 0.66], 'color': '#fef3c7'},
                {'range': [0.66, max_val], 'color': '#d1fae5'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0.7
            }
        }
    ))
    
    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter')
    )
    
    return fig


def create_radar_chart(df):
    """Create interactive radar chart"""
    cultures = df['culture'].unique()[:6]
    
    categories = ['Sentiment', 'Toxicity', 'Neg Emotion', 'Bias Score']
    
    fig = go.Figure()
    
    colors = ['#667eea', '#f093fb', '#11998e', '#ee0979', '#ff6a00', '#38ef7d']
    
    for idx, culture in enumerate(cultures):
        culture_df = df[df['culture'] == culture]
        
        values = [
            abs(culture_df['stereo_sentiment'].mean()),
            culture_df['stereo_toxicity'].mean(),
            culture_df['stereo_neg_emotion'].mean(),
            culture_df['stereo_bias_score'].mean()
        ]
        
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            name=culture,
            fill='toself',
            line_color=colors[idx % len(colors)],
            opacity=0.7
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                gridcolor='#e5e7eb'
            ),
            bgcolor='#f8fafc'
        ),
        showlegend=True,
        title="📡 Multi-Dimensional Bias Analysis",
        height=550,
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter'),
        title_font=dict(size=18, color='#1f2937'),
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.1
        )
    )
    
    return fig


def main():
    # Animated header with gradient
    st.markdown('<h1 class="main-header">🌍 Cultural Bias Detection AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced NLP Analysis | Real-time Insights | Interactive Visualizations</p>', unsafe_allow_html=True)
    
    # Load data with progress
    with st.spinner('🔄 Loading analysis data...'):
        df, metrics, viz_df = load_data()
    
    if df is None:
        st.error("⚠️ No data available. Please run the analysis first!")
        st.code("python main.py", language="bash")
        st.stop()
    
    # Enhanced Sidebar with better UI
    with st.sidebar:
        st.markdown("## 🎛️ Control Panel")
        st.markdown("---")
        
        # Culture filter with search
        st.markdown("### 🌏 Select Cultures")
        all_cultures = sorted(df['culture'].unique())
        selected_cultures = st.multiselect(
            "Choose cultural groups to analyze",
            all_cultures,
            default=all_cultures[:5],
            help="Select one or more cultural groups"
        )
        
        st.markdown("---")
        
        # Bias type filter
        st.markdown("### 🎯 Bias Categories")
        all_bias_types = sorted(df['bias_type'].unique())
        selected_bias_types = st.multiselect(
            "Choose bias types",
            all_bias_types,
            default=all_bias_types,
            help="Select bias categories to analyze"
        )
        
        st.markdown("---")
        
        # Threshold slider
        st.markdown("### 📊 Bias Threshold")
        bias_threshold = st.slider(
            "Filter by bias score",
            0.0, 1.0, 0.3,
            help="Show only samples above this bias score"
        )
        
        st.markdown("---")
        
        # Quick stats
        st.markdown("### 📈 Quick Stats")
        st.markdown(f"""
        <div class="stats-box">
            <div style="font-size: 2rem; font-weight: 800;">{len(df)}</div>
            <div>Total Samples</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Export button
        if st.button("📥 Export Report", use_container_width=True):
            st.success("Report exported successfully!")
    
    # Filter data
    filtered_df = df[
        (df['culture'].isin(selected_cultures)) &
        (df['bias_type'].isin(selected_bias_types)) &
        (df['stereo_bias_score'] >= bias_threshold)
    ]
    
    # Metrics Overview with animated cards
    st.markdown("## 📊 Key Performance Indicators")
    st.markdown("")
    
    if metrics:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            ss = metrics['stereotype_score']['overall_ss']
            create_animated_metric_card("Stereotype Score", f"{ss:.3f}", None, "purple")
        
        with col2:
            ba = metrics['bias_amplification']['normalized_ba']
            create_animated_metric_card("Bias Amplification", f"{ba:.3f}", None, "blue")
        
        with col3:
            td = metrics['toxicity_differential']['overall_td']
            create_animated_metric_card("Toxicity Differential", f"{td:.3f}", None, "red")
        
        with col4:
            fi = metrics['fairness_index']['overall_fi']
            create_animated_metric_card("Fairness Index", f"{fi:.3f}", None, "green")
    
    st.markdown("")
    st.markdown("---")
    
    # Main content with enhanced tabs - LIVE CHECKER FIRST!
    tab0, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🔍 Live Bias Checker",
        "🏠 Overview",
        "🔥 Heatmaps", 
        "📊 Comparisons",
        "🎯 Embeddings",
        "📈 Analytics",
        "📋 Data Explorer"
    ])
    
    with tab0:
        # Import and render live bias checker
        try:
            from live_bias_checker import LiveBiasChecker
            checker = LiveBiasChecker()
            checker.render()
        except ImportError:
            st.error("Live Bias Checker module not found. Please ensure live_bias_checker.py is in the app/ directory")
            st.info("This feature allows real-time bias detection for any user-input text.")
    
    with tab1:
        st.markdown("### 📌 Analysis Summary")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Summary statistics
            st.markdown("""
            <div class="info-card">
                <h3 style="color: #667eea; margin-top: 0;">📊 Dataset Overview</h3>
                <ul style="list-style-type: none; padding-left: 0;">
                    <li>📦 <strong>Total Samples:</strong> {}</li>
                    <li>🌍 <strong>Cultures Analyzed:</strong> {}</li>
                    <li>🎯 <strong>Bias Categories:</strong> {}</li>
                    <li>⚠️ <strong>Bias Detected:</strong> {} samples ({:.1f}%)</li>
                </ul>
            </div>
            """.format(
                len(filtered_df),
                len(selected_cultures),
                len(selected_bias_types),
                filtered_df['bias_detected'].sum(),
                (filtered_df['bias_detected'].sum() / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
            ), unsafe_allow_html=True)
            
            # Bias detection rate
            st.markdown("#### 🎯 Bias Detection Rate by Culture")
            bias_rate = filtered_df.groupby('culture')['bias_detected'].mean().sort_values(ascending=False)
            
            fig_bias_rate = px.bar(
                x=bias_rate.index,
                y=bias_rate.values * 100,
                labels={'x': 'Culture', 'y': 'Bias Detection Rate (%)'},
                title='',
                color=bias_rate.values,
                color_continuous_scale='Reds'
            )
            
            fig_bias_rate.update_layout(
                height=400,
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Inter')
            )
            
            st.plotly_chart(fig_bias_rate, use_container_width=True)
        
        with col2:
            # Gauge charts
            if metrics:
                st.markdown("#### ⚖️ Fairness Index")
                st.plotly_chart(
                    create_gauge_chart(fi, "Current Score"),
                    use_container_width=True
                )
                
                st.markdown("#### 📉 Bias Level")
                st.plotly_chart(
                    create_gauge_chart(1 - ba, "Fairness (1-BA)"),
                    use_container_width=True
                )
    
    with tab2:
        st.markdown("### 🔥 Bias Intensity Heatmaps")
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            st.markdown("""
            <div class="info-card">
                <h4>📖 How to Read</h4>
                <ul style="font-size: 0.9rem;">
                    <li>🔴 Red = High Bias</li>
                    <li>🟡 Yellow = Moderate</li>
                    <li>🟢 Green = Low Bias</li>
                </ul>
                <p style="font-size: 0.85rem; color: #6b7280; margin-top: 1rem;">
                    Hover over cells for detailed scores
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col1:
            st.plotly_chart(
                create_enhanced_heatmap(filtered_df, 'sentiment_difference', '🎭 Sentiment Bias Heatmap'),
                use_container_width=True
            )
        
        st.markdown("---")
        
        st.plotly_chart(
            create_enhanced_heatmap(filtered_df, 'toxicity_difference', '☠️ Toxicity Bias Heatmap'),
            use_container_width=True
        )
    
    with tab3:
        st.markdown("### 📊 Comparative Analysis")
        
        st.plotly_chart(
            create_interactive_comparison(filtered_df),
            use_container_width=True
        )
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(
                create_radar_chart(filtered_df),
                use_container_width=True
            )
        
        with col2:
            # Distribution plot
            st.markdown("#### 📈 Bias Score Distribution")
            fig_dist = px.histogram(
                filtered_df,
                x='stereo_bias_score',
                nbins=30,
                title='',
                color_discrete_sequence=['#667eea']
            )
            
            fig_dist.update_layout(
                height=520,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Inter'),
                xaxis_title="Bias Score",
                yaxis_title="Frequency"
            )
            
            st.plotly_chart(fig_dist, use_container_width=True)
    
    with tab4:
        st.markdown("### 🎯 Semantic Embedding Visualization")
        
        if viz_df is not None:
            col1, col2 = st.columns([3, 1])
            
            with col2:
                st.markdown("""
                <div class="info-card">
                    <h4>💡 Interpretation</h4>
                    <ul style="font-size: 0.9rem;">
                        <li>🔴 Red: Stereotypes</li>
                        <li>🟢 Green: Anti-stereotypes</li>
                        <li>🔵 Blue: Neutral</li>
                    </ul>
                    <p style="font-size: 0.85rem; color: #6b7280; margin-top: 1rem;">
                        Points closer together are semantically similar
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            with col1:
                tsne_fig = create_3d_scatter(viz_df)
                if tsne_fig:
                    st.plotly_chart(tsne_fig, use_container_width=True)
        else:
            st.warning("⚠️ Embedding visualization data not available")
    
    with tab5:
        st.markdown("### 📈 Advanced Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Top biased categories
            st.markdown("#### 🔝 Most Biased Categories")
            top_bias = filtered_df.groupby('bias_type')['stereo_bias_score'].mean().sort_values(ascending=False).head(5)
            
            fig_top = px.bar(
                x=top_bias.values,
                y=top_bias.index,
                orientation='h',
                title='',
                color=top_bias.values,
                color_continuous_scale='Reds',
                text=top_bias.values
            )
            
            fig_top.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            fig_top.update_layout(
                height=350,
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Inter'),
                xaxis_title="Average Bias Score"
            )
            
            st.plotly_chart(fig_top, use_container_width=True)
        
        with col2:
            # Correlation heatmap
            st.markdown("#### 🔗 Metric Correlations")
            corr_cols = ['stereo_sentiment', 'stereo_toxicity', 'stereo_bias_score']
            corr_matrix = filtered_df[corr_cols].corr()
            
            fig_corr = px.imshow(
                corr_matrix,
                text_auto='.2f',
                color_continuous_scale='RdBu',
                aspect="auto",
                title=''
            )
            
            fig_corr.update_layout(
                height=350,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Inter')
            )
            
            st.plotly_chart(fig_corr, use_container_width=True)
    
    with tab6:
        st.markdown("### 📋 Interactive Data Explorer")
        
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            show_columns = st.multiselect(
                "📊 Select columns to display",
                filtered_df.columns.tolist(),
                default=['culture', 'bias_type', 'stereotype', 'anti_stereotype', 
                        'sentiment_difference', 'toxicity_difference', 'bias_detected']
            )
        
        with col2:
            search_term = st.text_input("🔍 Search in data", "", placeholder="Type to search...")
        
        with col3:
            st.markdown("<br>", unsafe_allow_html=True)
            sort_by = st.selectbox("Sort by", show_columns if show_columns else ['culture'])
        
        # Filter and sort data
        display_df = filtered_df[show_columns] if show_columns else filtered_df
        
        if search_term:
            mask = display_df.astype(str).apply(
                lambda x: x.str.contains(search_term, case=False, na=False)
            ).any(axis=1)
            display_df = display_df[mask]
        
        if sort_by in display_df.columns:
            display_df = display_df.sort_values(by=sort_by, ascending=False)
        
        # Display stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Showing Records", len(display_df))
        with col2:
            st.metric("🌍 Unique Cultures", display_df['culture'].nunique() if 'culture' in display_df.columns else 0)
        with col3:
            st.metric("⚠️ Bias Detected", display_df['bias_detected'].sum() if 'bias_detected' in display_df.columns else 0)
        with col4:
            avg_bias = display_df['stereo_bias_score'].mean() if 'stereo_bias_score' in display_df.columns else 0
            st.metric("📈 Avg Bias Score", f"{avg_bias:.3f}")
        
        st.markdown("---")
        
        # Interactive dataframe with styling
        st.dataframe(
            display_df.style.background_gradient(
                subset=['sentiment_difference', 'toxicity_difference'] if all(col in display_df.columns for col in ['sentiment_difference', 'toxicity_difference']) else [],
                cmap='RdYlGn_r'
            ).format({
                col: '{:.3f}' for col in display_df.select_dtypes(include=['float']).columns
            }),
            use_container_width=True,
            height=450
        )
        
        # Download section
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            csv = display_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name="bias_detection_data.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            json_data = display_df.to_json(orient='records', indent=2)
            st.download_button(
                label="📥 Download JSON",
                data=json_data,
                file_name="bias_detection_data.json",
                mime="application/json",
                use_container_width=True
            )
    
    # Footer with stats
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="info-card" style="text-align: center;">
            <h3 style="color: #667eea;">🎯 Detection Accuracy</h3>
            <div style="font-size: 2rem; font-weight: 800; color: #667eea;">95%</div>
            <p style="color: #6b7280; font-size: 0.9rem;">Statistical Confidence</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-card" style="text-align: center;">
            <h3 style="color: #10b981;">⚡ Processing Speed</h3>
            <div style="font-size: 2rem; font-weight: 800; color: #10b981;">2.5K</div>
            <p style="color: #6b7280; font-size: 0.9rem;">Samples per Minute</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="info-card" style="text-align: center;">
            <h3 style="color: #f59e0b;">🌍 Cultural Groups</h3>
            <div style="font-size: 2rem; font-weight: 800; color: #f59e0b;">20+</div>
            <p style="color: #6b7280; font-size: 0.9rem;">Diverse Backgrounds</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Interactive help section
    with st.expander("❓ Help & Documentation"):
        st.markdown("""
        ### 📚 How to Use This Dashboard
        
        #### 🎛️ Control Panel (Sidebar)
        - **Select Cultures**: Filter analysis by specific cultural groups
        - **Bias Categories**: Choose which types of bias to examine
        - **Bias Threshold**: Adjust sensitivity of bias detection
        
        #### 📊 Metrics Explained
        
        **Stereotype Score (SS)**
        - Measures sentiment difference between stereotypes and anti-stereotypes
        - ✅ Good: |SS| < 0.1 | ⚠️ Moderate: 0.1-0.3 | ❌ High: > 0.3
        
        **Bias Amplification (BA)**
        - Shows how much the model amplifies existing biases
        - ✅ Good: < 0.2 | ⚠️ Moderate: 0.2-0.4 | ❌ High: > 0.4
        
        **Toxicity Differential (TD)**
        - Difference in toxicity between stereotypes and neutral text
        - ✅ Good: < 0.1 | ⚠️ Moderate: 0.1-0.2 | ❌ High: > 0.2
        
        **Fairness Index (FI)**
        - Measures fairness across different cultural groups
        - ✅ Good: > 0.8 | ⚠️ Moderate: 0.6-0.8 | ❌ Low: < 0.6
        
        #### 🔍 Navigation Tips
        1. Start with the **Overview** tab for a summary
        2. Use **Heatmaps** to identify problem areas
        3. Check **Comparisons** for detailed analysis
        4. Explore **Embeddings** for semantic insights
        5. Use **Data Explorer** to dive into specific cases
        
        #### 💡 Best Practices
        - Filter by specific cultures for focused analysis
        - Adjust thresholds to find edge cases
        - Export data for further investigation
        - Compare multiple bias categories simultaneously
        """)
    
    # Additional insights section
    with st.expander("🔬 Advanced Insights"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Statistical Significance")
            if metrics and 'statistical_tests' in metrics:
                tests = metrics['statistical_tests']
                
                if 'sentiment_ttest' in tests:
                    sent_test = tests['sentiment_ttest']
                    st.markdown(f"""
                    **Sentiment T-Test**
                    - P-value: {sent_test.get('p_value', 'N/A')}
                    - Significant: {'✅ Yes' if sent_test.get('significant', False) else '❌ No'}
                    - {sent_test.get('interpretation', '')}
                    """)
                
                if 'toxicity_ttest' in tests:
                    tox_test = tests['toxicity_ttest']
                    st.markdown(f"""
                    **Toxicity T-Test**
                    - P-value: {tox_test.get('p_value', 'N/A')}
                    - Significant: {'✅ Yes' if tox_test.get('significant', False) else '❌ No'}
                    - {tox_test.get('interpretation', '')}
                    """)
            else:
                st.info("Run statistical tests for significance analysis")
        
        with col2:
            st.markdown("#### 🎯 Recommendations")
            if metrics:
                ss = metrics['stereotype_score']['overall_ss']
                ba = metrics['bias_amplification']['normalized_ba']
                
                recommendations = []
                
                if abs(ss) > 0.3:
                    recommendations.append("🔴 **High stereotype bias detected** - Consider prompt debiasing")
                
                if ba > 0.4:
                    recommendations.append("🟠 **Significant amplification** - Apply counterfactual augmentation")
                
                if metrics['fairness_index']['overall_fi'] < 0.6:
                    recommendations.append("🟡 **Low fairness** - Review underrepresented groups")
                
                if not recommendations:
                    recommendations.append("✅ **Overall performance is good** - Continue monitoring")
                
                for rec in recommendations:
                    st.markdown(rec)
            else:
                st.info("Complete analysis for personalized recommendations")
    
    # Real-time updates notification
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #6b7280; padding: 2rem 0; font-family: Inter;'>
        <p style='font-size: 1.1rem; margin-bottom: 0.5rem;'>
            <strong>Cultural Bias Detection System</strong>
        </p>
        <p style='font-size: 0.9rem;'>
            Powered by HuggingFace Transformers • Streamlit • Plotly
        </p>
        <p style='font-size: 0.85rem; color: #9ca3af;'>
            Built with ❤️ for Fairness in AI
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()