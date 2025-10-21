"""
Streamlit Dashboard for Cultural Bias Detection
Interactive visualization of bias metrics and results
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path
import json


# Page configuration
st.set_page_config(
    page_title="Cultural Bias Detection",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load processed bias detection results"""
    try:
        # Dynamically resolve absolute paths (one level up from dashboard)
        base_dir = Path(__file__).resolve().parent
        project_root = base_dir.parent  # adjust if this file is inside /src/dashboard
        results_dir = project_root / "results"

        results_path = results_dir / "bias_analysis_results.csv"
        metrics_path = results_dir / "bias_metrics.json"
        viz_path = results_dir / "visualization_data.csv"

        # Debug info (optional)
        st.text(f"Looking for results in: {results_dir}")

        if results_path.exists():
            df = pd.read_csv(results_path)
        else:
            st.error(f"Results file not found at: {results_path}")
            return None, None, None

        metrics = None
        if metrics_path.exists():
            with open(metrics_path, 'r', encoding='utf-8') as f:
                metrics = json.load(f)

        viz_df = None
        if viz_path.exists():
            viz_df = pd.read_csv(viz_path)

        return df, metrics, viz_df

    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None



def create_heatmap(df, metric_col, title):
    """Create heatmap of bias scores by culture and bias type"""
    pivot_data = df.pivot_table(
        values=metric_col,
        index='bias_type',
        columns='culture',
        aggfunc='mean'
    )
    
    fig = px.imshow(
        pivot_data,
        labels=dict(x="Cultural Group", y="Bias Type", color=title),
        x=pivot_data.columns,
        y=pivot_data.index,
        color_continuous_scale='RdYlGn_r',
        aspect="auto",
        title=title
    )
    
    fig.update_layout(height=500)
    return fig


def create_comparison_bars(df):
    """Create bar chart comparing stereotype vs anti-stereotype scores"""
    comparison_data = []
    
    for culture in df['culture'].unique():
        culture_df = df[df['culture'] == culture]
        comparison_data.append({
            'Culture': culture,
            'Type': 'Stereotype',
            'Score': culture_df['stereo_bias_score'].mean()
        })
        comparison_data.append({
            'Culture': culture,
            'Type': 'Anti-Stereotype',
            'Score': culture_df['anti_stereo_bias_score'].mean()
        })
        comparison_data.append({
            'Culture': culture,
            'Type': 'Neutral',
            'Score': culture_df['neutral_bias_score'].mean()
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    fig = px.bar(
        comparison_df,
        x='Culture',
        y='Score',
        color='Type',
        barmode='group',
        title='Bias Scores: Stereotype vs Anti-Stereotype vs Neutral',
        color_discrete_map={
            'Stereotype': '#ff6b6b',
            'Anti-Stereotype': '#4ecdc4',
            'Neutral': '#95e1d3'
        }
    )
    
    fig.update_layout(height=500, xaxis_tickangle=-45)
    return fig


def create_tsne_plot(viz_df):
    """Create t-SNE visualization of embeddings"""
    if viz_df is None:
        return None
    
    fig = px.scatter(
        viz_df,
        x='x',
        y='y',
        color='type',
        symbol='culture',
        title='t-SNE Visualization of Sentence Embeddings',
        labels={'x': 't-SNE 1', 'y': 't-SNE 2'},
        color_discrete_map={
            'stereotype': '#ff6b6b',
            'anti-stereotype': '#4ecdc4',
            'neutral': '#95e1d3'
        },
        hover_data=['culture']
    )
    
    fig.update_layout(height=600)
    return fig


def create_metrics_gauge(value, title, threshold_good=0.7, threshold_warning=0.4):
    """Create gauge chart for metrics"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        title={'text': title},
        delta={'reference': threshold_good},
        gauge={
            'axis': {'range': [None, 1]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, threshold_warning], 'color': "lightcoral"},
                {'range': [threshold_warning, threshold_good], 'color': "lightyellow"},
                {'range': [threshold_good, 1], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': threshold_good
            }
        }
    ))
    
    fig.update_layout(height=250)
    return fig


def create_culture_radar(df):
    """Create radar chart showing bias across cultures"""
    cultures = df['culture'].unique()[:8]  # Limit to 8 for readability
    
    fig = go.Figure()
    
    for culture in cultures:
        culture_df = df[df['culture'] == culture]
        
        values = [
            culture_df['stereo_sentiment'].mean(),
            culture_df['stereo_toxicity'].mean(),
            culture_df['stereo_neg_emotion'].mean(),
            culture_df['stereo_bias_score'].mean()
        ]
        
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],  # Close the polygon
            theta=['Sentiment', 'Toxicity', 'Neg Emotion', 'Bias Score', 'Sentiment'],
            name=culture,
            fill='toself'
        ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="Bias Dimensions by Culture (Radar Chart)",
        height=500
    )
    
    return fig


def main():
    # Header
    st.markdown('<h1 class="main-header">🌍 Cultural Bias Detection Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("### Analyzing and Visualizing Bias in Language Model Outputs")
    
    # Load data
    df, metrics, viz_df = load_data()
    
    if df is None:
        st.warning("⚠️ No data available. Please run the analysis pipeline first:")
        st.code("python main.py", language="bash")
        return
    
    # Sidebar
    st.sidebar.header("🎛️ Controls")
    
    # Culture filter
    all_cultures = sorted(df['culture'].unique())
    selected_cultures = st.sidebar.multiselect(
        "Select Cultures",
        all_cultures,
        default=all_cultures[:5]
    )
    
    # Bias type filter
    all_bias_types = sorted(df['bias_type'].unique())
    selected_bias_types = st.sidebar.multiselect(
        "Select Bias Types",
        all_bias_types,
        default=all_bias_types
    )
    
    # Filter data
    filtered_df = df[
        (df['culture'].isin(selected_cultures)) &
        (df['bias_type'].isin(selected_bias_types))
    ]
    
    # Metrics Overview
    st.header("📊 Key Metrics Overview")
    
    if metrics:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            ss = metrics['stereotype_score']['overall_ss']
            st.metric(
                "Stereotype Score (SS)",
                f"{ss:.3f}",
                help="Difference in sentiment between stereotypes and anti-stereotypes"
            )
            st.caption(metrics['stereotype_score']['interpretation'])
        
        with col2:
            ba = metrics['bias_amplification']['normalized_ba']
            st.metric(
                "Bias Amplification (BA)",
                f"{ba:.3f}",
                help="How much the model amplifies stereotypes"
            )
            st.caption(metrics['bias_amplification']['interpretation'])
        
        with col3:
            td = metrics['toxicity_differential']['overall_td']
            st.metric(
                "Toxicity Differential (TD)",
                f"{td:.3f}",
                help="Difference in toxicity between stereotypes and neutral"
            )
            st.caption(metrics['toxicity_differential']['interpretation'])
        
        with col4:
            fi = metrics['fairness_index']['overall_fi']
            st.metric(
                "Fairness Index (FI)",
                f"{fi:.3f}",
                help="Fairness across different cultural groups"
            )
            st.caption(metrics['fairness_index']['interpretation'])
    
    st.markdown("---")
    
    # Visualizations
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", "🔥 Heatmaps", "📈 Comparisons", "🎯 Embeddings", "📋 Data Explorer"
    ])
    
    with tab1:
        st.subheader("Summary Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"""
            **Dataset Overview**
            - Total Samples: {len(filtered_df)}
            - Cultures Analyzed: {len(selected_cultures)}
            - Bias Types: {len(selected_bias_types)}
            - Bias Detected: {filtered_df['bias_detected'].sum()} samples
            """)
            
            # Bias detection rate by culture
            bias_rate = filtered_df.groupby('culture')['bias_detected'].mean().sort_values(ascending=False)
            fig_bias_rate = px.bar(
                x=bias_rate.index,
                y=bias_rate.values,
                labels={'x': 'Culture', 'y': 'Bias Detection Rate'},
                title='Bias Detection Rate by Culture',
                color=bias_rate.values,
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig_bias_rate, use_container_width=True)
        
        with col2:
            # Gauge charts for metrics
            if metrics:
                st.plotly_chart(
                    create_metrics_gauge(fi, "Fairness Index", 0.7, 0.5),
                    use_container_width=True
                )
                
                st.plotly_chart(
                    create_metrics_gauge(
                        1 - ba, "Fairness (1 - BA)", 0.6, 0.4
                    ),
                    use_container_width=True
                )
    
    with tab2:
        st.subheader("🔥 Bias Heatmaps")
        
        # Sentiment difference heatmap
        st.plotly_chart(
            create_heatmap(filtered_df, 'sentiment_difference', 'Sentiment Difference (Stereotype - Anti-Stereotype)'),
            use_container_width=True
        )
        
        # Toxicity difference heatmap
        st.plotly_chart(
            create_heatmap(filtered_df, 'toxicity_difference', 'Toxicity Difference'),
            use_container_width=True
        )
    
    with tab3:
        st.subheader("📈 Stereotype vs Anti-Stereotype Comparison")
        
        # Bar chart comparison
        st.plotly_chart(
            create_comparison_bars(filtered_df),
            use_container_width=True
        )
        
        # Radar chart
        st.plotly_chart(
            create_culture_radar(filtered_df),
            use_container_width=True
        )
    
    with tab4:
        st.subheader("🎯 Embedding Space Visualization")
        
        if viz_df is not None:
            # t-SNE plot
            tsne_fig = create_tsne_plot(viz_df)
            if tsne_fig:
                st.plotly_chart(tsne_fig, use_container_width=True)
                
                st.info("""
                **Interpretation:**
                - Points closer together are semantically similar
                - Stereotypes clustering separately from anti-stereotypes indicates bias
                - Overlapping clusters suggest less bias
                """)
        else:
            st.warning("Embedding visualization data not available")
    
    with tab5:
        st.subheader("📋 Raw Data Explorer")
        
        # Display options
        show_columns = st.multiselect(
            "Select columns to display",
            filtered_df.columns.tolist(),
            default=['culture', 'bias_type', 'stereotype', 'anti_stereotype', 
                    'sentiment_difference', 'toxicity_difference', 'bias_detected']
        )
        
        # Search
        search_term = st.text_input("Search in data:", "")
        
        display_df = filtered_df[show_columns]
        
        if search_term:
            mask = display_df.astype(str).apply(
                lambda x: x.str.contains(search_term, case=False, na=False)
            ).any(axis=1)
            display_df = display_df[mask]
        
        st.dataframe(display_df, use_container_width=True, height=400)
        
        # Download button
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data",
            data=csv,
            file_name="bias_detection_filtered.csv",
            mime="text/csv"
        )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>Cultural Bias Detection System | Built with Streamlit & HuggingFace Transformers</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()