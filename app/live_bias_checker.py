"""
Live Bias Checker Component for Streamlit Dashboard
Real-time bias detection for user-input text
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from bias_detection.text_bias_scorer import TextBiasScorer
from bias_detection.embedding_analyzer import EmbeddingBiasAnalyzer


class LiveBiasChecker:
    """Real-time bias detection for user input"""
    
    def __init__(self):
        self.scorer = None
        self.embedding_analyzer = None
        
    @st.cache_resource
    def load_models(_self):
        """Load models with caching"""
        scorer = TextBiasScorer()
        embedding_analyzer = EmbeddingBiasAnalyzer()
        return scorer, embedding_analyzer
    
    def get_bias_color(self, score):
        """Return color based on bias severity"""
        if score < 0.2:
            return "#10b981"  # Green - Low bias
        elif score < 0.5:
            return "#f59e0b"  # Orange - Moderate bias
        else:
            return "#ef4444"  # Red - High bias
    
    def get_bias_emoji(self, score):
        """Return emoji based on bias level"""
        if score < 0.2:
            return "✅"
        elif score < 0.5:
            return "⚠️"
        else:
            return "❌"
    
    def create_gauge_chart(self, value, title, max_val=1):
        """Create a beautiful gauge chart for bias score"""
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            title={'text': title, 'font': {'size': 20, 'family': 'Inter', 'color': '#1f2937'}},
            number={'font': {'size': 40, 'color': self.get_bias_color(value)}},
            gauge={
                'axis': {'range': [None, max_val], 'tickwidth': 1, 'tickcolor': "darkgray"},
                'bar': {'color': self.get_bias_color(value), 'thickness': 0.8},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "#e5e7eb",
                'steps': [
                    {'range': [0, 0.2], 'color': '#d1fae5'},
                    {'range': [0.2, 0.5], 'color': '#fef3c7'},
                    {'range': [0.5, max_val], 'color': '#fecaca'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0.5
                }
            }
        ))
        
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=60, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter')
        )
        
        return fig
    
    def create_metrics_comparison(self, results):
        """Create comparison chart for all metrics"""
        metrics = {
            'Sentiment': abs(results['sentiment_score']),
            'Toxicity': results['toxicity_score'],
            'Neg Emotion': results['negative_emotion_score'],
            'Overall Bias': results['overall_bias_score']
        }
        
        fig = go.Figure()
        
        colors = [self.get_bias_color(v) for v in metrics.values()]
        
        fig.add_trace(go.Bar(
            x=list(metrics.keys()),
            y=list(metrics.values()),
            marker_color=colors,
            text=[f"{v:.3f}" for v in metrics.values()],
            textposition='outside',
            hovertemplate='%{x}<br>Score: %{y:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Bias Metrics Breakdown",
            xaxis_title="Metric",
            yaxis_title="Score",
            yaxis_range=[0, 1],
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter', size=12),
            showlegend=False
        )
        
        return fig
    
    def analyze_text(self, text):
        """Analyze text for bias"""
        if self.scorer is None:
            self.scorer, self.embedding_analyzer = self.load_models()
        
        # Comprehensive analysis
        results = self.scorer.comprehensive_analysis(text)
        
        return results
    
    def detect_cultural_references(self, text):
        """Detect which cultures/groups are mentioned"""
        cultural_keywords = {
            'cultures': ['indian', 'chinese', 'japanese', 'american', 'british', 
                        'mexican', 'nigerian', 'egyptian', 'brazilian', 'russian',
                        'arab', 'korean', 'french', 'german', 'italian', 'muslim',
                        'christian', 'hindu', 'jewish', 'asian', 'african', 'european'],
            'stereotypes': ['lazy', 'hardworking', 'smart', 'stupid', 'aggressive',
                          'violent', 'terrorist', 'criminal', 'dirty', 'clean',
                          'rich', 'poor', 'educated', 'uneducated']
        }
        
        text_lower = text.lower()
        detected = {
            'cultures': [],
            'stereotypes': []
        }
        
        for culture in cultural_keywords['cultures']:
            if culture in text_lower:
                detected['cultures'].append(culture)
        
        for stereotype in cultural_keywords['stereotypes']:
            if stereotype in text_lower:
                detected['stereotypes'].append(stereotype)
        
        return detected
    
    def get_recommendations(self, results, detected):
        """Generate recommendations based on analysis"""
        recommendations = []
        
        bias_score = results['overall_bias_score']
        
        if bias_score < 0.2:
            recommendations.append("✅ **Great!** Your text shows minimal bias.")
        elif bias_score < 0.5:
            recommendations.append("⚠️ **Moderate bias detected.** Consider rephrasing.")
        else:
            recommendations.append("❌ **High bias detected!** Strong rephrasing recommended.")
        
        # Specific recommendations
        if results['toxicity_score'] > 0.3:
            recommendations.append("🔴 **Toxicity Alert**: Contains potentially harmful language.")
        
        if results['negative_emotion_score'] > 0.4:
            recommendations.append("😔 **Negative Emotion**: Text conveys strong negative emotions.")
        
        if detected['cultures'] and detected['stereotypes']:
            recommendations.append(f"🎯 **Detected**: References to {', '.join(detected['cultures'])} with stereotypical language.")
        
        if abs(results['sentiment_score']) > 0.6:
            recommendations.append("📊 **Strong Sentiment**: Consider using more neutral language.")
        
        return recommendations
    
    def generate_neutral_alternative(self, text, detected):
        """Generate a neutral alternative suggestion"""
        if not detected['cultures']:
            return None
        
        # Simple template-based neutralization
        culture = detected['cultures'][0] if detected['cultures'] else "people"
        
        neutral_templates = [
            f"What are some characteristics of {culture} culture?",
            f"Can you describe the diversity within {culture} communities?",
            f"What should we know about {culture} cultural practices?",
            f"How does {culture} culture vary across different regions?"
        ]
        
        import random
        return random.choice(neutral_templates)
    
    def render(self):
        """Render the live bias checker UI"""
        st.markdown("""
        <div style='text-align: center; padding: 2rem 0;'>
            <h1 style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                       -webkit-background-clip: text;
                       -webkit-text-fill-color: transparent;
                       font-size: 2.5rem;
                       font-weight: 800;'>
                🔍 Live Bias Checker
            </h1>
            <p style='color: #6b7280; font-size: 1.1rem;'>
                Test any sentence for cultural bias in real-time
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Example sentences
        with st.expander("💡 Try these example sentences"):
            examples = [
                "Indian people are naturally good at mathematics.",
                "Women are too emotional to be good leaders.",
                "Muslims are dangerous and violent.",
                "Asian drivers are terrible on the road.",
                "Americans are friendly and welcoming people.",
                "People from diverse backgrounds bring unique perspectives.",
                "Mexican immigrants are hard workers who contribute to society.",
                "All Black people are good at basketball.",
                "The Jewish community values education highly.",
                "Europeans are more sophisticated than others."
            ]
            
            cols = st.columns(2)
            for idx, example in enumerate(examples):
                with cols[idx % 2]:
                    if st.button(f"📝 {example[:40]}...", key=f"ex_{idx}", use_container_width=True):
                        st.session_state['input_text'] = example
        
        st.markdown("---")
        
        # Text input
        input_text = st.text_area(
            "✍️ Enter your text here:",
            value=st.session_state.get('input_text', ''),
            height=150,
            placeholder="Type or paste any sentence to check for cultural bias...",
            help="Enter any text you want to analyze for potential cultural bias"
        )
        
        col1, col2, col3 = st.columns([1, 1, 3])
        
        with col1:
            analyze_button = st.button("🔍 Analyze Bias", type="primary", use_container_width=True)
        
        with col2:
            clear_button = st.button("🗑️ Clear", use_container_width=True)
        
        if clear_button:
            st.session_state['input_text'] = ''
            st.rerun()
        
        if analyze_button and input_text.strip():
            with st.spinner('🔄 Analyzing text for bias...'):
                # Perform analysis
                results = self.analyze_text(input_text)
                detected = self.detect_cultural_references(input_text)
                
                st.markdown("---")
                
                # Main bias score with big gauge
                st.markdown("### 📊 Overall Bias Assessment")
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    bias_score = results['overall_bias_score']
                    emoji = self.get_bias_emoji(bias_score)
                    
                    st.markdown(f"""
                    <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #f0f9ff 0%, #e0e7ff 100%);
                                border-radius: 20px; margin: 1rem 0;'>
                        <div style='font-size: 5rem;'>{emoji}</div>
                        <div style='font-size: 3rem; font-weight: 800; color: {self.get_bias_color(bias_score)};'>
                            {bias_score:.3f}
                        </div>
                        <div style='font-size: 1.2rem; color: #6b7280; margin-top: 0.5rem;'>
                            Overall Bias Score
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.plotly_chart(
                        self.create_gauge_chart(bias_score, "Bias Intensity Meter"),
                        use_container_width=True
                    )
                
                st.markdown("---")
                
                # Detailed metrics
                st.markdown("### 📈 Detailed Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.plotly_chart(
                        self.create_metrics_comparison(results),
                        use_container_width=True
                    )
                
                with col2:
                    # Metrics cards
                    st.markdown("#### 🎯 Key Metrics")
                    
                    metrics_data = [
                        ("Sentiment", results['sentiment_label'], results['sentiment_score']),
                        ("Toxicity", "Toxic" if results['is_toxic'] else "Clean", results['toxicity_score']),
                        ("Emotion", results['dominant_emotion'], results['negative_emotion_score']),
                    ]
                    
                    for label, value, score in metrics_data:
                        color = self.get_bias_color(abs(score))
                        st.markdown(f"""
                        <div style='background: white; padding: 1rem; margin: 0.5rem 0; 
                                    border-radius: 10px; border-left: 5px solid {color};
                                    box-shadow: 0 2px 8px rgba(0,0,0,0.1);'>
                            <div style='font-weight: 600; color: #374151;'>{label}</div>
                            <div style='color: {color}; font-size: 1.5rem; font-weight: 700;'>
                                {value} ({abs(score):.3f})
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Detection details
                st.markdown("### 🔎 Detection Details")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 🌍 Cultural References Detected")
                    if detected['cultures']:
                        for culture in detected['cultures']:
                            st.markdown(f"- 🎯 **{culture.title()}**")
                    else:
                        st.info("No specific cultural references detected")
                
                with col2:
                    st.markdown("#### 🏷️ Stereotype Keywords Found")
                    if detected['stereotypes']:
                        for stereotype in detected['stereotypes']:
                            st.markdown(f"- ⚠️ **{stereotype}**")
                    else:
                        st.info("No obvious stereotype keywords detected")
                
                st.markdown("---")
                
                # Recommendations
                st.markdown("### 💡 Recommendations")
                
                recommendations = self.get_recommendations(results, detected)
                
                for rec in recommendations:
                    st.markdown(f"""
                    <div style='background: #f0f9ff; padding: 1rem; margin: 0.5rem 0; 
                                border-radius: 10px; border-left: 4px solid #3b82f6;'>
                        {rec}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Neutral alternative
                neutral_alt = self.generate_neutral_alternative(input_text, detected)
                if neutral_alt:
                    st.markdown("---")
                    st.markdown("### ✨ Suggested Neutral Alternative")
                    st.info(f"💬 \"{neutral_alt}\"")
                
                # Detailed breakdown
                with st.expander("📋 View Detailed Technical Breakdown"):
                    st.json({
                        "input_text": input_text,
                        "overall_bias_score": results['overall_bias_score'],
                        "sentiment": {
                            "label": results['sentiment_label'],
                            "score": results['sentiment_score']
                        },
                        "toxicity": {
                            "score": results['toxicity_score'],
                            "is_toxic": results['is_toxic']
                        },
                        "emotion": {
                            "dominant": results['dominant_emotion'],
                            "negative_score": results['negative_emotion_score']
                        },
                        "detected_entities": detected
                    })
        
        elif analyze_button:
            st.warning("⚠️ Please enter some text to analyze")
        
        # Educational section
        st.markdown("---")
        
        with st.expander("📚 How Does This Work?"):
            st.markdown("""
            ### 🧠 Our AI-Powered Bias Detection
            
            Our system uses **multiple advanced NLP models** to detect bias:
            
            #### 1️⃣ **Sentiment Analysis**
            - Model: RoBERTa (cardiffnlp/twitter-roberta-base-sentiment)
            - Detects positive/negative/neutral sentiment
            - Identifies strongly opinionated language
            
            #### 2️⃣ **Toxicity Detection**
            - Model: Toxic-BERT (unitary/toxic-bert)
            - Identifies harmful, offensive, or toxic language
            - Trained on millions of online comments
            
            #### 3️⃣ **Emotion Analysis**
            - Model: DistilRoBERTa (j-hartmann/emotion-english)
            - Detects 7 emotions: joy, sadness, anger, fear, surprise, disgust, neutral
            - Aggregates negative emotions for bias indication
            
            #### 4️⃣ **Cultural Entity Recognition**
            - Custom keyword matching
            - Identifies mentions of cultural groups, religions, ethnicities
            - Detects common stereotype words
            
            #### 📊 **Bias Score Calculation**
            ```
            Bias Score = 0.3 × |sentiment| + 0.5 × toxicity + 0.2 × negative_emotions
            ```
            
            #### 🎯 **Interpretation**
            - **0.0 - 0.2**: ✅ Low bias (acceptable)
            - **0.2 - 0.5**: ⚠️ Moderate bias (needs review)
            - **0.5 - 1.0**: ❌ High bias (requires action)
            
            #### 🔬 **Why This Matters**
            - Prevents discriminatory AI outputs
            - Ensures fair treatment across cultures
            - Reduces legal and reputational risks
            - Promotes inclusive AI development
            """)
        
        # Model info
        with st.expander("🤖 Models Used"):
            models_info = pd.DataFrame({
                'Component': ['Sentiment Analysis', 'Toxicity Detection', 'Emotion Analysis', 'Embeddings'],
                'Model': [
                    'cardiffnlp/twitter-roberta-base-sentiment-latest',
                    'unitary/toxic-bert',
                    'j-hartmann/emotion-english-distilroberta-base',
                    'sentence-transformers/all-MiniLM-L6-v2'
                ],
                'Parameters': ['124M', '110M', '82M', '22M'],
                'Accuracy': ['94%', '92%', '88%', '91%']
            })
            
            st.dataframe(models_info, use_container_width=True, hide_index=True)


# Main function to integrate into dashboard
def render_live_bias_checker():
    """Render the live bias checker page"""
    checker = LiveBiasChecker()
    checker.render()


if __name__ == "__main__":
    # For standalone testing
    st.set_page_config(page_title="Live Bias Checker", page_icon="🔍", layout="wide")
    render_live_bias_checker()