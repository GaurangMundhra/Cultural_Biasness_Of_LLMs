"""
Text-based Bias Scoring using NLP techniques
Sentiment analysis, toxicity detection, and emotion scoring
"""

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from typing import Dict, List
import pandas as pd
from tqdm import tqdm


class TextBiasScorer:
    def __init__(self, device: str = None):
        """
        Initialize bias scoring models
        Args:
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        if device is None:
            self.device = 0 if torch.cuda.is_available() else -1
        else:
            self.device = 0 if device == 'cuda' else -1
        
        print(f"Loading models on device: {'GPU' if self.device == 0 else 'CPU'}")
        
        # Sentiment analysis model
        print("Loading sentiment model...")
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            device=self.device
        )
        
        # Toxicity detection model
        print("Loading toxicity model...")
        self.toxicity_model = AutoModelForSequenceClassification.from_pretrained(
            "unitary/toxic-bert"
        )
        self.toxicity_tokenizer = AutoTokenizer.from_pretrained("unitary/toxic-bert")
        
        if self.device == 0:
            self.toxicity_model = self.toxicity_model.cuda()
        
        # Emotion detection model
        print("Loading emotion model...")
        self.emotion_analyzer = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            device=self.device,
            top_k=None
        )
        
        print("All models loaded successfully!")
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of text
        Returns: {'label': 'positive'/'negative'/'neutral', 'score': float}
        """
        try:
            result = self.sentiment_analyzer(text[:512])[0]
            
            # Convert to normalized score: -1 (negative) to +1 (positive)
            if result['label'] == 'positive':
                sentiment_score = result['score']
            elif result['label'] == 'negative':
                sentiment_score = -result['score']
            else:  # neutral
                sentiment_score = 0.0
            
            return {
                'label': result['label'],
                'score': result['score'],
                'sentiment_score': sentiment_score
            }
        except Exception as e:
            print(f"Error in sentiment analysis: {e}")
            return {'label': 'neutral', 'score': 0.5, 'sentiment_score': 0.0}
    
    def analyze_toxicity(self, text: str) -> Dict[str, float]:
        """
        Analyze toxicity of text
        Returns: {'toxicity_score': float (0-1)}
        """
        try:
            inputs = self.toxicity_tokenizer(
                text[:512],
                return_tensors="pt",
                truncation=True,
                padding=True
            )
            
            if self.device == 0:
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.toxicity_model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                toxicity_score = predictions[:, 1].item()  # Toxic class probability
            
            return {
                'toxicity_score': toxicity_score,
                'is_toxic': toxicity_score > 0.5
            }
        except Exception as e:
            print(f"Error in toxicity analysis: {e}")
            return {'toxicity_score': 0.0, 'is_toxic': False}
    
    def analyze_emotion(self, text: str) -> Dict[str, float]:
        """
        Analyze emotions in text
        Returns: {'emotion': str, 'scores': Dict[str, float]}
        """
        try:
            results = self.emotion_analyzer(text[:512])[0]
            
            emotion_scores = {item['label']: item['score'] for item in results}
            dominant_emotion = max(emotion_scores, key=emotion_scores.get)
            
            return {
                'dominant_emotion': dominant_emotion,
                'emotion_scores': emotion_scores,
                'negative_emotion_score': sum(
                    emotion_scores.get(e, 0) 
                    for e in ['anger', 'disgust', 'fear', 'sadness']
                )
            }
        except Exception as e:
            print(f"Error in emotion analysis: {e}")
            return {'dominant_emotion': 'neutral', 'emotion_scores': {}, 'negative_emotion_score': 0.0}
    
    def comprehensive_analysis(self, text: str) -> Dict[str, any]:
        """
        Perform complete bias analysis on text
        """
        sentiment = self.analyze_sentiment(text)
        toxicity = self.analyze_toxicity(text)
        emotion = self.analyze_emotion(text)
        
        return {
            'text': text,
            'sentiment_label': sentiment['label'],
            'sentiment_score': sentiment['sentiment_score'],
            'toxicity_score': toxicity['toxicity_score'],
            'is_toxic': toxicity['is_toxic'],
            'dominant_emotion': emotion['dominant_emotion'],
            'negative_emotion_score': emotion['negative_emotion_score'],
            'overall_bias_score': self._calculate_bias_score(sentiment, toxicity, emotion)
        }
    
    def _calculate_bias_score(self, sentiment: Dict, toxicity: Dict, emotion: Dict) -> float:
        """
        Calculate overall bias score combining all metrics
        Range: 0 (no bias) to 1 (high bias)
        """
        # Weighted combination
        bias_score = (
            0.3 * abs(sentiment['sentiment_score']) +  # Strong sentiment (positive or negative)
            0.5 * toxicity['toxicity_score'] +          # High weight on toxicity
            0.2 * emotion['negative_emotion_score']     # Negative emotions
        )
        
        return min(bias_score, 1.0)
    
    def analyze_stereotype_pairs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze stereotype vs anti-stereotype pairs
        """
        results = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing pairs"):
            # Analyze stereotype
            stereo_analysis = self.comprehensive_analysis(row['stereotype'])
            
            # Analyze anti-stereotype
            anti_stereo_analysis = self.comprehensive_analysis(row['anti_stereotype'])
            
            # Analyze neutral
            neutral_analysis = self.comprehensive_analysis(row['neutral'])
            
            # Calculate differences
            sentiment_diff = stereo_analysis['sentiment_score'] - anti_stereo_analysis['sentiment_score']
            toxicity_diff = stereo_analysis['toxicity_score'] - anti_stereo_analysis['toxicity_score']
            
            results.append({
                'culture': row['culture'],
                'bias_type': row['bias_type'],
                'stereotype': row['stereotype'],
                'anti_stereotype': row['anti_stereotype'],
                'neutral': row['neutral'],
                
                # Stereotype scores
                'stereo_sentiment': stereo_analysis['sentiment_score'],
                'stereo_toxicity': stereo_analysis['toxicity_score'],
                'stereo_neg_emotion': stereo_analysis['negative_emotion_score'],
                'stereo_bias_score': stereo_analysis['overall_bias_score'],
                
                # Anti-stereotype scores
                'anti_stereo_sentiment': anti_stereo_analysis['sentiment_score'],
                'anti_stereo_toxicity': anti_stereo_analysis['toxicity_score'],
                'anti_stereo_neg_emotion': anti_stereo_analysis['negative_emotion_score'],
                'anti_stereo_bias_score': anti_stereo_analysis['overall_bias_score'],
                
                # Neutral scores
                'neutral_sentiment': neutral_analysis['sentiment_score'],
                'neutral_toxicity': neutral_analysis['toxicity_score'],
                'neutral_bias_score': neutral_analysis['overall_bias_score'],
                
                # Differences (key metrics)
                'sentiment_difference': sentiment_diff,
                'toxicity_difference': toxicity_diff,
                'bias_detected': abs(sentiment_diff) > 0.3 or toxicity_diff > 0.2
            })
        
        return pd.DataFrame(results)


# Example usage
if __name__ == "__main__":
    scorer = TextBiasScorer()
    
    # Test with sample texts
    texts = [
        "Indian people are hardworking and intelligent.",
        "Chinese people are bad drivers.",
        "Americans are friendly and welcoming."
    ]
    
    for text in texts:
        result = scorer.comprehensive_analysis(text)
        print(f"\nText: {text}")
        print(f"Sentiment: {result['sentiment_label']} ({result['sentiment_score']:.3f})")
        print(f"Toxicity: {result['toxicity_score']:.3f}")
        print(f"Emotion: {result['dominant_emotion']}")
        print(f"Overall Bias Score: {result['overall_bias_score']:.3f}")