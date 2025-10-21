"""
Prompt Debiasing and Mitigation Techniques
Phase 5: Bias Mitigation/Resolution
"""

import re
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
from transformers import pipeline
import random


class PromptDebiaser:
    """
    Implements various debiasing techniques:
    1. Prompt rephrasing/neutralization
    2. Counterfactual data augmentation
    3. Post-hoc filtering
    """
    
    def __init__(self):
        self.sensitive_entities = {
            'cultures': [
                'Indian', 'Chinese', 'Japanese', 'American', 'British',
                'Mexican', 'Nigerian', 'Egyptian', 'Brazilian', 'Russian',
                'Arab', 'Korean', 'French', 'German', 'Italian'
            ],
            'religions': [
                'Muslim', 'Christian', 'Hindu', 'Buddhist', 'Jewish',
                'Sikh', 'Islamic', 'Catholic', 'Protestant'
            ],
            'genders': ['man', 'woman', 'male', 'female', 'men', 'women']
        }
        
        # Stereotype keywords to detect
        self.stereotype_keywords = {
            'negative': [
                'lazy', 'aggressive', 'violent', 'stupid', 'dirty',
                'criminal', 'terrorist', 'illegal', 'dangerous', 'inferior',
                'backward', 'primitive', 'uncivilized'
            ],
            'positive_stereotypes': [
                'good at math', 'hardworking', 'naturally talented',
                'born to', 'all are', 'always', 'never'
            ]
        }
        
        # Neutral rephrasing templates
        self.neutral_templates = [
            "What are common characteristics found among {group} individuals?",
            "Can you describe the diversity within {group} culture?",
            "What are some misconceptions about {group} people?",
            "How does {group} culture vary across different regions?",
            "What should we know about {group} cultural practices?"
        ]
    
    def detect_sensitive_content(self, text: str) -> Dict[str, List[str]]:
        """
        Detect sensitive entities in text
        Returns: Dict of detected sensitive words by category
        """
        detected = {
            'cultures': [],
            'religions': [],
            'genders': [],
            'stereotypes': []
        }
        
        text_lower = text.lower()
        
        # Check for cultural references
        for culture in self.sensitive_entities['cultures']:
            if culture.lower() in text_lower:
                detected['cultures'].append(culture)
        
        # Check for religious references
        for religion in self.sensitive_entities['religions']:
            if religion.lower() in text_lower:
                detected['religions'].append(religion)
        
        # Check for gender references
        for gender in self.sensitive_entities['genders']:
            if gender.lower() in text_lower:
                detected['genders'].append(gender)
        
        # Check for stereotype keywords
        for keyword in (self.stereotype_keywords['negative'] + 
                       self.stereotype_keywords['positive_stereotypes']):
            if keyword.lower() in text_lower:
                detected['stereotypes'].append(keyword)
        
        return detected
    
    def neutralize_prompt(self, prompt: str) -> str:
        """
        Rephrase potentially biased prompt to neutral version
        """
        detected = self.detect_sensitive_content(prompt)
        
        # If no sensitive content, return as is
        if not any(detected.values()):
            return prompt
        
        # Extract the main group being discussed
        main_group = None
        if detected['cultures']:
            main_group = detected['cultures'][0]
        elif detected['religions']:
            main_group = detected['religions'][0]
        
        if main_group:
            # Replace with neutral template
            neutral_prompt = random.choice(self.neutral_templates).format(group=main_group)
            return neutral_prompt
        
        # Fallback: remove stereotype keywords
        neutralized = prompt
        for keyword in detected['stereotypes']:
            neutralized = re.sub(
                r'\b' + re.escape(keyword) + r'\b',
                '[neutral term]',
                neutralized,
                flags=re.IGNORECASE
            )
        
        return neutralized
    
    def generate_counterfactuals(self, text: str, n_variants: int = 3) -> List[str]:
        """
        Generate counterfactual variants by swapping cultural references
        Counterfactual Data Augmentation (CDA)
        """
        detected = self.detect_sensitive_content(text)
        
        if not detected['cultures']:
            return [text]  # No culture to swap
        
        original_culture = detected['cultures'][0]
        counterfactuals = []
        
        # Generate variants with different cultures
        alternative_cultures = [
            c for c in self.sensitive_entities['cultures'] 
            if c != original_culture
        ]
        
        selected_cultures = random.sample(
            alternative_cultures,
            min(n_variants, len(alternative_cultures))
        )
        
        for alt_culture in selected_cultures:
            counterfactual = text.replace(original_culture, alt_culture)
            counterfactuals.append(counterfactual)
        
        return counterfactuals
    
    def filter_toxic_responses(self, responses: List[str], 
                               toxicity_threshold: float = 0.5) -> List[str]:
        """
        Post-hoc filtering: Remove toxic responses
        """
        try:
            from src.bias_detection.text_bias_scorer import TextBiasScorer
            
            scorer = TextBiasScorer()
            filtered = []
            
            for response in responses:
                toxicity = scorer.analyze_toxicity(response)
                
                if toxicity['toxicity_score'] < toxicity_threshold:
                    filtered.append(response)
            
            return filtered
        
        except Exception as e:
            print(f"Warning: Could not filter responses: {e}")
            return responses
    
    def debias_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply debiasing to entire dataset
        """
        debiased_rows = []
        
        for idx, row in df.iterrows():
            # Neutralize stereotype
            neutralized = self.neutralize_prompt(row['stereotype'])
            
            # Generate counterfactuals
            counterfactuals = self.generate_counterfactuals(row['stereotype'])
            
            debiased_row = row.copy()
            debiased_row['stereotype_neutralized'] = neutralized
            debiased_row['counterfactuals'] = '|'.join(counterfactuals)
            debiased_row['debiasing_applied'] = True
            
            debiased_rows.append(debiased_row)
        
        return pd.DataFrame(debiased_rows)
    
    def compare_before_after(self, original_df: pd.DataFrame, 
                            debiased_df: pd.DataFrame) -> Dict:
        """
        Compare bias metrics before and after debiasing
        """
        from src.bias_detection.text_bias_scorer import TextBiasScorer
        
        scorer = TextBiasScorer()
        
        # Analyze original
        print("Analyzing original dataset...")
        original_results = scorer.analyze_stereotype_pairs(original_df)
        
        # Analyze debiased
        print("Analyzing debiased dataset...")
        # Create temporary debiased dataset with neutralized stereotypes
        temp_debiased = debiased_df.copy()
        temp_debiased['stereotype'] = temp_debiased['stereotype_neutralized']
        
        debiased_results = scorer.analyze_stereotype_pairs(temp_debiased)
        
        # Calculate improvements
        comparison = {
            'original_bias_score': original_results['stereo_bias_score'].mean(),
            'debiased_bias_score': debiased_results['stereo_bias_score'].mean(),
            'bias_reduction': (
                original_results['stereo_bias_score'].mean() - 
                debiased_results['stereo_bias_score'].mean()
            ),
            'reduction_percentage': (
                (original_results['stereo_bias_score'].mean() - 
                 debiased_results['stereo_bias_score'].mean()) /
                original_results['stereo_bias_score'].mean() * 100
            ),
            'original_toxicity': original_results['stereo_toxicity'].mean(),
            'debiased_toxicity': debiased_results['stereo_toxicity'].mean(),
            'toxicity_reduction': (
                original_results['stereo_toxicity'].mean() - 
                debiased_results['stereo_toxicity'].mean()
            )
        }
        
        return comparison
    
    def generate_mitigation_report(self, comparison: Dict) -> str:
        """
        Generate human-readable mitigation report
        """
        report = []
        report.append("="*70)
        report.append("BIAS MITIGATION REPORT")
        report.append("="*70)
        report.append("")
        
        report.append("📊 BIAS SCORE COMPARISON")
        report.append(f"   Before Debiasing: {comparison['original_bias_score']:.3f}")
        report.append(f"   After Debiasing:  {comparison['debiased_bias_score']:.3f}")
        report.append(f"   Reduction:        {comparison['bias_reduction']:.3f} "
                     f"({comparison['reduction_percentage']:.1f}%)")
        report.append("")
        
        report.append("☠️  TOXICITY COMPARISON")
        report.append(f"   Before Debiasing: {comparison['original_toxicity']:.3f}")
        report.append(f"   After Debiasing:  {comparison['debiased_toxicity']:.3f}")
        report.append(f"   Reduction:        {comparison['toxicity_reduction']:.3f}")
        report.append("")
        
        # Interpretation
        if comparison['reduction_percentage'] > 30:
            status = "✅ Significant improvement achieved!"
        elif comparison['reduction_percentage'] > 15:
            status = "⚠️  Moderate improvement - consider additional techniques"
        else:
            status = "❌ Limited improvement - may need stronger debiasing"
        
        report.append(f"📈 OVERALL ASSESSMENT")
        report.append(f"   {status}")
        report.append("")
        report.append("="*70)
        
        return "\n".join(report)


# Example usage
if __name__ == "__main__":
    debiaser = PromptDebiaser()
    
    # Test prompt neutralization
    biased_prompts = [
        "Are Muslims violent people?",
        "Indian people are naturally good at math.",
        "Chinese people are bad drivers.",
        "People from Mexico are all illegal immigrants."
    ]
    
    print("PROMPT NEUTRALIZATION TEST")
    print("="*70)
    
    for prompt in biased_prompts:
        detected = debiaser.detect_sensitive_content(prompt)
        neutralized = debiaser.neutralize_prompt(prompt)
        
        print(f"\nOriginal:    {prompt}")
        print(f"Detected:    {detected}")
        print(f"Neutralized: {neutralized}")
    
    print("\n" + "="*70)
    print("COUNTERFACTUAL GENERATION TEST")
    print("="*70)
    
    test_text = "Indian people are hardworking and dedicated."
    counterfactuals = debiaser.generate_counterfactuals(test_text, n_variants=3)
    
    print(f"\nOriginal: {test_text}")
    print("Counterfactuals:")
    for i, cf in enumerate(counterfactuals, 1):
        print(f"  {i}. {cf}")