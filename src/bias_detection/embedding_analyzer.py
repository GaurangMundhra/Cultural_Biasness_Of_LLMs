"""
Embedding-based Bias Detection
Uses sentence embeddings to measure semantic distance and clustering
"""

import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
from typing import List, Dict, Tuple
from tqdm import tqdm


class EmbeddingBiasAnalyzer:
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        """
        Initialize embedding model
        Args:
            model_name: HuggingFace model for sentence embeddings
        """
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device)
        print(f"Model loaded on {self.device}")
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Get sentence embeddings for list of texts
        """
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32,
            convert_to_numpy=True
        )
        return embeddings
    
    def calculate_cosine_distance(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Calculate cosine distance between two embeddings
        Returns: distance (0 = identical, 2 = opposite)
        """
        similarity = cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0][0]
        distance = 1 - similarity
        return distance
    
    def analyze_stereotype_separation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze how well stereotypes and anti-stereotypes are separated
        """
        print("Generating embeddings for stereotypes...")
        stereo_embeddings = self.get_embeddings(df['stereotype'].tolist())
        
        print("Generating embeddings for anti-stereotypes...")
        anti_stereo_embeddings = self.get_embeddings(df['anti_stereotype'].tolist())
        
        print("Generating embeddings for neutral statements...")
        neutral_embeddings = self.get_embeddings(df['neutral'].tolist())
        
        results = []
        
        for idx in tqdm(range(len(df)), desc="Calculating distances"):
            stereo_emb = stereo_embeddings[idx]
            anti_stereo_emb = anti_stereo_embeddings[idx]
            neutral_emb = neutral_embeddings[idx]
            
            # Calculate distances
            stereo_to_neutral = self.calculate_cosine_distance(stereo_emb, neutral_emb)
            anti_stereo_to_neutral = self.calculate_cosine_distance(anti_stereo_emb, neutral_emb)
            stereo_to_anti_stereo = self.calculate_cosine_distance(stereo_emb, anti_stereo_emb)
            
            # Bias detected if stereotype is closer to neutral than anti-stereotype
            # Or if stereotype and anti-stereotype are too similar (not well separated)
            bias_indicator = stereo_to_neutral < anti_stereo_to_neutral
            separation_score = stereo_to_anti_stereo
            
            results.append({
                'culture': df.iloc[idx]['culture'],
                'bias_type': df.iloc[idx]['bias_type'],
                'stereo_to_neutral_dist': stereo_to_neutral,
                'anti_stereo_to_neutral_dist': anti_stereo_to_neutral,
                'stereo_to_anti_stereo_dist': separation_score,
                'distance_difference': abs(stereo_to_neutral - anti_stereo_to_neutral),
                'embedding_bias_detected': bias_indicator,
                'poorly_separated': separation_score < 0.3  # Too similar
            })
        
        result_df = pd.DataFrame(results)
        
        # Store embeddings for visualization
        self.stereo_embeddings = stereo_embeddings
        self.anti_stereo_embeddings = anti_stereo_embeddings
        self.neutral_embeddings = neutral_embeddings
        
        return result_df
    
    def cluster_by_culture(self, df: pd.DataFrame, embeddings: np.ndarray) -> Dict:
        """
        Analyze how embeddings cluster by cultural group
        """
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        
        # Group embeddings by culture
        culture_embeddings = {}
        for culture in df['culture'].unique():
            mask = df['culture'] == culture
            culture_embeddings[culture] = embeddings[mask]
        
        # Calculate intra-culture similarity (how similar are statements about same culture)
        culture_cohesion = {}
        for culture, embs in culture_embeddings.items():
            if len(embs) > 1:
                similarities = cosine_similarity(embs)
                # Average similarity excluding diagonal
                mask = ~np.eye(similarities.shape[0], dtype=bool)
                avg_similarity = similarities[mask].mean()
                culture_cohesion[culture] = avg_similarity
        
        # Calculate inter-culture similarity (how different are cultures from each other)
        cultures = list(culture_embeddings.keys())
        inter_culture_sim = np.zeros((len(cultures), len(cultures)))
        
        for i, culture1 in enumerate(cultures):
            for j, culture2 in enumerate(cultures):
                if i != j:
                    emb1_mean = culture_embeddings[culture1].mean(axis=0)
                    emb2_mean = culture_embeddings[culture2].mean(axis=0)
                    sim = cosine_similarity(emb1_mean.reshape(1, -1), 
                                          emb2_mean.reshape(1, -1))[0][0]
                    inter_culture_sim[i][j] = sim
        
        return {
            'culture_cohesion': culture_cohesion,
            'inter_culture_similarity': inter_culture_sim,
            'culture_labels': cultures
        }
    
    def reduce_dimensions_tsne(self, embeddings: np.ndarray, perplexity: int = 30) -> np.ndarray:
        """
        Reduce embeddings to 2D using t-SNE for visualization
        """
        print("Performing t-SNE dimensionality reduction...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        embeddings_2d = tsne.fit_transform(embeddings)
        return embeddings_2d
    
    def reduce_dimensions_pca(self, embeddings: np.ndarray, n_components: int = 2) -> np.ndarray:
        """
        Reduce embeddings to 2D using PCA for visualization
        """
        print("Performing PCA dimensionality reduction...")
        pca = PCA(n_components=n_components, random_state=42)
        embeddings_2d = pca.fit_transform(embeddings)
        return embeddings_2d
    
    def prepare_visualization_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for t-SNE/PCA visualization
        """
        # Combine all embeddings
        all_embeddings = np.vstack([
            self.stereo_embeddings,
            self.anti_stereo_embeddings,
            self.neutral_embeddings
        ])
        
        # Create labels
        labels = (
            ['stereotype'] * len(self.stereo_embeddings) +
            ['anti-stereotype'] * len(self.anti_stereo_embeddings) +
            ['neutral'] * len(self.neutral_embeddings)
        )
        
        cultures = df['culture'].tolist() * 3
        
        # Reduce dimensions
        embeddings_2d = self.reduce_dimensions_tsne(all_embeddings)
        
        viz_df = pd.DataFrame({
            'x': embeddings_2d[:, 0],
            'y': embeddings_2d[:, 1],
            'type': labels,
            'culture': cultures
        })
        
        return viz_df
    
    def calculate_bias_amplification(self, df: pd.DataFrame, 
                                     embeddings_dict: Dict[str, np.ndarray]) -> Dict:
        """
        Calculate bias amplification: how much model amplifies stereotypes
        """
        results = {}
        
        for culture in df['culture'].unique():
            culture_mask = df['culture'] == culture
            
            stereo_embs = embeddings_dict['stereotype'][culture_mask]
            anti_stereo_embs = embeddings_dict['anti_stereotype'][culture_mask]
            
            # Calculate mean embeddings
            stereo_mean = stereo_embs.mean(axis=0)
            anti_stereo_mean = anti_stereo_embs.mean(axis=0)
            
            # Amplification: distance between stereotype and anti-stereotype centers
            amplification = self.calculate_cosine_distance(stereo_mean, anti_stereo_mean)
            
            results[culture] = {
                'bias_amplification': amplification,
                'highly_amplified': amplification > 0.5
            }
        
        return results


# Example usage
if __name__ == "__main__":
    # Load sample data
    analyzer = EmbeddingBiasAnalyzer()
    
    # Test with sample sentences
    sentences = [
        "Indian people are hardworking.",
        "Indian people are lazy.",
        "People from India have diverse characteristics.",
        "Chinese people are good at math.",
        "Chinese people are bad drivers.",
        "Chinese culture is rich and diverse."
    ]
    
    embeddings = analyzer.get_embeddings(sentences)
    print(f"\nGenerated embeddings shape: {embeddings.shape}")
    
    # Calculate distances
    for i in range(0, len(sentences), 3):
        stereo = embeddings[i]
        anti = embeddings[i+1]
        neutral = embeddings[i+2]
        
        dist = analyzer.calculate_cosine_distance(stereo, anti)
        print(f"\nDistance between stereotype and anti-stereotype: {dist:.3f}")
        print(f"Stereotype to neutral: {analyzer.calculate_cosine_distance(stereo, neutral):.3f}")
        print(f"Anti-stereotype to neutral: {analyzer.calculate_cosine_distance(anti, neutral):.3f}")