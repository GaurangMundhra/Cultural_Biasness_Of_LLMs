from pathlib import Path

config_content = """
'''
Configuration file for Cultural Bias Detection
'''

CONFIG = {
    # Model configurations
    'models': {
        'sentiment': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
        'toxicity': 'unitary/toxic-bert',
        'embedding': 'sentence-transformers/all-MiniLM-L6-v2',
    },
    
    # Processing parameters
    'batch_size': 32,
    'device': 'auto',  # 'cuda', 'cpu', or 'auto'
    
    # Paths
    'data_dir': 'data',
    'output_dir': 'results',
    'models_dir': 'models',
    
    # Dataset generation
    'n_stereotype_pairs': 1000,
    'n_completion_prompts': 500,
    'n_qa_prompts': 500,
    'n_sentiment_prompts': 500,
    
    # Bias thresholds
    'bias_thresholds': {
        'sentiment_diff': 0.3,
        'toxicity_diff': 0.2,
        'embedding_distance': 0.3,
    },
    
    # Visualization
    'viz_params': {
        'tsne_perplexity': 30,
        'pca_components': 2,
        'plot_height': 600,
    }
}
"""

print("\nCreating config.py...")
config_path = Path('src/utils/config.py')
with open(config_path, 'w') as f:
    f.write(config_content.strip())
print(f"  ✓ Created: {config_path}")
