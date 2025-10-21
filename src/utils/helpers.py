from pathlib import Path

def create_project_structure():
    # Create helpers.py template
    helpers_content = """
'''
Utility helper functions
'''

import pandas as pd
import numpy as np
from pathlib import Path
import json


def load_dataset(filepath: str) -> pd.DataFrame:
    '''Load dataset from file'''
    path = Path(filepath)
    
    if path.suffix == '.csv':
        return pd.read_csv(filepath)
    elif path.suffix == '.json':
        return pd.read_json(filepath)
    elif path.suffix == '.xlsx':
        return pd.read_excel(filepath)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def save_results(data, filepath: str):
    '''Save results to file'''
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if isinstance(data, pd.DataFrame):
        if path.suffix == '.csv':
            data.to_csv(filepath, index=False)
        elif path.suffix == '.json':
            data.to_json(filepath, orient='records', indent=2)
        elif path.suffix == '.xlsx':
            data.to_excel(filepath, index=False)
    elif isinstance(data, dict):
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")


def calculate_percentage_change(original: float, new: float) -> float:
    '''Calculate percentage change'''
    if original == 0:
        return 0.0
    return ((new - original) / abs(original)) * 100


def format_metric(value: float, metric_type: str = 'default') -> str:
    '''Format metric for display'''
    if metric_type == 'percentage':
        return f"{value:.1f}%"
    elif metric_type == 'score':
        return f"{value:.3f}"
    else:
        return f"{value:.2f}"
"""

    print("\nCreating helpers.py...")
    helpers_path = Path('src/utils/helpers.py')
    helpers_path.parent.mkdir(parents=True, exist_ok=True)
    with open(helpers_path, 'w') as f:
        f.write(helpers_content.strip())
    print(f"  ✓ Created: {helpers_path}")
    
    print("\n" + "="*70)
    print("✅ Project structure created successfully!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Create and activate virtual environment:")
    print("     python -m venv venv")
    print("     source venv/bin/activate  (Linux/Mac)")
    print("     venv\\Scripts\\activate     (Windows)")
    print("\n  2. Install dependencies:")
    print("     pip install -r requirements.txt")
    print("\n  3. Run the pipeline:")
    print("     python main.py")
    print("\n  4. Launch dashboard:")
    print("     streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    create_project_structure()
