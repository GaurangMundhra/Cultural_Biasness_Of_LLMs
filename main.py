"""
Main Pipeline for Cultural Bias Detection
Orchestrates all phases from data generation to visualization
"""

import sys
from pathlib import Path
import pandas as pd
import json
import argparse
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data_preparation.dataset_generator import CulturalBiasDatasetGenerator
from bias_detection.text_bias_scorer import TextBiasScorer
from bias_detection.embedding_analyzer import EmbeddingBiasAnalyzer
from metrics.bias_metrics import BiasMetricsCalculator


class BiasDetectionPipeline:
    """Main pipeline orchestrating all bias detection phases"""
    
    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.data_dir = Path("data")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        print("="*70)
        print("CULTURAL BIAS DETECTION PIPELINE")
        print("="*70)
    
    def phase1_generate_dataset(self, force_regenerate: bool = False):
        """Phase 1: Generate synthetic dataset"""
        print("\n📊 PHASE 1: DATASET PREPARATION")
        print("-" * 70)
        
        synthetic_dir = self.data_dir / "synthetic"
        metadata_file = synthetic_dir / "metadata.json"
        
        if metadata_file.exists() and not force_regenerate:
            print("✓ Dataset already exists. Use --regenerate to create new dataset.")
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            print(f"  Loaded existing dataset: {metadata['total_samples']} samples")
            return
        
        print("Generating synthetic cultural bias dataset...")
        generator = CulturalBiasDatasetGenerator(output_dir=str(synthetic_dir))
        datasets = generator.generate_full_dataset()
        generator.save_datasets(datasets)
        
        print("✓ Phase 1 Complete!")
    
    def phase2_detect_bias(self):
        """Phase 2: Detect bias using multiple methods"""
        print("\n🔍 PHASE 2: BIAS DETECTION")
        print("-" * 70)
        
        # Load stereotype pairs dataset
        dataset_path = self.data_dir / "synthetic" / "stereotype_pairs.csv"
        
        if not dataset_path.exists():
            print("❌ Error: Dataset not found. Run Phase 1 first.")
            return None
        
        df = pd.read_csv(dataset_path)
        print(f"Loaded {len(df)} stereotype pairs")
        
        # Method 1: Text-based bias scoring
        print("\n🔹 Method 1: Text-based Bias Scoring")
        scorer = TextBiasScorer()
        text_results = scorer.analyze_stereotype_pairs(df)
        
        # Method 2: Embedding analysis
        print("\n🔹 Method 2: Embedding-based Analysis")
        embedding_analyzer = EmbeddingBiasAnalyzer()
        embedding_results = embedding_analyzer.analyze_stereotype_separation(df)
        
        # Combine results
        combined_results = text_results.copy()
        
        # Add embedding metrics
        for col in ['stereo_to_neutral_dist', 'anti_stereo_to_neutral_dist', 
                    'stereo_to_anti_stereo_dist', 'embedding_bias_detected']:
            combined_results[col] = embedding_results[col]
        
        # Save results
        output_file = self.output_dir / "bias_analysis_results.csv"
        combined_results.to_csv(output_file, index=False)
        print(f"\n✓ Bias detection complete! Results saved to {output_file}")
        
        # Generate visualization data
        print("\nPreparing visualization data...")
        viz_df = embedding_analyzer.prepare_visualization_data(df)
        viz_file = self.output_dir / "visualization_data.csv"
        viz_df.to_csv(viz_file, index=False)
        print(f"✓ Visualization data saved to {viz_file}")
        
        return combined_results
    
    def phase3_calculate_metrics(self, results_df: pd.DataFrame):
        """Phase 3: Calculate comprehensive bias metrics"""
        print("\n📈 PHASE 3: METRICS CALCULATION")
        print("-" * 70)
        
        calculator = BiasMetricsCalculator()
        metrics = calculator.calculate_all_metrics(results_df)
        
        # Statistical significance
        print("\nCalculating statistical significance...")
        significance = calculator.calculate_statistical_significance(results_df)
        
        # Add significance to metrics
        metrics['statistical_tests'] = significance
        
        # Generate summary report
        print("\n" + calculator.generate_summary_report())
        
        # Save metrics
        metrics_file = self.output_dir / "bias_metrics.json"
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            elif hasattr(obj, 'item'):  # numpy types
                return obj.item()
            return obj
        
        metrics_serializable = convert_types(metrics)
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics_serializable, f, indent=2)
        
        print(f"\n✓ Metrics saved to {metrics_file}")
        
        return metrics
    
    def phase4_generate_report(self, metrics: dict):
        """Phase 4: Generate comprehensive HTML report"""
        print("\n📄 PHASE 4: REPORT GENERATION")
        print("-" * 70)
        
        report_file = self.output_dir / "bias_detection_report.html"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Cultural Bias Detection Report</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .header {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 30px;
                    border-radius: 10px;
                    text-align: center;
                }}
                .metric-card {{
                    background: white;
                    padding: 20px;
                    margin: 20px 0;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .metric-value {{
                    font-size: 2em;
                    font-weight: bold;
                    color: #667eea;
                }}
                .interpretation {{
                    color: #666;
                    margin-top: 10px;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                th, td {{
                    padding: 12px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                th {{
                    background-color: #667eea;
                    color: white;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🌍 Cultural Bias Detection Report</h1>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="metric-card">
                <h2>📊 Key Metrics</h2>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                        <th>Interpretation</th>
                    </tr>
                    <tr>
                        <td>Stereotype Score (SS)</td>
                        <td class="metric-value">{metrics['stereotype_score']['overall_ss']:.3f}</td>
                        <td class="interpretation">{metrics['stereotype_score']['interpretation']}</td>
                    </tr>
                    <tr>
                        <td>Bias Amplification (BA)</td>
                        <td class="metric-value">{metrics['bias_amplification']['normalized_ba']:.3f}</td>
                        <td class="interpretation">{metrics['bias_amplification']['interpretation']}</td>
                    </tr>
                    <tr>
                        <td>Toxicity Differential (TD)</td>
                        <td class="metric-value">{metrics['toxicity_differential']['overall_td']:.3f}</td>
                        <td class="interpretation">{metrics['toxicity_differential']['interpretation']}</td>
                    </tr>
                    <tr>
                        <td>Fairness Index (FI)</td>
                        <td class="metric-value">{metrics['fairness_index']['overall_fi']:.3f}</td>
                        <td class="interpretation">{metrics['fairness_index']['interpretation']}</td>
                    </tr>
                </table>
            </div>
            
            <div class="metric-card">
                <h2>🎯 Recommendations</h2>
                <ul>
                    <li>Review cultures with highest bias scores for targeted debiasing</li>
                    <li>Implement prompt debiasing techniques for problematic categories</li>
                    <li>Consider counterfactual data augmentation for underrepresented groups</li>
                    <li>Monitor fairness index to ensure equitable treatment across cultures</li>
                </ul>
            </div>
            
            <div class="metric-card">
                <h2>📈 Next Steps</h2>
                <p>Launch the interactive dashboard for detailed analysis:</p>
                <code style="background: #f0f0f0; padding: 10px; display: block; margin: 10px 0;">
                    streamlit run app/streamlit_app.py
                </code>
            </div>
        </body>
        </html>
        """
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        print(f"✓ HTML report generated: {report_file}")
    
    def run_full_pipeline(self, regenerate_data: bool = False):
        """Run complete pipeline from start to finish"""
        start_time = datetime.now()
        
        try:
            # Phase 1: Dataset Generation
            self.phase1_generate_dataset(force_regenerate=regenerate_data)
            
            # Phase 2: Bias Detection
            results = self.phase2_detect_bias()
            
            if results is None:
                print("\n❌ Pipeline failed at Phase 2")
                return
            
            # Phase 3: Metrics Calculation
            metrics = self.phase3_calculate_metrics(results)
            
            # Phase 4: Report Generation
            self.phase4_generate_report(metrics)
            
            # Summary
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            print("\n" + "="*70)
            print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*70)
            print(f"Total execution time: {duration:.2f} seconds")
            print(f"\nResults saved in: {self.output_dir}/")
            print("\n🚀 Next steps:")
            print("  1. View HTML report: open results/bias_detection_report.html")
            print("  2. Launch dashboard: streamlit run app/streamlit_app.py")
            print("  3. Explore data: check results/bias_analysis_results.csv")
            
        except Exception as e:
            print(f"\n❌ Error in pipeline: {e}")
            import traceback
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Cultural Bias Detection Pipeline")
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Force regenerate dataset even if it exists"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    pipeline = BiasDetectionPipeline(output_dir=args.output_dir)
    pipeline.run_full_pipeline(regenerate_data=args.regenerate)


if __name__ == "__main__":
    main()