#!/usr/bin/env python3
"""
Few-Shot Evaluation for CHQ-Summ
Supports 0-shot, 2-shot, and 5-shot evaluation
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import everything from evaluate module
from evaluation.evaluate import *


class FewShotExperimentRunner(ExperimentRunner):
    """Extended experiment runner with few-shot support."""
    
    def run_all(self, test_data, train_data, num_samples=None, save_preds=True, few_shot_configs=None):
        """Run experiments with configurable few-shot settings."""
        
        methods = ['standard', 'chain_of_density', 'hierarchical', 'element_aware']
        shots = few_shot_configs if few_shot_configs else [0, 2, 5]  # Few-shot configurations
        
        if num_samples:
            test_data = test_data[:num_samples]
        
        print(f"\n{'='*70}")
        print(f"🔬 FEW-SHOT Evaluating {self.model_type.upper()}")
        print(f"   Samples: {len(test_data)} | Methods: {len(methods)} | Shots: {shots}")
        print(f"   Total Experiments: {len(methods) * len(shots)}")
        print(f"{'='*70}\n")
        
        for method in methods:
            for num_shots in shots:
                print(f"\n📝 {method.upper().replace('_', ' ')} | {num_shots}-shot")
                
                metrics_agg = {k: [] for k in ['rouge_l', 'semantic_coherence', 'entailment', 
                                                'summac', 'bertscore', 'qe_overlap', 'entity_preservation']}
                preds = []
                
                for idx, sample in enumerate(tqdm(test_data, desc=f"{method}-{num_shots}")):
                    # Select examples for few-shot
                    examples = train_data[:num_shots] if num_shots > 0 else None
                    
                    # Generate prompt
                    if method == 'standard':
                        prompt = self.prompting.standard_prompting(sample['text'], num_shots, examples)
                    elif method == 'chain_of_density':
                        prompt = self.prompting.chain_of_density(sample['text'], num_shots, examples)
                    elif method == 'hierarchical':
                        prompt = self.prompting.hierarchical(sample['text'], num_shots, examples)
                    else:
                        prompt = self.prompting.element_aware(sample['text'], num_shots, examples)
                    
                    try:
                        pred = self.model.generate_summary(prompt)
                        scores = self.metrics.evaluate_all(pred, sample['summary'], sample['text'])
                        
                        for k, v in scores.items():
                            metrics_agg[k].append(v)
                        
                        if save_preds:
                            preds.append({
                                'model': self.model_type,
                                'method': method,
                                'num_shots': num_shots,
                                'sample_id': sample.get('id', idx),
                                'source_text': sample['text'],
                                'reference_summary': sample['summary'],
                                'generated_summary': pred,
                                **scores
                            })
                    except Exception as e:
                        print(f"⚠️  Error on sample {idx}: {e}")
                
                avg_metrics = {k: np.mean(v) if v else 0.0 for k, v in metrics_agg.items()}
                
                self.results.append({
                    'model': self.model_type,
                    'method': method,
                    'num_shots': num_shots,
                    **avg_metrics
                })
                
                if save_preds:
                    self.predictions.extend(preds)
                
                print(f"📊 Results: " + " | ".join([f"{k}={v:.4f}" for k, v in list(avg_metrics.items())[:4]]))
        
        self.save()


# ============ MAIN ============

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Few-Shot Evaluation for CHQ-Summ')
    parser.add_argument('--model', type=str, required=True,
                       choices=['bart', 'pegasus', 't5', 'prophetnet'],
                       help='Model to evaluate')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to fine-tuned model')
    parser.add_argument('--data_dir', type=str, default='data/',
                       help='Data directory')
    parser.add_argument('--samples', type=int, default=None,
                       help='Number of test samples')
    parser.add_argument('--shots', type=int, nargs='+', default=[0, 2, 5],
                       help='Few-shot configurations (e.g., --shots 0 2 5)')
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🚀 CHQ-SUMM FEW-SHOT EVALUATION")
    print("="*70)
    
    # Load data
    train_data, val_data, test_data = load_yahoo_dataset(args.data_dir)
    
    if not test_data or not train_data:
        print("❌ Could not load dataset")
        exit(1)
    
    # Run few-shot evaluation
    runner = FewShotExperimentRunner(args.model_path, args.model)
    runner.run_all(test_data, train_data, args.samples, save_preds=True, few_shot_configs=args.shots)
    
    print("\n✓ Few-shot evaluation complete!")
