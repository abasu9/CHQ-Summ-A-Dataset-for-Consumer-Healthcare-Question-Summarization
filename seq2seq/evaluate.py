"""
Comprehensive evaluation script for seq2seq models
Metrics: ROUGE-Lsum, BERTScore, METEOR, Semantic Coherence (SBERT cosine), Entailment
"""

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from rouge_score import rouge_scorer
import logging
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings
import os
import nltk
from nltk.translate.meteor_score import meteor_score
warnings.filterwarnings('ignore')

# Download NLTK resources for METEOR (AWS-safe: use local directory)
NLTK_DIR = os.path.join(os.path.dirname(__file__), "nltk_data")
os.makedirs(NLTK_DIR, exist_ok=True)
nltk.data.path.append(NLTK_DIR)

try:
    nltk.download('wordnet', download_dir=NLTK_DIR, quiet=True)
    nltk.download('omw-1.4', download_dir=NLTK_DIR, quiet=True)
except Exception as e:
    logger_init = logging.getLogger(__name__)
    logger_init.warning(f"Failed to download NLTK resources: {e}")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Evaluation sample cap to avoid long runs
EVAL_SAMPLE_LIMIT = 50


def _safe_str(x):
    """Convert None/NaN to empty string, otherwise strip string."""
    if pd.isna(x) or x is None:
        return ""
    return str(x).strip()


class MetricsCalculator:
    """Calculate evaluation metrics for seq2seq models"""

    def __init__(self):
        logger.info("Initializing evaluation models...")
        
        self.rouge_scorer = rouge_scorer.RougeScorer(["rougeLsum"], use_stemmer=True)
        
        # Load sentence transformer for semantic coherence
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2', device=DEVICE)
        
        # Load NLI model for entailment
        nli_model_name = 'roberta-large-mnli'
        self.nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name).to(DEVICE).eval()
        
        self.device = DEVICE
        logger.info("All evaluation models loaded!")

    def to_text(self, x):
        """Convert to text: empty string for None/NaN, else str(x).strip()."""
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return ""
        return str(x).strip()

    def validate_text(self, text):
        text = self.to_text(text)
        return text and len(text.strip()) >= 3 and len(text.split()) >= 2

    def calculate_rouge_l(self, pred, ref):
        pred = self.to_text(pred)
        ref = self.to_text(ref)
        if not pred or not ref:
            return 0.0
        try:
            return self.rouge_scorer.score(ref, pred)["rougeLsum"].fmeasure
        except Exception:
            return 0.0

    def calculate_bertscore(self, preds, refs):
        if not preds or not refs:
            return {"f1": 0.0}

        pairs = []
        for p, r in zip(preds, refs):
            p = self.to_text(p)
            r = self.to_text(r)
            if self.validate_text(p) and self.validate_text(r):
                pairs.append((p, r))

        if not pairs:
            return {"f1": 0.0}

        valid_preds, valid_refs = zip(*pairs)

        try:
            from bert_score import score as bert_score_fn
            _, _, F1 = bert_score_fn(
                list(valid_preds),
                list(valid_refs),
                lang="en",
                verbose=False,
                device=DEVICE,
            )
            return {"f1": max(float(F1.mean().item()), 0.0)}
        except Exception:
            return {"f1": 0.0}

    def calculate_meteor(self, pred, ref):
        """METEOR score using NLTK implementation."""
        pred = self.to_text(pred)
        ref = self.to_text(ref)
        if not pred or not ref:
            return 0.0
        try:
            # meteor_score expects reference as list of token lists, hypothesis as token list
            score = meteor_score([ref.split()], pred.split())
            return max(float(score), 0.0)
        except Exception:
            return 0.0

    def calculate_semantic_coherence(self, pred, question):
        """Semantic coherence between generated summary and original question."""
        pred = self.to_text(pred)
        question = self.to_text(question)
        if not pred or not question:
            return 0.0
        try:
            e1 = self.sentence_model.encode(pred, convert_to_tensor=True)
            e2 = self.sentence_model.encode(question[:512], convert_to_tensor=True)
            return max(float(util.cos_sim(e1, e2).item()), 0.0)
        except Exception:
            return 0.0


    def calculate_entailment(self, pred, question):
        """NLI entailment probability: question entails summary (faithfulness)."""
        pred = self.to_text(pred)
        question = self.to_text(question)
        if not pred or not question:
            return 0.0
        try:
            inputs = self.nli_tokenizer(
                question[:512],  # premise = question
                pred[:512],      # hypothesis = summary
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512,
            )
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            with torch.no_grad():
                logits = self.nli_model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)[0]

            id2label = {int(k): v for k, v in self.nli_model.config.id2label.items()}
            entail_id = next((i for i, lab in id2label.items() if "ENTAIL" in lab.upper()), None)
            if entail_id is None:
                entail_id = 2  # fallback for common MNLI ordering

            return float(probs[entail_id].item())
        except Exception:
            return 0.0

    def evaluate_all(self, summaries_excel_file):
        """
        Run all evaluation metrics
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"COMPREHENSIVE EVALUATION: {summaries_excel_file}")
        logger.info(f"{'='*80}\n")

        # Load summaries from Excel
        df = pd.read_excel(summaries_excel_file, engine='openpyxl')
        
        # Validate required columns exist
        required_cols = ['Original Question', 'Generated Summary', 'Human Summary']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            logger.error(f"Available columns: {df.columns.tolist()}")
            raise ValueError(f"Excel file missing required columns: {missing_cols}")

        questions = [_safe_str(q) for q in df['Original Question']]
        predictions = [_safe_str(p) for p in df['Generated Summary']]
        references = [_safe_str(r) for r in df['Human Summary']]

        logger.info(f"Total examples loaded: {len(predictions)}")
        
        # Filter valid pairs (non-empty pred and ref)
        valid_indices = [
            i for i in range(len(predictions))
            if self.validate_text(predictions[i]) and self.validate_text(references[i]) and self.validate_text(questions[i])
        ]
        
        logger.info(f"Valid pairs after filtering: {len(valid_indices)}/{len(predictions)}")
        
        if len(valid_indices) == 0:
            logger.error("No valid pairs found! All predictions/references/questions are empty or too short.")
            return {
                'rouge': {'rougeLsum': 0.0},
                'bertscore': {'f1': 0.0},
                'meteor': {'meteor': 0.0},
                'entailment': {'entailment_mean': 0.0, 'entailment_std': 0.0},
                'semantic_coherence': {'semantic_coherence_mean': 0.0},
                'summary': {
                    'num_examples': 0,
                    'avg_prediction_length': 0.0,
                    'avg_reference_length': 0.0,
                    'avg_question_length': 0.0
                }
            }

        # Optionally cap evaluation to a fixed sample size
        if len(valid_indices) > EVAL_SAMPLE_LIMIT:
            logger.info(f"Capping evaluation to first {EVAL_SAMPLE_LIMIT} valid samples (from {len(valid_indices)})")
            valid_indices = valid_indices[:EVAL_SAMPLE_LIMIT]

        # Use only valid pairs
        questions = [questions[i] for i in valid_indices]
        predictions = [predictions[i] for i in valid_indices]
        references = [references[i] for i in valid_indices]

        logger.info(f"Evaluating {len(predictions)} valid predictions...\n")

        # Compute all metrics
        results = {}

        # 1. ROUGE-Lsum
        logger.info("Computing ROUGE-Lsum scores...")
        rouge_scores = [self.calculate_rouge_l(p, r) for p, r in tqdm(zip(predictions, references), total=len(predictions), desc="ROUGE")]
        results['rouge'] = {'rougeLsum': round(np.mean(rouge_scores), 4)}

        # 2. BERTScore
        results['bertscore'] = self.calculate_bertscore(predictions, references)

        # 3. METEOR
        logger.info("Computing METEOR scores...")
        meteor_scores = [self.calculate_meteor(p, r) for p, r in tqdm(zip(predictions, references), total=len(predictions), desc="METEOR")]
        results['meteor'] = {'meteor': round(np.mean(meteor_scores), 4)}

        # 4. Entailment
        logger.info("Computing entailment scores...")
        entailment_scores = [self.calculate_entailment(p, q) for p, q in tqdm(zip(predictions, questions), total=len(predictions), desc="Entailment")]
        results['entailment'] = {
            'entailment_mean': round(np.mean(entailment_scores), 4),
            'entailment_std': round(np.std(entailment_scores), 4)
        }

        # 5. Semantic Coherence
        logger.info("Computing semantic coherence...")
        semantic_scores = [self.calculate_semantic_coherence(p, q) for p, q in tqdm(zip(predictions, questions), total=len(predictions), desc="Semantic")]
        results['semantic_coherence'] = {'semantic_coherence_mean': round(np.mean(semantic_scores), 4)}

        # Add summary statistics
        results['summary'] = {
            'num_examples': len(predictions),
            'avg_prediction_length': round(np.mean([len(p.split()) for p in predictions]), 2),
            'avg_reference_length': round(np.mean([len(r.split()) for r in references]), 2),
            'avg_question_length': round(np.mean([len(q.split()) for q in questions]), 2)
        }

        return results


def save_metrics_to_excel(results, output_file):
    """Save evaluation metrics to Excel file"""
    
    logger.info(f"Saving metrics to Excel: {output_file}")
    
    # Create summary metrics DataFrame
    metrics_data = {
        'Metric': [],
        'Score': []
    }
    
    # ROUGE scores
    for metric, score in results['rouge'].items():
        metrics_data['Metric'].append(f'ROUGE-{metric.replace("rouge", "").upper()}')
        metrics_data['Score'].append(score)
    
    # BERTScore
    metrics_data['Metric'].append('BERTScore-F1')
    metrics_data['Score'].append(results['bertscore']['f1'])
    
    # METEOR
    metrics_data['Metric'].append('METEOR')
    metrics_data['Score'].append(results['meteor']['meteor'])
    
    # Entailment
    metrics_data['Metric'].append('Entailment')
    metrics_data['Score'].append(results['entailment']['entailment_mean'])
    
    # Semantic Coherence
    metrics_data['Metric'].append('Semantic Coherence')
    metrics_data['Score'].append(results['semantic_coherence']['semantic_coherence_mean'])
    
    df = pd.DataFrame(metrics_data)
    
    # Save to Excel
    df.to_excel(output_file, index=False, engine='openpyxl')
    
    logger.info(f"✓ Metrics saved to: {output_file}")


def print_results(results):
    """Print results in readable format"""
    
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    print("\n--- ROUGE Scores ---")
    for metric, score in results['rouge'].items():
        print(f"{metric}: {score:.4f}")
    
    print("\n--- BERTScore ---")
    print(f"F1: {results['bertscore']['f1']:.4f}")
    
    print("\n--- METEOR ---")
    print(f"Score: {results['meteor']['meteor']:.4f}")
    
    print("\n--- Entailment (Q → S) ---")
    print(f"Mean: {results['entailment']['entailment_mean']:.4f}")
    print(f"Std: {results['entailment']['entailment_std']:.4f}")
    
    print("\n--- Semantic Coherence ---")
    print(f"Mean: {results['semantic_coherence']['semantic_coherence_mean']:.4f}")
    
    print("\n--- Summary Statistics ---")
    for metric, value in results['summary'].items():
        print(f"{metric}: {value}")
    
    print("\n" + "="*80)


def main():
    """Evaluate all models"""
    
    BASE_RESULTS_DIR = "./yahoo_l6_results"
    
    model_dirs = [
        "prophetnet-large-uncased",
        "pegasus-large",
        "bart-large",
        "t5-base"
    ]
    
    logger.info("="*80)
    logger.info("EVALUATING ALL MODELS")
    logger.info("="*80)
    
    # Initialize evaluator once
    evaluator = MetricsCalculator()
    
    for model_dir in model_dirs:
        summaries_file = os.path.join(BASE_RESULTS_DIR, model_dir, "summaries.xlsx")
        
        if not os.path.exists(summaries_file):
            logger.warning(f"Summaries file not found: {summaries_file}")
            continue
        
        logger.info(f"\nEvaluating {model_dir}...")
        
        try:
            # Run evaluation
            results = evaluator.evaluate_all(summaries_file)
            
            # Print results
            print_results(results)
            
            # Save metrics to Excel
            metrics_file = os.path.join(BASE_RESULTS_DIR, model_dir, "metrics.xlsx")
            save_metrics_to_excel(results, metrics_file)
            
        except Exception as e:
            logger.error(f"Error evaluating {model_dir}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    logger.info("\n✓ All evaluations completed!")


if __name__ == "__main__":
    main()
