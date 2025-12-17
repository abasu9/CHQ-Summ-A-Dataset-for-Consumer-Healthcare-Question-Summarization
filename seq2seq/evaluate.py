"""
Comprehensive evaluation script - ALL CORRECTIONS APPLIED
Metrics: ROUGE-L, BERTScore, Semantic Coherence, Q Entity Overlap, 
         Entity Preservation, SummaC (REAL), Entailment
"""

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from rouge_score import rouge_scorer
import logging
import spacy
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from summac.model_summac import SummaCZS
import warnings
import os
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComprehensiveEvaluator:
    """Comprehensive evaluation with all metrics corrected"""
    
    def __init__(self):
        logger.info("Initializing evaluation models...")
        
        # Initialize ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Load NER model for entity extraction
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            logger.warning("Spacy model not found. Installing...")
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
        
        # Load sentence transformer for semantic coherence
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load NLI model for entailment
        nli_model_name = 'microsoft/deberta-v2-xlarge-mnli'
        self.nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name)
        self.nli_model.eval()
        
        # Detect entailment label index from config
        id2label = {int(k): v.lower() for k, v in self.nli_model.config.id2label.items()}
        self.ent_idx = [i for i, lbl in id2label.items() if "entail" in lbl][0]
        logger.info(f"Entailment label index: {self.ent_idx} ({id2label[self.ent_idx]})")
        
        if torch.cuda.is_available():
            self.nli_model = self.nli_model.cuda()
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        
        # Load REAL SummaC model (SummaCZS)
        logger.info("Loading SummaC model (this may take a moment)...")
        self.summac_model = SummaCZS(granularity="sentence", model_name="vitc", device=self.device)
        
        logger.info("All evaluation models loaded!")
    
    def compute_rouge(self, predictions, references):
        """Compute ROUGE scores"""
        logger.info("Computing ROUGE scores...")
        
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        
        for pred, ref in zip(predictions, references):
            try:
                # Ensure strings are not None
                if pred is None:
                    pred = ""
                if ref is None:
                    ref = ""
                    
                scores = self.rouge_scorer.score(ref, pred)
                rouge1_scores.append(scores['rouge1'].fmeasure)
                rouge2_scores.append(scores['rouge2'].fmeasure)
                rougeL_scores.append(scores['rougeL'].fmeasure)
            except Exception as e:
                logger.warning(f"Error computing ROUGE: {e}")
                rouge1_scores.append(0.0)
                rouge2_scores.append(0.0)
                rougeL_scores.append(0.0)
        
        return {
            'rouge1': round(np.mean(rouge1_scores) * 100, 4),
            'rouge2': round(np.mean(rouge2_scores) * 100, 4),
            'rougeL': round(np.mean(rougeL_scores) * 100, 4),
            'rougeLsum': round(np.mean(rougeL_scores) * 100, 4)
        }
    
    def compute_bertscore(self, predictions, references):
        """
        FIXED: Compute BERTScore with correct model_type
        """
        logger.info("Computing BERTScore...")
        
        from bert_score import score
        
        # FIXED: Use correct model_type matching NLI model
        P, R, F1 = score(
            predictions,
            references,
            model_type='microsoft/deberta-v2-xlarge-mnli',  # FIXED: was 'deberta-xlarge-mnli'
            lang='en',
            verbose=False,
            device=self.device
        )
        
        return {
            'bertscore_precision': round(P.mean().item() * 100, 4),
            'bertscore_recall': round(R.mean().item() * 100, 4),
            'bertscore_f1': round(F1.mean().item() * 100, 4)
        }
    
    def compute_semantic_coherence(self, questions, summaries):
        """
        Compute semantic coherence between questions and summaries
        """
        logger.info("Computing semantic coherence...")
        
        # Get embeddings
        question_embeddings = self.semantic_model.encode(questions, show_progress_bar=True, batch_size=32)
        summary_embeddings = self.semantic_model.encode(summaries, show_progress_bar=True, batch_size=32)
        
        # Compute cosine similarities
        similarities = []
        for q_emb, s_emb in zip(question_embeddings, summary_embeddings):
            similarity = np.dot(q_emb, s_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(s_emb))
            similarities.append(similarity)
        
        return {
            'semantic_coherence_mean': round(np.mean(similarities) * 100, 4),
            'semantic_coherence_std': round(np.std(similarities) * 100, 4),
            'semantic_coherence_scores': [round(s * 100, 4) for s in similarities]
        }
    
    def extract_entities(self, text):
        """Extract named entities from text"""
        doc = self.nlp(text)
        entities = set()
        
        for ent in doc.ents:
            entities.add(ent.text.lower().strip())
        
        return entities
    
    def compute_entity_metrics(self, questions, summaries):
        """
        Compute Q Entity Overlap and Entity Preservation
        
        Q Entity Overlap = Jaccard similarity of entities
        Entity Preservation = % of question entities in summary
        """
        logger.info("Computing entity metrics...")
        
        q_entity_overlap_scores = []  # Jaccard
        entity_preservation_scores = []  # Preservation
        
        for question, summary in tqdm(zip(questions, summaries), total=len(questions), desc="Entity analysis"):
            q_entities = self.extract_entities(question)
            s_entities = self.extract_entities(summary)
            
            # Handle edge cases
            if len(q_entities) == 0:
                # No entities in question to overlap or preserve
                q_entity_overlap = 0.0
                entity_preservation = 1.0  # Nothing to preserve
            elif len(s_entities) == 0:
                # Summary has no entities
                q_entity_overlap = 0.0
                entity_preservation = 0.0  # Failed to preserve
            else:
                intersection = len(q_entities & s_entities)
                union = len(q_entities | s_entities)
                
                # Q Entity Overlap (Jaccard similarity)
                q_entity_overlap = intersection / union if union > 0 else 0.0
                
                # Entity Preservation (% of question entities preserved)
                entity_preservation = intersection / len(q_entities)
            
            q_entity_overlap_scores.append(q_entity_overlap)
            entity_preservation_scores.append(entity_preservation)
        
        return {
            'q_entity_overlap_mean': round(np.mean(q_entity_overlap_scores) * 100, 4),
            'q_entity_overlap_std': round(np.std(q_entity_overlap_scores) * 100, 4),
            'entity_preservation_mean': round(np.mean(entity_preservation_scores) * 100, 4),
            'entity_preservation_std': round(np.std(entity_preservation_scores) * 100, 4),
            'q_entity_overlap_scores': [round(j * 100, 4) for j in q_entity_overlap_scores],
            'entity_preservation_scores': [round(p * 100, 4) for p in entity_preservation_scores]
        }
    
    def compute_entailment_batch(self, premises, hypotheses, batch_size=16):
        """
        Batched entailment computation for speed
        Returns probability that premise entails hypothesis
        """
        all_scores = []
        
        for i in range(0, len(premises), batch_size):
            batch_premises = premises[i:i+batch_size]
            batch_hypotheses = hypotheses[i:i+batch_size]
            
            inputs = self.nli_tokenizer(
                batch_premises,
                batch_hypotheses,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors='pt'
            )
            
            if self.device == 'cuda':
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.nli_model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
            
            # Use detected entailment index
            batch_scores = probs[:, self.ent_idx].cpu().numpy()
            all_scores.extend(batch_scores)
        
        return all_scores
    
    def compute_entailment_metrics(self, questions, summaries):
        """
        Compute entailment scores (does question entail summary?)
        One direction: Q → S
        """
        logger.info("Computing entailment scores (Q → S)...")
        
        entailment_scores = self.compute_entailment_batch(questions, summaries, batch_size=16)
        
        return {
            'entailment_mean': round(np.mean(entailment_scores) * 100, 4),
            'entailment_std': round(np.std(entailment_scores) * 100, 4),
            'entailment_scores': [round(s * 100, 4) for s in entailment_scores]
        }
    
    def compute_summac(self, questions, summaries):
        """
        FIXED: Compute REAL SummaC scores using SummaCZS model
        """
        logger.info("Computing SummaC scores (REAL SummaC model)...")
        
        summac_scores = []
        
        # SummaCZS expects documents (sources) and summaries
        for question, summary in tqdm(zip(questions, summaries), total=len(questions), desc="SummaC"):
            # SummaC score_one takes: document (source) and generated_text (summary)
            score = self.summac_model.score([question], [summary])
            summac_scores.append(score['scores'][0])
        
        return {
            'summac_mean': round(np.mean(summac_scores) * 100, 4),
            'summac_std': round(np.std(summac_scores) * 100, 4),
            'summac_scores': [round(s * 100, 4) for s in summac_scores]
        }
    
    def evaluate_all(self, summaries_excel_file):
        """
        Run all evaluation metrics
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"COMPREHENSIVE EVALUATION: {summaries_excel_file}")
        logger.info(f"{'='*80}\n")
        
        # Load summaries from Excel
        df = pd.read_excel(summaries_excel_file, engine='openpyxl')
        
        questions = df['Original Question'].tolist()
        predictions = df['Generated Summary'].tolist()
        references = df['Human Summary'].tolist()
        
        logger.info(f"Evaluating {len(predictions)} predictions...\n")
        
        # Compute all metrics
        results = {}
        
        # 1. ROUGE
        results['rouge'] = self.compute_rouge(predictions, references)
        
        # 2. BERTScore (FIXED)
        results['bertscore'] = self.compute_bertscore(predictions, references)
        
        # 3. Semantic Coherence
        results['semantic_coherence'] = self.compute_semantic_coherence(questions, predictions)
        
        # 4 & 5. Q Entity Overlap & Entity Preservation
        results['entity_metrics'] = self.compute_entity_metrics(questions, predictions)
        
        # 6. SummaC (REAL SummaC model - FIXED)
        results['summac'] = self.compute_summac(questions, predictions)
        
        # 7. Entailment
        results['entailment'] = self.compute_entailment_metrics(questions, predictions)
        
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
        metrics_data['Metric'].append(f'ROUGE-{metric.replace("rouge", "")}')
        metrics_data['Score'].append(score)
    
    # BERTScore
    for metric, score in results['bertscore'].items():
        name = metric.replace('bertscore_', 'BERTScore-')
        metrics_data['Metric'].append(name)
        metrics_data['Score'].append(score)
    
    # Semantic Coherence
    metrics_data['Metric'].append('Semantic Coherence')
    metrics_data['Score'].append(results['semantic_coherence']['semantic_coherence_mean'])
    
    # Q Entity Overlap
    metrics_data['Metric'].append('Q Entity Overlap')
    metrics_data['Score'].append(results['entity_metrics']['q_entity_overlap_mean'])
    
    # Entity Preservation
    metrics_data['Metric'].append('Entity Preservation')
    metrics_data['Score'].append(results['entity_metrics']['entity_preservation_mean'])
    
    # SummaC
    metrics_data['Metric'].append('SummaC')
    metrics_data['Score'].append(results['summac']['summac_mean'])
    
    # Entailment
    metrics_data['Metric'].append('Entailment')
    metrics_data['Score'].append(results['entailment']['entailment_mean'])
    
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
    for metric, score in results['bertscore'].items():
        print(f"{metric}: {score:.4f}")
    
    print("\n--- Semantic Coherence ---")
    print(f"Mean: {results['semantic_coherence']['semantic_coherence_mean']:.4f}")
    print(f"Std: {results['semantic_coherence']['semantic_coherence_std']:.4f}")
    
    print("\n--- Q Entity Overlap ---")
    print(f"Mean: {results['entity_metrics']['q_entity_overlap_mean']:.4f}")
    print(f"Std: {results['entity_metrics']['q_entity_overlap_std']:.4f}")
    
    print("\n--- Entity Preservation ---")
    print(f"Mean: {results['entity_metrics']['entity_preservation_mean']:.4f}")
    print(f"Std: {results['entity_metrics']['entity_preservation_std']:.4f}")
    
    print("\n--- SummaC (REAL) ---")
    print(f"Mean: {results['summac']['summac_mean']:.4f}")
    print(f"Std: {results['summac']['summac_std']:.4f}")
    
    print("\n--- Entailment (Q → S) ---")
    print(f"Mean: {results['entailment']['entailment_mean']:.4f}")
    print(f"Std: {results['entailment']['entailment_std']:.4f}")
    
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
    evaluator = ComprehensiveEvaluator()
    
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
