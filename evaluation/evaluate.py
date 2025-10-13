#!/usr/bin/env python3
"""
Modern Evaluation Framework for CHQ-Summ Dataset
Supports BART, PEGASUS, T5, ProphetNet and custom models
"""

import os
import json
import re
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, List, Tuple
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    from lxml import etree as LET
    LXML_AVAILABLE = True
except ImportError:
    LET = None
    LXML_AVAILABLE = False

import xml.etree.ElementTree as ET

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util
import spacy

# Directories
BASE_DIR = Path.home() / "Yahoo-CHQ-Summ"
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# Create directories
for directory in [DATA_DIR, MODEL_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Model Configurations
MODEL_CONFIGS = {
    'bart': {
        'model_name': 'facebook/bart-large',
        'max_input_length': 1024,
        'max_target_length': 150,
        'min_target_length': 10,
        'num_beams': 6,
        'length_penalty': 2.0
    },
    'pegasus': {
        'model_name': 'google/pegasus-large',
        'max_input_length': 512,
        'max_target_length': 128,
        'min_target_length': 10,
        'num_beams': 6,
        'length_penalty': 2.0
    },
    't5': {
        'model_name': 't5-large',
        'max_input_length': 512,
        'max_target_length': 150,
        'min_target_length': 10,
        'num_beams': 6,
        'length_penalty': 2.0
    },
    'prophetnet': {
        'model_name': 'microsoft/prophetnet-large-uncased',
        'max_input_length': 512,
        'max_target_length': 150,
        'min_target_length': 10,
        'num_beams': 6,
        'length_penalty': 2.0
    }
}

SUMMARY_TAG_CANDIDATES = [
    'summary', 'abstract', 'synopsis', 'answersummary',
    'answer_summary', 'bestanswersummary', 'shortanswer', 'short_answer',
]


# ============ DATA LOADING ============

def _clean_text(blob: str, collapse_newlines: bool = True) -> str:
    """Clean and normalize text."""
    if blob is None:
        return ""
    text = str(blob).replace('\r', ' ').strip()
    if collapse_newlines:
        return re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' ?\n ?', '\n', text)
    return text.strip()


def _fallback_summary(answer: str, max_sentences: int = 3, max_chars: int = 400) -> str:
    """Generate fallback summary from answer."""
    if not answer:
        return ""
    sentences = re.split(r'(?<=[.!?])\s+', answer.strip())
    summary_parts = []
    char_budget = 0
    for sent in sentences:
        cleaned = _clean_text(sent)
        if not cleaned:
            continue
        if summary_parts and (len(summary_parts) >= max_sentences or char_budget + len(cleaned) > max_chars):
            break
        summary_parts.append(cleaned)
        char_budget += len(cleaned) + 1
    return " ".join(summary_parts)


def parse_yahoo_xml(xml_path: str) -> Dict[str, Dict]:
    """Parse Yahoo L6 XML file."""
    print(f"📂 Loading XML from: {xml_path}")
    
    if not os.path.exists(xml_path):
        print(f"❌ File not found: {xml_path}")
        return {}
    
    try:
        if LXML_AVAILABLE:
            parser = LET.XMLParser(recover=True, encoding='utf-8', huge_tree=True)
            tree = LET.parse(xml_path, parser)
            root = tree.getroot()
        else:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        
        data_dict = {}
        documents = root.findall('.//document')
        
        if not documents:
            documents = root.findall('.//doc')
        if not documents:
            documents = root.findall('.//item')
        
        print(f"✓ Found {len(documents)} documents")
        
        for idx, doc in enumerate(documents):
            doc_id = str(idx)
            for tag in ['id', 'uri', 'qid', 'nbestanswers']:
                elem = doc.find(tag)
                if elem is not None and elem.text:
                    doc_id = elem.text
                    break
            
            def get_text(tags):
                for tag in tags:
                    elem = doc.find(tag)
                    if elem is not None and elem.text:
                        return elem.text
                return ""
            
            subject = _clean_text(get_text(['subject', 'question']))
            content = _clean_text(get_text(['content', 'body']))
            answer = _clean_text(get_text(['bestanswer', 'answer']))
            summary_text = _clean_text(get_text(SUMMARY_TAG_CANDIDATES))
            
            if not summary_text:
                summary_text = _fallback_summary(answer)
            
            text_parts = []
            if subject:
                text_parts.append(f"Question: {subject}")
            if content:
                text_parts.append(f"Details: {content}")
            if answer:
                text_parts.append(f"Best Answer: {answer}")
            
            full_text = "\n\n".join(text_parts).strip()
            
            if full_text and summary_text:
                entry = {
                    'id': doc_id,
                    'text': full_text,
                    'summary': summary_text
                }
                data_dict[doc_id] = entry
                data_dict[str(idx)] = entry
        
        print(f"✓ Parsed {len(data_dict)} documents")
        return data_dict
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return {}


def load_annotations(json_path: str) -> List:
    """Load train/val/test split annotations."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            return []
        
        if data and isinstance(data[0], dict) and 'human_summary' in data[0]:
            print(f"✓ {os.path.basename(json_path)} contains full data")
            return [{'id': item['id'], 'summary': item['human_summary']} for item in data]
        
        if data and isinstance(data[0], dict) and 'text' in data[0] and 'summary' in data[0]:
            print(f"✓ {os.path.basename(json_path)} has complete question-summary pairs")
            return data
        
        ids = [str(item if not isinstance(item, dict) else item.get('id', idx))
               for idx, item in enumerate(data)]
        print(f"✓ Loaded {len(ids)} IDs from {os.path.basename(json_path)}")
        return ids
        
    except Exception as e:
        print(f"❌ Error loading {json_path}: {e}")
        return []


def load_yahoo_dataset(data_dir: str = 'data/') -> Tuple[List, List, List]:
    """Load Yahoo CHQ dataset with splits."""
    data_dir = Path(data_dir)
    
    print("="*70)
    print("📊 LOADING YAHOO CHQ-SUMM DATASET")
    print("="*70)
    
    # Try loading pre-processed JSON first
    try:
        test_path = data_dir / 'test.json'
        with open(test_path, 'r') as f:
            sample = json.load(f)
        
        if sample and isinstance(sample[0], dict) and 'text' in sample[0]:
            with open(data_dir / 'train.json', 'r') as f:
                train_data = json.load(f)
            with open(data_dir / 'test.json', 'r') as f:
                test_data = json.load(f)
            with open(data_dir / 'val.json', 'r') as f:
                val_data = json.load(f)
            
            print(f"✓ Loaded pre-processed data")
            print(f"  Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")
            return train_data, val_data, test_data
    except:
        pass
    
    # Parse XML and splits
    xml_candidates = ['yahool6.xml', 'FullOct2007.xml']
    xml_path = None
    for candidate in xml_candidates:
        candidate_path = data_dir / candidate
        if candidate_path.exists():
            xml_path = candidate_path
            break
    
    if not xml_path:
        print(f"❌ Yahoo XML file not found in {data_dir}")
        return None, None, None
    
    yahoo_data = parse_yahoo_xml(str(xml_path))
    
    if not yahoo_data:
        return None, None, None
    
    train_ids = load_annotations(str(data_dir / 'train.json'))
    test_ids = load_annotations(str(data_dir / 'test.json'))
    val_ids = load_annotations(str(data_dir / 'val.json'))
    
    if train_ids and isinstance(train_ids[0], dict) and 'text' in train_ids[0]:
        train_data = train_ids
        test_data = test_ids
        val_data = val_ids
    else:
        train_data = [yahoo_data[str(id)] for id in train_ids if str(id) in yahoo_data]
        test_data = [yahoo_data[str(id)] for id in test_ids if str(id) in yahoo_data]
        val_data = [yahoo_data[str(id)] for id in val_ids if str(id) in yahoo_data]
    
    print(f"✓ Dataset loaded successfully")
    print(f"  Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")
    return train_data, val_data, test_data


# ============ PROMPTING METHODS ============

class PromptingMethods:
    """Different prompting strategies for summarization."""
    
    @staticmethod
    def standard_prompting(text: str, num_shots: int = 0, examples: List = None) -> str:
        prompt = ""
        if num_shots > 0 and examples:
            prompt += "Examples:\n\n"
            for i, ex in enumerate(examples[:num_shots]):
                prompt += f"{i+1}. Text: {ex['text'][:150]}...\nSummary: {ex['summary']}\n\n"
        max_len = 800 if num_shots > 0 else 1000
        text_use = text[:max_len] + "..." if len(text) > max_len else text
        prompt += f"Summarize:\n\nText: {text_use}\n\nSummary:"
        return prompt
    
    @staticmethod
    def chain_of_density(text: str, num_shots: int = 0, examples: List = None) -> str:
        prompt = ""
        if num_shots > 0 and examples:
            prompt += "Dense summaries:\n\n"
            for i, ex in enumerate(examples[:num_shots]):
                prompt += f"{i+1}. Text: {ex['text'][:150]}...\nSummary: {ex['summary']}\n\n"
        max_len = 800 if num_shots > 0 else 1000
        text_use = text[:max_len] + "..." if len(text) > max_len else text
        prompt += f"Dense summary:\n\nText: {text_use}\n\nSummary:"
        return prompt
    
    @staticmethod
    def hierarchical(text: str, num_shots: int = 0, examples: List = None) -> str:
        prompt = ""
        if num_shots > 0 and examples:
            prompt += "Examples:\n\n"
            for i, ex in enumerate(examples[:num_shots]):
                prompt += f"{i+1}. Text: {ex['text'][:150]}...\nSummary: {ex['summary']}\n\n"
        max_len = 800 if num_shots > 0 else 1000
        text_use = text[:max_len] + "..." if len(text) > max_len else text
        prompt += f"Hierarchical summary:\n\nText: {text_use}\n\nSummary:"
        return prompt
    
    @staticmethod
    def element_aware(text: str, num_shots: int = 0, examples: List = None) -> str:
        prompt = ""
        if num_shots > 0 and examples:
            prompt += "Examples:\n\n"
            for i, ex in enumerate(examples[:num_shots]):
                prompt += f"{i+1}. Text: {ex['text'][:150]}...\nSummary: {ex['summary']}\n\n"
        max_len = 800 if num_shots > 0 else 1000
        text_use = text[:max_len] + "..." if len(text) > max_len else text
        prompt += f"WHO/WHAT/WHY:\n\nText: {text_use}\n\nSummary:"
        return prompt


# ============ EVALUATION METRICS ============

class EvaluationMetrics:
    """Comprehensive evaluation metrics for summarization."""
    
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.nlp = spacy.load('en_core_web_sm')
    
    def rouge_l(self, pred: str, ref: str) -> float:
        """ROUGE-L F1 score."""
        return self.rouge_scorer.score(ref, pred)['rougeL'].fmeasure
    
    def semantic_coherence(self, text: str) -> float:
        """Measure inter-sentence coherence."""
        sents = [s.text.strip() for s in self.nlp(text).sents if s.text.strip()]
        if len(sents) < 2:
            return 1.0
        embs = self.sentence_model.encode(sents)
        scores = [util.cos_sim(embs[i], embs[i+1]).item() for i in range(len(embs)-1)]
        return np.mean(scores) if scores else 1.0
    
    def entailment_score(self, summ: str, src: str) -> float:
        """Source-summary entailment score."""
        s_emb = self.sentence_model.encode(summ)
        src_emb = self.sentence_model.encode(src)
        return max(0.0, min(1.0, util.cos_sim(s_emb, src_emb).item()))
    
    def summac_score(self, summ: str, src: str) -> float:
        """SummaC consistency score."""
        sents = [s.text.strip() for s in self.nlp(summ).sents if s.text.strip()]
        if not sents:
            return 0.0
        src_emb = self.sentence_model.encode(src)
        scores = [max(0.0, util.cos_sim(self.sentence_model.encode(s), src_emb).item()) for s in sents]
        return np.mean(scores) if scores else 0.0
    
    def bertscore_metric(self, pred: str, ref: str) -> float:
        """BERTScore F1."""
        if not pred or len(pred.strip()) < 3 or not ref or len(ref.strip()) < 3:
            return 0.0
        try:
            from bert_score import score as bert_score_fn
            P, R, F1 = bert_score_fn([pred], [ref], lang='en', model_type='microsoft/deberta-xlarge-mnli', verbose=False)
            return F1.mean().item()
        except:
            try:
                p_emb = self.sentence_model.encode(pred)
                r_emb = self.sentence_model.encode(ref)
                return max(0.0, min(1.0, util.cos_sim(p_emb, r_emb).item()))
            except:
                return 0.0
    
    def qe_overlap(self, summ: str, src: str) -> float:
        """Question-entity overlap."""
        s_doc = self.nlp(summ)
        src_doc = self.nlp(src)
        s_ents = set([e.text.lower() for e in s_doc.ents])
        src_ents = set([e.text.lower() for e in src_doc.ents])
        if not src_ents:
            return 0.0
        return len(s_ents & src_ents) / len(src_ents)
    
    def entity_preservation(self, summ: str, src: str) -> float:
        """Entity preservation rate."""
        s_ents = [e.text.lower() for e in self.nlp(summ).ents]
        src_ents = [e.text.lower() for e in self.nlp(src).ents]
        if not src_ents:
            return 1.0
        return sum(1 for e in src_ents if e in s_ents) / len(src_ents)
    
    def evaluate_all(self, pred: str, ref: str, src: str) -> Dict[str, float]:
        """Run all evaluation metrics."""
        return {
            'rouge_l': self.rouge_l(pred, ref),
            'semantic_coherence': self.semantic_coherence(pred),
            'entailment': self.entailment_score(pred, src),
            'summac': self.summac_score(pred, src),
            'bertscore': self.bertscore_metric(pred, ref),
            'qe_overlap': self.qe_overlap(pred, src),
            'entity_preservation': self.entity_preservation(pred, src)
        }


# ============ MODEL EVALUATOR ============

class ModelEvaluator:
    """Wrapper for loading and evaluating models."""
    
    def __init__(self, model_path: str, model_type: str):
        self.model_type = model_type
        self.config = MODEL_CONFIGS.get(model_type, MODEL_CONFIGS['bart'])
        
        print(f"\n🤖 Loading {model_type.upper()} model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Model loaded on {self.device}")
    
    def generate_summary(self, prompt: str) -> str:
        """Generate summary for given prompt."""
        if 't5' in self.model_type.lower():
            prompt = "summarize: " + prompt
        
        inputs = self.tokenizer(
            prompt,
            max_length=self.config['max_input_length'],
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            ids = self.model.generate(
                inputs['input_ids'],
                max_length=self.config['max_target_length'],
                min_length=self.config['min_target_length'],
                num_beams=self.config['num_beams'],
                length_penalty=self.config['length_penalty'],
                early_stopping=True,
                no_repeat_ngram_size=3
            )
        
        summary = self.tokenizer.decode(ids[0], skip_special_tokens=True).strip()
        return summary if summary and len(summary) >= 5 else "No summary generated."


# ============ EXPERIMENT RUNNER ============

class ExperimentRunner:
    """Run complete evaluation experiments."""
    
    def __init__(self, model_path: str, model_type: str):
        self.model = ModelEvaluator(model_path, model_type)
        self.metrics = EvaluationMetrics()
        self.prompting = PromptingMethods()
        self.model_type = model_type
        self.results = []
        self.predictions = []
    
    def run_all(self, test_data, train_data, num_samples=None, save_preds=True):
        """Run all evaluation experiments."""
        methods = ['standard', 'chain_of_density', 'hierarchical', 'element_aware']
        shots = [0]  # 0-shot only
        
        if num_samples:
            test_data = test_data[:num_samples]
        
        print(f"\n{'='*70}")
        print(f"🔬 Evaluating {self.model_type.upper()}")
        print(f"   Samples: {len(test_data)} | Methods: {len(methods)} | Shots: {shots}")
        print(f"{'='*70}\n")
        
        for method in methods:
            for num_shots in shots:
                print(f"\n📝 {method.upper().replace('_', ' ')} | {num_shots}-shot")
                
                metrics_agg = {k: [] for k in ['rouge_l', 'semantic_coherence', 'entailment', 
                                                'summac', 'bertscore', 'qe_overlap', 'entity_preservation']}
                preds = []
                
                for idx, sample in enumerate(tqdm(test_data, desc=f"{method}-{num_shots}")):
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
    
    def save(self):
        """Save results to CSV files."""
        df = pd.DataFrame(self.results)
        df.to_csv(str(RESULTS_DIR / f'{self.model_type}_eval.csv'), index=False)
        print(f"✓ Saved metrics: {self.model_type}_eval.csv")
        
        if self.predictions:
            pdf = pd.DataFrame(self.predictions)
            pdf.to_csv(str(RESULTS_DIR / f'{self.model_type}_summaries.csv'), index=False)
            print(f"✓ Saved summaries: {self.model_type}_summaries.csv")


# ============ MAIN EVALUATION ============

def evaluate_all_models(test_data, train_data, model_paths: Dict = None, num_samples=None):
    """Evaluate all available models."""
    
    if model_paths is None:
        # Default: look for fine-tuned models
        model_paths = {
            'bart': MODEL_DIR / 'bart' / 'final',
            'pegasus': MODEL_DIR / 'pegasus' / 'final',
            't5': MODEL_DIR / 't5' / 'final',
            'prophetnet': MODEL_DIR / 'prophetnet' / 'final'
        }
    
    existing = {k: v for k, v in model_paths.items() if Path(v).exists()}
    
    if not existing:
        print("⚠️  No fine-tuned models found. Using pre-trained models...")
        existing = {k: MODEL_CONFIGS[k]['model_name'] for k in MODEL_CONFIGS.keys()}
    
    print(f"\n🎯 Found {len(existing)} models to evaluate")
    
    all_results = []
    
    for model_type, model_path in existing.items():
        print(f"\n{'#'*70}")
        print(f"# {model_type.upper()}")
        print(f"{'#'*70}")
        
        try:
            runner = ExperimentRunner(str(model_path), model_type)
            runner.run_all(test_data, train_data, num_samples, save_preds=True)
            all_results.append(pd.DataFrame(runner.results))
        except Exception as e:
            print(f"❌ Error evaluating {model_type}: {e}")
            import traceback
            traceback.print_exc()
    
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(str(RESULTS_DIR / 'ALL_MODELS_eval.csv'), index=False)
        
        print("\n" + "="*70)
        print("📊 FINAL SUMMARY")
        print("="*70)
        print(combined.groupby('model')[['rouge_l', 'bertscore', 'entailment']].mean().round(4))
        print(f"\n✓ All results saved to: {RESULTS_DIR}/")


# ============ CLI ============

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate CHQ-Summ models')
    parser.add_argument('--model', type=str, default='all', 
                       choices=['all', 'bart', 'pegasus', 't5', 'prophetnet'],
                       help='Model to evaluate')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to fine-tuned model')
    parser.add_argument('--data_dir', type=str, default='data/',
                       help='Data directory')
    parser.add_argument('--samples', type=int, default=None,
                       help='Number of test samples (default: all)')
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🚀 CHQ-SUMM EVALUATION FRAMEWORK")
    print("="*70)
    
    # Load data
    train_data, val_data, test_data = load_yahoo_dataset(args.data_dir)
    
    if not test_data or not train_data:
        print("❌ Could not load dataset")
        exit(1)
    
    # Evaluate
    if args.model != 'all' and args.model_path:
        # Single model evaluation
        runner = ExperimentRunner(args.model_path, args.model)
        runner.run_all(test_data, train_data, args.samples)
    else:
        # Evaluate all models
        evaluate_all_models(test_data, train_data, num_samples=args.samples)
    
    print("\n✓ Evaluation complete!")
