"""
CHQ Summarization and Answer Retrieval Pipeline - All 10 Models with Top-K=4
Retrieves TOP 4 answers and calculates expanded metrics
"""

import os
import json
import xml.etree.ElementTree as ET
import pandas as pd
import torch
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM,
    BartForConditionalGeneration, PegasusForConditionalGeneration,
    T5ForConditionalGeneration
)
from rank_bm25 import BM25Okapi
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class LiveQADataset:
    """Load and parse LiveQA XML dataset"""
    
    def __init__(self, xml_path, max_questions=50):
        self.xml_path = xml_path
        self.max_questions = max_questions
        self.questions = []
        self.load_data()
    
    def load_data(self):
        """Parse XML and extract questions and summaries"""
        tree = ET.parse(self.xml_path)
        root = tree.getroot()
        
        count = 0
        for nlm_question in root.findall('NLM-QUESTION'):
            if count >= self.max_questions:
                break
                
            qid = nlm_question.get('qid')
            
            # Get original question
            orig_q = nlm_question.find('Original-Question')
            subject = orig_q.find('SUBJECT').text if orig_q.find('SUBJECT') is not None else ""
            message = orig_q.find('MESSAGE').text if orig_q.find('MESSAGE') is not None else ""
            original_question = f"{subject} {message}".strip()
            
            # Get reference summary (NLM-Summary)
            ref_summary_elem = nlm_question.find('NLM-Summary')
            reference_summary = ref_summary_elem.text.strip() if ref_summary_elem is not None else ""
            
            self.questions.append({
                'qid': qid,
                'original_question': original_question,
                'reference_summary': reference_summary
            })
            
            count += 1
        
        print(f"Loaded {len(self.questions)} questions from LiveQA dataset")

class MedQuadRetriever:
    """BM25-based answer retrieval from MedQuad collection"""
    
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.answers = []
        self.answer_ids = []
        self.bm25 = None
        self.load_data()
    
    def load_data(self):
        """Load MedQuad collection"""
        df = pd.read_csv(self.csv_path)
        print(f"Loaded {len(df)} answers from MedQuad")
        
        self.answer_ids = df['AnswerID'].tolist()
        self.answers = df['Answer'].fillna('').tolist()
        
        # Tokenize answers for BM25
        tokenized_answers = [answer.lower().split() for answer in self.answers]
        self.bm25 = BM25Okapi(tokenized_answers)
        print("BM25 index built")
    
    def retrieve(self, query, top_k=4):
        """Retrieve top-k answers for a query"""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'answer_id': self.answer_ids[idx],
                'answer': self.answers[idx],
                'bm25_score': float(scores[idx])
            })
        
        return results

class SummarizationModel:
    """Wrapper for different summarization models"""
    
    def __init__(self, model_name, model_path=None):
        self.model_name = model_name
        self.model_path = model_path or model_name
        self.model = None
        self.tokenizer = None
        self.is_causal_lm = False
        self.load_model()
    
    def load_model(self):
        """Load the model and tokenizer"""
        print(f"Loading {self.model_name}...")
        
        try:
            if 'mistral' in self.model_name.lower() or 'llama' in self.model_name.lower() or \
               'qwen' in self.model_name.lower() or 'deepseek' in self.model_name.lower() or \
               'gemma' in self.model_name.lower():
                # Causal LM models
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
                self.is_causal_lm = True
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    
            elif 'bart' in self.model_name.lower():
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.model = BartForConditionalGeneration.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
                
            elif 'pegasus' in self.model_name.lower():
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.model = PegasusForConditionalGeneration.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
                
            elif 't5' in self.model_name.lower():
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.model = T5ForConditionalGeneration.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
                
            elif 'prophetnet' in self.model_name.lower():
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
            
            else:
                # Generic seq2seq
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
            
            self.model.eval()
            print(f"✓ {self.model_name} loaded successfully")
            
        except Exception as e:
            print(f"✗ Error loading {self.model_name}: {str(e)}")
            raise
    
    def generate_summary(self, question, max_length=64, min_length=10):
        """Generate summary for a question"""
        try:
            if self.is_causal_lm:
                prompt = f"""Summarize the following medical question into a concise, clear question:

Question: {question}

Summary:"""
                
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_length,
                        min_new_tokens=min_length,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                
                full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                if "Summary:" in full_text:
                    summary = full_text.split("Summary:")[-1].strip()
                else:
                    summary = full_text[len(prompt):].strip()
                
                summary = summary.split('\n')[0].strip()
                
            else:
                if 't5' in self.model_name.lower():
                    input_text = f"summarize: {question}"
                else:
                    input_text = question
                
                inputs = self.tokenizer(
                    input_text,
                    return_tensors="pt",
                    max_length=512,
                    truncation=True,
                    padding=True
                )
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_length=max_length,
                        min_length=min_length,
                        num_beams=4,
                        length_penalty=2.0,
                        early_stopping=True,
                        no_repeat_ngram_size=3
                    )
                
                summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return summary.strip()
            
        except Exception as e:
            print(f"Error generating summary with {self.model_name}: {str(e)}")
            return question

class LLMJudge:
    """LLM-as-Judge for scoring answer relevance (1-4 scale)"""
    
    def __init__(self, judge_model="mistralai/Mistral-7B-Instruct-v0.3"):
        print(f"Loading judge model: {judge_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(judge_model)
        self.model = AutoModelForCausalLM.from_pretrained(
            judge_model,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.model.eval()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print("✓ Judge model loaded")
    
    def score_answer(self, question, answer):
        """
        Score answer quality on 1-4 scale:
        4 = Correct and Complete Answer
        3 = Correct but Incomplete
        2 = Incorrect but Related
        1 = Incorrect
        """
        
        prompt = f"""You are an expert medical information evaluator. Score the following answer to a medical question on a scale of 1-4:

4 = Correct and Complete Answer - The answer fully addresses the question with accurate medical information
3 = Correct but Incomplete - The answer is medically accurate but doesn't fully address all aspects of the question
2 = Incorrect but Related - The answer contains relevant medical information but has inaccuracies or doesn't address the question properly
1 = Incorrect - The answer is medically inaccurate or completely irrelevant to the question

Question: {question}

Answer: {answer[:1000]}

Provide ONLY a single number (1, 2, 3, or 4) as your score. No explanation needed.

Score:"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "Score:" in response:
            score_text = response.split("Score:")[-1].strip()
        else:
            score_text = response[len(prompt):].strip()
        
        try:
            score = int(score_text[0])
            if score < 1 or score > 4:
                score = 2
        except:
            score = 2
        
        return score

def calculate_metrics_topk(all_scores):
    """
    Calculate Top-K metrics
    all_scores: list of lists, each inner list has 4 scores [score1, score2, score3, score4]
    
    Returns metrics:
    - avgTop1: Average score of top-1 (first) answer (0-3 scale)
    - avgBest@4: Average of best score among top-4 (0-3 scale)
    - nDCG@4: Normalized Discounted Cumulative Gain at 4
    - succ@3(3+): Success rate for questions with at least one answer ≥3
    - succ@4(4+): Success rate for questions with at least one answer =4
    - prec@3(3+): Precision for answers ≥3
    - prec@4(4+): Precision for answers =4
    - MRR@4(3+): Mean Reciprocal Rank for first answer ≥3
    - MRR@4(4+): Mean Reciprocal Rank for first answer =4
    """
    
    total_questions = len(all_scores)
    if total_questions == 0:
        return {
            'avgTop1': 0.0,
            'avgBest@4': 0.0,
            'nDCG@4': 0.0,
            'succ@3(3+)': 0.0,
            'succ@4(4+)': 0.0,
            'prec@3(3+)': 0.0,
            'prec@4(4+)': 0.0,
            'MRR@4(3+)': 0.0,
            'MRR@4(4+)': 0.0
        }
    
    # avgTop1: Average score of first answer (convert 1-4 to 0-3)
    top1_scores = [(scores[0] - 1) for scores in all_scores]
    avg_top1 = sum(top1_scores) / total_questions
    
    # avgBest@4: Average of best score among 4 answers (convert 1-4 to 0-3)
    best_scores = [(max(scores) - 1) for scores in all_scores]
    avg_best4 = sum(best_scores) / total_questions
    
    # nDCG@4: Normalized Discounted Cumulative Gain
    ndcg_scores = []
    for scores in all_scores:
        # Convert to 0-3 scale for relevance
        relevances = [s - 1 for s in scores]
        
        # DCG@4
        dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances))
        
        # IDCG@4 (ideal: sorted descending)
        ideal_relevances = sorted(relevances, reverse=True)
        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevances))
        
        ndcg = dcg / idcg if idcg > 0 else 0
        ndcg_scores.append(ndcg)
    
    avg_ndcg4 = sum(ndcg_scores) / total_questions
    
    # succ@3(3+): Questions with at least one answer ≥3
    succ3_count = sum(1 for scores in all_scores if any(s >= 3 for s in scores))
    succ3 = succ3_count / total_questions
    
    # succ@4(4+): Questions with at least one answer =4
    succ4_count = sum(1 for scores in all_scores if any(s == 4 for s in scores))
    succ4 = succ4_count / total_questions
    
    # prec@3(3+): Precision of answers ≥3
    total_answers = total_questions * 4
    answers_geq3 = sum(sum(1 for s in scores if s >= 3) for scores in all_scores)
    prec3 = answers_geq3 / total_answers
    
    # prec@4(4+): Precision of answers =4
    answers_eq4 = sum(sum(1 for s in scores if s == 4) for scores in all_scores)
    prec4 = answers_eq4 / total_answers
    
    # MRR@4(3+): Mean Reciprocal Rank for first answer ≥3
    mrr3_scores = []
    for scores in all_scores:
        rank = next((i + 1 for i, s in enumerate(scores) if s >= 3), 0)
        mrr3_scores.append(1.0 / rank if rank > 0 else 0.0)
    mrr3 = sum(mrr3_scores) / total_questions
    
    # MRR@4(4+): Mean Reciprocal Rank for first answer =4
    mrr4_scores = []
    for scores in all_scores:
        rank = next((i + 1 for i, s in enumerate(scores) if s == 4), 0)
        mrr4_scores.append(1.0 / rank if rank > 0 else 0.0)
    mrr4 = sum(mrr4_scores) / total_questions
    
    return {
        'avgTop1': round(avg_top1, 3),
        'avgBest@4': round(avg_best4, 3),
        'nDCG@4': round(avg_ndcg4, 3),
        'succ@3(3+)': round(succ3, 3),
        'succ@4(4+)': round(succ4, 3),
        'prec@3(3+)': round(prec3, 3),
        'prec@4(4+)': round(prec4, 3),
        'MRR@4(3+)': round(mrr3, 3),
        'MRR@4(4+)': round(mrr4, 3)
    }

def main():
    # Configuration
    LIVEQA_PATH = "liveqa.xml"
    MEDQUAD_PATH = "medquad.csv"
    OUTPUT_DIR = "results"
    NUM_QUESTIONS = 50
    TOP_K = 4
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # ═══════════════════════════════════════════════════════════
    # ALL 10 MODELS
    # ═══════════════════════════════════════════════════════════
    models_config = [
        {"name": "Gemma-7B", "path": "google/gemma-7b-it"},
        {"name": "Mistral-7B-v0.3", "path": "mistralai/Mistral-7B-Instruct-v0.3"},
        {"name": "Llama-3.1-8B", "path": "meta-llama/Llama-3.1-8B-Instruct"},
        {"name": "Llama-3.2-3B", "path": "meta-llama/Llama-3.2-3B-Instruct"},
        {"name": "Pegasus-Large", "path": "google/pegasus-large"},
        {"name": "T5-Base", "path": "t5-base"},
        {"name": "Qwen2-7B", "path": "Qwen/Qwen2-7B-Instruct"},
        {"name": "DeepSeek-7B", "path": "deepseek-ai/deepseek-llm-7b-chat"},
        {"name": "BART-Large", "path": "facebook/bart-large-cnn"},
        {"name": "ProphetNet-Large", "path": "microsoft/prophetnet-large-uncased"}
    ]
    
    # Load datasets
    print("="*80)
    print(f"Loading Datasets (Limited to {NUM_QUESTIONS} questions)...")
    print("="*80)
    liveqa = LiveQADataset(LIVEQA_PATH, max_questions=NUM_QUESTIONS)
    retriever = MedQuadRetriever(MEDQUAD_PATH)
    
    # Initialize judge
    print("\n" + "="*80)
    print("Initializing LLM Judge...")
    print("="*80)
    judge = LLMJudge()
    
    # Store all results
    all_results = []
    model_metrics = {}
    
    # Process each model
    for model_config in models_config:
        model_name = model_config["name"]
        model_path = model_config["path"]
        
        print(f"\n{'='*80}")
        print(f"Processing Model: {model_name}")
        print(f"{'='*80}")
        
        try:
            # Load model
            summarizer = SummarizationModel(model_name, model_path)
            
            # Store scores for metric calculation
            scores_original_all = []
            scores_reference_all = []
            scores_generated_all = []
            
            for i, q_data in enumerate(tqdm(liveqa.questions, desc=f"{model_name}")):
                qid = q_data['qid']
                original_question = q_data['original_question']
                reference_summary = q_data['reference_summary']
                
                # Generate summary
                generated_summary = summarizer.generate_summary(original_question)
                
                # ═══════════════════════════════════════════════════════════
                # THREE-WAY RETRIEVAL with TOP-K=4
                # ═══════════════════════════════════════════════════════════
                
                # 1. Retrieve TOP 4 using ORIGINAL QUESTION
                retrieved_orig = retriever.retrieve(original_question, top_k=TOP_K)
                
                # 2. Retrieve TOP 4 using REFERENCE SUMMARY
                retrieved_ref = retriever.retrieve(reference_summary, top_k=TOP_K)
                
                # 3. Retrieve TOP 4 using GENERATED SUMMARY
                retrieved_gen = retriever.retrieve(generated_summary, top_k=TOP_K)
                
                # ═══════════════════════════════════════════════════════════
                # SCORE ALL 4 ANSWERS FOR EACH QUERY TYPE
                # ═══════════════════════════════════════════════════════════
                
                # Score 4 answers from original question
                scores_original = []
                answers_original = []
                for ret in retrieved_orig:
                    score = judge.score_answer(original_question, ret['answer'])
                    scores_original.append(score)
                    answers_original.append(ret['answer'])
                
                # Score 4 answers from reference summary
                scores_reference = []
                answers_reference = []
                for ret in retrieved_ref:
                    score = judge.score_answer(original_question, ret['answer'])
                    scores_reference.append(score)
                    answers_reference.append(ret['answer'])
                
                # Score 4 answers from generated summary
                scores_generated = []
                answers_generated = []
                for ret in retrieved_gen:
                    score = judge.score_answer(original_question, ret['answer'])
                    scores_generated.append(score)
                    answers_generated.append(ret['answer'])
                
                # Store for metric calculation
                scores_original_all.append(scores_original)
                scores_reference_all.append(scores_reference)
                scores_generated_all.append(scores_generated)
                
                # Store result
                result = {
                    'qid': qid,
                    'model': model_name,
                    'original_question': original_question,
                    'reference_summary': reference_summary,
                    'generated_summary': generated_summary,
                    # Original query results
                    'answer_original_rank1': answers_original[0],
                    'answer_original_rank2': answers_original[1],
                    'answer_original_rank3': answers_original[2],
                    'answer_original_rank4': answers_original[3],
                    'score_original_rank1': scores_original[0],
                    'score_original_rank2': scores_original[1],
                    'score_original_rank3': scores_original[2],
                    'score_original_rank4': scores_original[3],
                    # Reference summary results
                    'answer_reference_rank1': answers_reference[0],
                    'answer_reference_rank2': answers_reference[1],
                    'answer_reference_rank3': answers_reference[2],
                    'answer_reference_rank4': answers_reference[3],
                    'score_reference_rank1': scores_reference[0],
                    'score_reference_rank2': scores_reference[1],
                    'score_reference_rank3': scores_reference[2],
                    'score_reference_rank4': scores_reference[3],
                    # Generated summary results
                    'answer_generated_rank1': answers_generated[0],
                    'answer_generated_rank2': answers_generated[1],
                    'answer_generated_rank3': answers_generated[2],
                    'answer_generated_rank4': answers_generated[3],
                    'score_generated_rank1': scores_generated[0],
                    'score_generated_rank2': scores_generated[1],
                    'score_generated_rank3': scores_generated[2],
                    'score_generated_rank4': scores_generated[3]
                }
                all_results.append(result)
            
            # Calculate metrics for all THREE query types
            metrics_original = calculate_metrics_topk(scores_original_all)
            metrics_reference = calculate_metrics_topk(scores_reference_all)
            metrics_generated = calculate_metrics_topk(scores_generated_all)
            
            model_metrics[model_name] = {
                'original_question': metrics_original,
                'reference_summary': metrics_reference,
                'generated_summary': metrics_generated
            }
            
            print(f"\n{model_name} Metrics:")
            print(f"  Original Questions:")
            print(f"    avgTop1={metrics_original['avgTop1']:.3f}, avgBest@4={metrics_original['avgBest@4']:.3f}, "
                  f"nDCG@4={metrics_original['nDCG@4']:.3f}")
            print(f"    succ@3={metrics_original['succ@3(3+)']:.3f}, succ@4={metrics_original['succ@4(4+)']:.3f}, "
                  f"MRR@4(3+)={metrics_original['MRR@4(3+)']:.3f}")
            
            print(f"  Reference Summaries:")
            print(f"    avgTop1={metrics_reference['avgTop1']:.3f}, avgBest@4={metrics_reference['avgBest@4']:.3f}, "
                  f"nDCG@4={metrics_reference['nDCG@4']:.3f}")
            print(f"    succ@3={metrics_reference['succ@3(3+)']:.3f}, succ@4={metrics_reference['succ@4(4+)']:.3f}, "
                  f"MRR@4(3+)={metrics_reference['MRR@4(3+)']:.3f}")
            
            print(f"  Generated Summaries:")
            print(f"    avgTop1={metrics_generated['avgTop1']:.3f}, avgBest@4={metrics_generated['avgBest@4']:.3f}, "
                  f"nDCG@4={metrics_generated['nDCG@4']:.3f}")
            print(f"    succ@3={metrics_generated['succ@3(3+)']:.3f}, succ@4={metrics_generated['succ@4(4+)']:.3f}, "
                  f"MRR@4(3+)={metrics_generated['MRR@4(3+)']:.3f}")
            
            # Clear model from memory
            del summarizer.model
            del summarizer
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"✗ Error processing {model_name}: {str(e)}")
            continue
    
    # Save detailed results
    print("\n" + "="*80)
    print("Saving Results...")
    print("="*80)
    
    # Save as CSV
    results_df = pd.DataFrame(all_results)
    results_csv_path = os.path.join(OUTPUT_DIR, "detailed_results_all_models_topk4.csv")
    results_df.to_csv(results_csv_path, index=False)
    print(f"✓ Detailed results saved to: {results_csv_path}")
    
    # Save metrics
    metrics_path = os.path.join(OUTPUT_DIR, "model_metrics_all_models_topk4.json")
    with open(metrics_path, 'w') as f:
        json.dump(model_metrics, f, indent=2)
    print(f"✓ Model metrics saved to: {metrics_path}")
    
    # Create summary table
    summary_rows = []
    for model_name, metrics in model_metrics.items():
        for query_type in ['original_question', 'reference_summary', 'generated_summary']:
            query_type_label = query_type.replace('_', ' ').title()
            summary_rows.append({
                'Model': model_name,
                'Query Type': query_type_label,
                **metrics[query_type]
            })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(OUTPUT_DIR, "metrics_summary_all_models_topk4.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"✓ Metrics summary saved to: {summary_path}")
    
    print("\n" + "="*80)
    print("Pipeline Complete!")
    print("="*80)
    print("\nProcessed All 10 Models:")
    print("  1. Gemma-7B")
    print("  2. Mistral-7B-v0.3")
    print("  3. Llama-3.1-8B")
    print("  4. Llama-3.2-3B")
    print("  5. Pegasus-Large")
    print("  6. T5-Base")
    print("  7. Qwen2-7B")
    print("  8. DeepSeek-7B")
    print("  9. BART-Large")
    print(" 10. ProphetNet-Large")

if __name__ == "__main__":
    main()


echo "✓ Created ALL 10 models pipeline with Top-K=4 and expanded metrics"
