"""
Training script for Yahoo L6 question summarization
Trains: ProphetNet, PEGASUS, BART, T5
"""

import os
import xml.etree.ElementTree as ET
import torch
import sys
from transformers import (
    ProphetNetTokenizer,
    ProphetNetForConditionalGeneration,
    PegasusTokenizer,
    PegasusForConditionalGeneration,
    BartTokenizer,
    BartForConditionalGeneration,
    T5Tokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import Dataset as HFDataset
import evaluate
import numpy as np
import logging
from datetime import datetime
import json
from rouge_score import rouge_scorer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class YahooL6XMLParser:
    """Parse Yahoo L6 XML files"""
    
    def __init__(self, xml_file_path):
        self.xml_file_path = xml_file_path
        self.data = []
        
    def parse(self):
        """Parse XML and extract questions and summaries"""
        logger.info(f"Parsing XML file: {self.xml_file_path}")
        
        tree = ET.parse(self.xml_file_path)
        root = tree.getroot()
        
        documents = root.findall('.//document')
        
        for doc in documents:
            doc_id = doc.find('id')
            subject = doc.find('original_question_subject')
            content = doc.find('original_question_content')
            summary = doc.find('reference_summary')
            
            if subject is not None and summary is not None:
                # Combine subject and content
                question_text = subject.text.strip() if subject.text else ""
                
                if content is not None and content.text:
                    content_text = content.text.strip()
                    if content_text and content_text.lower() != question_text.lower():
                        question_text = f"{question_text} {content_text}"
                
                summary_text = summary.text.strip() if summary.text else ""
                
                if question_text and summary_text:
                    self.data.append({
                        'id': doc_id.text if doc_id is not None else '',
                        'question': question_text,
                        'summary': summary_text
                    })
        
        logger.info(f"Loaded {len(self.data)} examples from {self.xml_file_path}")
        return self.data


def prepare_dataset_for_model(data, tokenizer, max_source_length=512, max_target_length=128, model_type='seq2seq'):
    """Prepare dataset for training"""
    
    def preprocess_function(examples):
        # For T5, add prefix
        if model_type == 't5':
            inputs = ["summarize: " + q for q in examples['question']]
        else:
            inputs = examples['question']
        
        # Tokenize inputs
        model_inputs = tokenizer(
            inputs,
            max_length=max_source_length,
            truncation=True,
            padding='max_length'
        )
        
        # Tokenize targets
        labels = tokenizer(
            examples['summary'],
            max_length=max_target_length,
            truncation=True,
            padding='max_length'
        )
        
        model_inputs['labels'] = labels['input_ids']
        return model_inputs
    
    # Convert to HuggingFace Dataset
    hf_dataset = HFDataset.from_dict({
        'question': [item['question'] for item in data],
        'summary': [item['summary'] for item in data]
    })
    
    # Apply preprocessing
    tokenized_dataset = hf_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=hf_dataset.column_names
    )
    
    return tokenized_dataset


def compute_metrics(eval_pred, tokenizer):
    """Compute ROUGE metrics during training"""
    predictions, labels = eval_pred
    
    # Decode predictions with error handling
    decoded_preds = []
    for pred in predictions:
        try:
            decoded = tokenizer.decode(pred, skip_special_tokens=True)
            decoded_preds.append(decoded if decoded else "")
        except:
            decoded_preds.append("")
    
    # Replace -100 in labels
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    # Decode labels with error handling
    decoded_labels = []
    for label in labels:
        try:
            decoded = tokenizer.decode(label, skip_special_tokens=True)
            decoded_labels.append(decoded if decoded else "")
        except:
            decoded_labels.append("")
    
    # Filter out None values and handle edge cases
    filtered_preds = [p if p is not None else "" for p in decoded_preds]
    filtered_labels = [l if l is not None else "" for l in decoded_labels]
    
    # Compute ROUGE scores using rouge_score
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    
    rouge1_scores = []
    rougeL_scores = []
    
    for pred, ref in zip(filtered_preds, filtered_labels):
        # Only compute if both are non-empty after stripping
        if pred.strip() and ref.strip():
            try:
                scores = scorer.score(ref, pred)
                rouge1_scores.append(scores['rouge1'].fmeasure)
                rougeL_scores.append(scores['rougeL'].fmeasure)
            except:
                # Skip pairs that cause scoring errors
                continue
    
    return {
        'rouge1': round(np.mean(rouge1_scores) * 100, 4) if rouge1_scores else 0.0,
        'rougeL': round(np.mean(rougeL_scores) * 100, 4) if rougeL_scores else 0.0
    }


def train_model(
    model_name,
    train_data,
    eval_data,
    output_dir,
    num_train_epochs=3,
    batch_size=8,
    learning_rate=5e-5,
    max_source_length=512,
    max_target_length=128
):
    """Train a single model"""
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Training {model_name}")
    logger.info(f"{'='*80}\n")
    
    # Load model and tokenizer
    logger.info(f"Loading model and tokenizer...")
    
    model_type = 'seq2seq'
    if 'prophetnet' in model_name.lower():
        tokenizer = ProphetNetTokenizer.from_pretrained(model_name)
        model = ProphetNetForConditionalGeneration.from_pretrained(model_name, use_cache=False)
    elif 'pegasus' in model_name.lower():
        tokenizer = PegasusTokenizer.from_pretrained(model_name)
        model = PegasusForConditionalGeneration.from_pretrained(model_name, use_cache=False)
    elif 'bart' in model_name.lower():
        tokenizer = BartTokenizer.from_pretrained(model_name)
        model = BartForConditionalGeneration.from_pretrained(model_name, use_cache=False)
    elif 't5' in model_name.lower():
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name, use_cache=False)
        model_type = 't5'
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Prepare datasets
    logger.info("Preparing datasets...")
    train_dataset = prepare_dataset_for_model(
        train_data, tokenizer, max_source_length, max_target_length, model_type
    )
    eval_dataset = prepare_dataset_for_model(
        eval_data, tokenizer, max_source_length, max_target_length, model_type
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # Training arguments - optimized for g5.16xlarge
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=num_train_epochs,
        predict_with_generate=True,
        fp16=True,
        push_to_hub=False,
        logging_dir=f"{output_dir}/logs",
        logging_steps=50,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="rougeL",
        greater_is_better=True,
        generation_max_length=max_target_length,
        generation_num_beams=2,  # Reduced from 4 for stability
        dataloader_num_workers=4,
        gradient_accumulation_steps=1,
    )
    
    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, tokenizer)
    )
    
    # Train
    logger.info("Starting training...")
    start_time = datetime.now()
    train_result = trainer.train()
    training_time = (datetime.now() - start_time).total_seconds()
    
    # Save model
    logger.info("Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # Save training info
    train_info = {
        'model_name': model_name,
        'training_time_seconds': training_time,
        'train_samples': len(train_data),
        'eval_samples': len(eval_data),
        'epochs': num_train_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
    }
    
    with open(f"{output_dir}/training_info.json", 'w') as f:
        json.dump(train_info, f, indent=2)
    
    # Evaluate
    logger.info("Running final evaluation...")
    metrics = trainer.evaluate()
    
    # Save metrics
    with open(f"{output_dir}/training_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"\nTraining completed in {training_time/60:.2f} minutes")
    logger.info(f"Final ROUGE-L: {metrics.get('eval_rougeL', 'N/A')}")
    
    return trainer, metrics


def main():
    """Main execution function"""
    
    # Configuration
    TRAIN_XML = "/home/ubuntu/yahoo_l6_project/data/merged_train_found_only.xml"
    TEST_XML = "/home/ubuntu/yahoo_l6_project/data/merged_test_found_only.xml"
    
    BASE_OUTPUT_DIR = "./yahoo_l6_results"
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    
    # Models to train with their specific hyperparameters
    ALL_MODELS = {
        "microsoft/prophetnet-large-uncased": {"lr": 5e-5, "epochs": 5},
        "google/pegasus-large": {"lr": 2e-5, "epochs": 4},  # Pre-trained for summarization
        "facebook/bart-large": {"lr": 3e-5, "epochs": 5},   # Balanced approach
        "google-t5/t5-base": {"lr": 5e-5, "epochs": 5}
    }
    
    # Parse command line argument
    if len(sys.argv) > 1:
        selected_model = sys.argv[1].lower()
        # Find matching model
        MODELS = {}
        for model_name, params in ALL_MODELS.items():
            if selected_model in model_name.lower():
                MODELS[model_name] = params
                break
        if not MODELS:
            logger.error(f"Model '{selected_model}' not found. Available models:")
            for m in ALL_MODELS.keys():
                logger.error(f"  - {m}")
            return
    else:
        MODELS = ALL_MODELS
    
    # Global hyperparameters (same for all models)
    BATCH_SIZE = 4  # Safe for large seq2seq models on limited VRAM
    MAX_SOURCE_LENGTH = 512  # Good for longer questions; can reduce to 256 for speed
    MAX_TARGET_LENGTH = 64  # Reduced from 128; sufficient for question summarization
    WARMUP_STEPS = 500  # Helps training stability
    
    # Parse data
    logger.info("="*80)
    logger.info("LOADING DATASETS")
    logger.info("="*80)
    
    train_parser = YahooL6XMLParser(TRAIN_XML)
    train_data = train_parser.parse()
    
    test_parser = YahooL6XMLParser(TEST_XML)
    test_data = test_parser.parse()
    
    logger.info(f"\nTrain samples: {len(train_data)}")
    logger.info(f"Test samples: {len(test_data)}")
    
    # Train each model with its specific hyperparameters
    results_summary = []
    
    for model_name, hyperparams in MODELS.items():
        try:
            model_short_name = model_name.split('/')[-1]
            output_dir = os.path.join(BASE_OUTPUT_DIR, model_short_name)
            
            logger.info(f"\nTraining {model_short_name} with LR={hyperparams['lr']}, Epochs={hyperparams['epochs']}")
            
            trainer, metrics = train_model(
                model_name=model_name,
                train_data=train_data,
                eval_data=test_data,
                output_dir=output_dir,
                num_train_epochs=hyperparams['epochs'],
                batch_size=BATCH_SIZE,
                learning_rate=hyperparams['lr'],
                max_source_length=MAX_SOURCE_LENGTH,
                max_target_length=MAX_TARGET_LENGTH
            )
            
            results_summary.append({
                'model': model_name,
                'output_dir': output_dir,
                'rougeL': metrics.get('eval_rougeL', None),
                'status': 'SUCCESS'
            })
            
        except Exception as e:
            logger.error(f"Error training {model_name}: {str(e)}")
            results_summary.append({
                'model': model_name,
                'status': 'FAILED',
                'error': str(e)
            })
    
    # Save summary
    logger.info("\n" + "="*80)
    logger.info("TRAINING SUMMARY")
    logger.info("="*80)
    
    for result in results_summary:
        logger.info(f"\nModel: {result['model']}")
        logger.info(f"Status: {result['status']}")
        if result['status'] == 'SUCCESS':
            logger.info(f"ROUGE-L: {result['rougeL']}")
            logger.info(f"Output: {result['output_dir']}")
    
    summary_file = os.path.join(BASE_OUTPUT_DIR, "training_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    logger.info(f"\nTraining summary saved to: {summary_file}")
    logger.info("\nAll training completed!")


if __name__ == "__main__":
    main()
