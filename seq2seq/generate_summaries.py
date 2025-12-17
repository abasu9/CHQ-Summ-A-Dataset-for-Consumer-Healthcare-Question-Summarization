"""
Generate summaries using trained models
Outputs Excel files with: original question, human summary, generated summary
"""

import os
import xml.etree.ElementTree as ET
import torch
from transformers import (
    ProphetNetTokenizer,
    ProphetNetForConditionalGeneration,
    PegasusTokenizer,
    PegasusForConditionalGeneration,
    BartTokenizer,
    BartForConditionalGeneration,
    T5Tokenizer,
    T5ForConditionalGeneration,
)
from tqdm import tqdm
import logging
import pandas as pd
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestDataLoader:
    """Load test data"""
    
    def __init__(self, xml_file_path):
        self.xml_file_path = xml_file_path
        self.data = []
        
    def parse(self):
        """Parse XML file"""
        logger.info(f"Parsing test file: {self.xml_file_path}")
        
        tree = ET.parse(self.xml_file_path)
        root = tree.getroot()
        
        documents = root.findall('.//document')
        
        for doc in documents:
            doc_id = doc.find('id')
            subject = doc.find('original_question_subject')
            content = doc.find('original_question_content')
            summary = doc.find('reference_summary')
            
            if subject is not None:
                # Combine subject and content
                question_text = subject.text.strip() if subject.text else ""
                
                if content is not None and content.text:
                    content_text = content.text.strip()
                    if content_text and content_text.lower() != question_text.lower():
                        question_text = f"{question_text} {content_text}"
                
                item = {
                    'id': doc_id.text if doc_id is not None else '',
                    'question': question_text
                }
                
                # Include reference for output file (not for model)
                if summary is not None and summary.text:
                    item['reference_summary'] = summary.text.strip()
                
                self.data.append(item)
        
        logger.info(f"Loaded {len(self.data)} test examples")
        return self.data


def load_model_and_tokenizer(model_path):
    """Load trained model and tokenizer"""
    logger.info(f"Loading model from {model_path}")
    
    # Detect model type from path
    if 'prophetnet' in model_path.lower():
        tokenizer = ProphetNetTokenizer.from_pretrained(model_path)
        model = ProphetNetForConditionalGeneration.from_pretrained(model_path)
        model_type = 'prophetnet'
    elif 'pegasus' in model_path.lower():
        tokenizer = PegasusTokenizer.from_pretrained(model_path)
        model = PegasusForConditionalGeneration.from_pretrained(model_path)
        model_type = 'pegasus'
    elif 'bart' in model_path.lower():
        tokenizer = BartTokenizer.from_pretrained(model_path)
        model = BartForConditionalGeneration.from_pretrained(model_path)
        model_type = 'bart'
    elif 't5' in model_path.lower():
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        model = T5ForConditionalGeneration.from_pretrained(model_path)
        model_type = 't5'
    else:
        raise ValueError(f"Unknown model type in path: {model_path}")
    
    return model, tokenizer, model_type


def generate_summaries(
    model,
    tokenizer,
    test_data,
    model_type='seq2seq',
    max_source_length=512,
    max_target_length=128,
    batch_size=16,
    num_beams=4
):
    """Generate summaries for test data"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    model.to(device)
    model.eval()
    
    summaries = []
    
    for i in tqdm(range(0, len(test_data), batch_size), desc="Generating summaries"):
        batch = test_data[i:i+batch_size]
        questions = [item['question'] for item in batch]
        
        # Add T5 prefix if needed
        if model_type == 't5':
            questions = ["summarize: " + q for q in questions]
        
        # Tokenize
        inputs = tokenizer(
            questions,
            max_length=max_source_length,
            truncation=True,
            padding=True,
            return_tensors='pt'
        ).to(device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_target_length,
                num_beams=num_beams,
                length_penalty=2.0,
                early_stopping=True,
                no_repeat_ngram_size=3
            )
        
        # Decode
        batch_summaries = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        summaries.extend(batch_summaries)
    
    return summaries


def save_to_excel(test_data, predictions, output_file):
    """Save results to Excel file"""
    
    logger.info(f"Saving results to Excel: {output_file}")
    
    # Prepare data for Excel
    excel_data = []
    for item, pred in zip(test_data, predictions):
        row = {
            'ID': item['id'],
            'Original Question': item['question'],
            'Human Summary': item.get('reference_summary', ''),
            'Generated Summary': pred
        }
        excel_data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(excel_data)
    
    # Save to Excel
    df.to_excel(output_file, index=False, engine='openpyxl')
    
    logger.info(f"✓ Excel file saved: {output_file}")
    logger.info(f"  Rows: {len(df)}")
    logger.info(f"  Columns: {', '.join(df.columns)}")


def generate_for_model(model_path, test_xml, output_dir):
    """Run generation for a single model"""
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Generating summaries: {model_path}")
    logger.info(f"{'='*80}\n")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load test data
    test_loader = TestDataLoader(test_xml)
    test_data = test_loader.parse()
    
    # Load model
    model, tokenizer, model_type = load_model_and_tokenizer(model_path)
    
    # Generate summaries
    start_time = datetime.now()
    predictions = generate_summaries(
        model=model,
        tokenizer=tokenizer,
        test_data=test_data,
        model_type=model_type,
        max_source_length=512,
        max_target_length=128,
        batch_size=16,
        num_beams=4
    )
    generation_time = (datetime.now() - start_time).total_seconds()
    
    logger.info(f"Generation completed in {generation_time:.2f} seconds")
    logger.info(f"Average time per example: {generation_time/len(test_data):.3f} seconds")
    
    # Save to Excel
    excel_file = os.path.join(output_dir, "summaries.xlsx")
    save_to_excel(test_data, predictions, excel_file)
    
    return excel_file


def main():
    """Main function for batch generation"""
    
    BASE_RESULTS_DIR = "./yahoo_l6_results"
    TEST_XML = "/mnt/user-data/uploads/merged_test_found_only.xml"
    
    # Model directories
    model_dirs = [
        "prophetnet-large-uncased",
        "pegasus-large",
        "bart-large",
        "t5-base"
    ]
    
    logger.info("="*80)
    logger.info("GENERATING SUMMARIES FOR ALL MODELS")
    logger.info("="*80)
    
    generated_files = []
    
    for model_dir in model_dirs:
        model_path = os.path.join(BASE_RESULTS_DIR, model_dir)
        
        if not os.path.exists(model_path):
            logger.warning(f"Model directory not found: {model_path}")
            logger.warning("Skipping...")
            continue
        
        output_dir = model_path
        
        try:
            excel_file = generate_for_model(model_path, TEST_XML, output_dir)
            generated_files.append({
                'model': model_dir,
                'file': excel_file
            })
        except Exception as e:
            logger.error(f"Error generating for {model_dir}: {str(e)}")
            continue
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("GENERATION SUMMARY")
    logger.info("="*80)
    
    for item in generated_files:
        logger.info(f"\nModel: {item['model']}")
        logger.info(f"Summaries: {item['file']}")
    
    logger.info("\n✓ All summaries generated!")


if __name__ == "__main__":
    main()
