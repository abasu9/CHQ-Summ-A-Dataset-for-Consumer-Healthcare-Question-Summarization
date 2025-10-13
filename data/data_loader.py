#!/usr/bin/env python3
"""
Data Loading and Processing for CHQ-Summ Dataset
Compatible with Transformers 4.35+
"""

import os
import json
import re
from typing import Dict, List, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

try:
    from lxml import etree as LET
    LXML_AVAILABLE = True
except ImportError:
    LET = None
    LXML_AVAILABLE = False

import xml.etree.ElementTree as ET


SUMMARY_TAG_CANDIDATES = [
    'summary', 'abstract', 'synopsis', 'answersummary',
    'answer_summary', 'bestanswersummary', 'shortanswer', 'short_answer',
]


# ============ TEXT CLEANING ============

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


# ============ XML PARSING ============

def parse_yahoo_xml(xml_path: str) -> Dict[str, Dict]:
    """
    Parse Yahoo L6 XML file.
    
    Args:
        xml_path: Path to Yahoo L6 XML file (yahool6.xml or FullOct2007.xml)
        
    Returns:
        Dictionary mapping document IDs to document data
    """
    print(f"📂 Loading XML: {xml_path}")
    
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
        if not documents:
            # Try vespaadd tag for Yahoo L6 format
            for event, elem in ET.iterparse(xml_path, events=('end',)):
                if elem.tag == 'vespaadd':
                    doc = elem.find('document')
                    if doc is not None:
                        documents.append(doc)
        
        print(f"✓ Found {len(documents)} documents")
        
        for idx, doc in enumerate(documents):
            doc_id = str(idx)
            
            # Try multiple ID tags
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
            
            # Construct full text
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
        
        print(f"✓ Parsed {len(data_dict)} valid documents")
        return data_dict
        
    except Exception as e:
        print(f"❌ Error parsing XML: {e}")
        import traceback
        traceback.print_exc()
        return {}


# ============ JSON LOADING ============

def load_annotations(json_path: str) -> List:
    """
    Load train/val/test split annotations.
    
    Args:
        json_path: Path to JSON file with IDs or full data
        
    Returns:
        List of IDs or full data dictionaries
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            return []
        
        # Check if full data with human_summary
        if data and isinstance(data[0], dict) and 'human_summary' in data[0]:
            print(f"✓ {os.path.basename(json_path)} contains full CHQ-Summ data")
            return [{'id': item['id'], 'summary': item['human_summary'], 
                    'question_focus': item.get('question_focus', []),
                    'question_type': item.get('question_type', [])} for item in data]
        
        # Check if already has text and summary
        if data and isinstance(data[0], dict) and 'text' in data[0] and 'summary' in data[0]:
            print(f"✓ {os.path.basename(json_path)} has complete data")
            return data
        
        # Otherwise, treat as ID list
        ids = [str(item if not isinstance(item, dict) else item.get('id', idx))
               for idx, item in enumerate(data)]
        print(f"✓ Loaded {len(ids)} IDs from {os.path.basename(json_path)}")
        return ids
        
    except Exception as e:
        print(f"❌ Error loading {json_path}: {e}")
        return []


# ============ DATASET LOADING ============

def load_yahoo_dataset(data_dir: str = 'data/') -> Tuple[List, List, List]:
    """
    Load Yahoo CHQ dataset with train/val/test splits.
    
    Args:
        data_dir: Directory containing data files
        
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    data_dir = Path(data_dir)
    
    print("="*70)
    print("📊 LOADING YAHOO CHQ-SUMM DATASET")
    print("="*70)
    
    # Try loading pre-processed JSON first
    try:
        test_path = data_dir / 'test.json'
        with open(test_path, 'r', encoding='utf-8') as f:
            sample = json.load(f)
        
        if sample and isinstance(sample[0], dict) and 'text' in sample[0]:
            with open(data_dir / 'train.json', 'r', encoding='utf-8') as f:
                train_data = json.load(f)
            with open(data_dir / 'test.json', 'r', encoding='utf-8') as f:
                test_data = json.load(f)
            with open(data_dir / 'val.json', 'r', encoding='utf-8') as f:
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
        print(f"   Looking for: {', '.join(xml_candidates)}")
        return None, None, None
    
    yahoo_data = parse_yahoo_xml(str(xml_path))
    
    if not yahoo_data:
        print(f"❌ No data extracted from XML")
        return None, None, None
    
    # Load splits
    train_ids = load_annotations(str(data_dir / 'train.json'))
    test_ids = load_annotations(str(data_dir / 'test.json'))
    val_ids = load_annotations(str(data_dir / 'val.json'))
    
    # Build datasets
    if train_ids and isinstance(train_ids[0], dict) and 'text' in train_ids[0]:
        train_data = train_ids
        test_data = test_ids
        val_data = val_ids
    else:
        train_data = []
        test_data = []
        val_data = []
        
        for id_item in train_ids:
            if isinstance(id_item, dict):
                doc_id = id_item['id']
                if str(doc_id) in yahoo_data:
                    data_entry = yahoo_data[str(doc_id)].copy()
                    data_entry['summary'] = id_item.get('summary', data_entry['summary'])
                    train_data.append(data_entry)
            else:
                if str(id_item) in yahoo_data:
                    train_data.append(yahoo_data[str(id_item)])
        
        for id_item in test_ids:
            if isinstance(id_item, dict):
                doc_id = id_item['id']
                if str(doc_id) in yahoo_data:
                    data_entry = yahoo_data[str(doc_id)].copy()
                    data_entry['summary'] = id_item.get('summary', data_entry['summary'])
                    test_data.append(data_entry)
            else:
                if str(id_item) in yahoo_data:
                    test_data.append(yahoo_data[str(id_item)])
        
        for id_item in val_ids:
            if isinstance(id_item, dict):
                doc_id = id_item['id']
                if str(doc_id) in yahoo_data:
                    data_entry = yahoo_data[str(doc_id)].copy()
                    data_entry['summary'] = id_item.get('summary', data_entry['summary'])
                    val_data.append(data_entry)
            else:
                if str(id_item) in yahoo_data:
                    val_data.append(yahoo_data[str(id_item)])
    
    print(f"✓ Dataset loaded successfully")
    print(f"  Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")
    return train_data, val_data, test_data


# ============ LEGACY SUPPORT ============

def read_langs(file_name: Tuple[str, str], max_src=None, max_tgt=None):
    """
    Legacy function for reading source/target files.
    Compatible with original codebase.
    """
    data = []
    try:
        with open(file_name[0], "r", encoding='utf-8') as f1, \
             open(file_name[1], "r", encoding='utf-8') as f2:
            for src_line, tgt_line in zip(f1.readlines(), f2.readlines()):
                src = src_line.strip()
                tgt = tgt_line.strip()
                
                d = {
                    "x": src,
                    "y": tgt
                }
                
                if max_src is not None:
                    d["x"] = ' '.join(d["x"].strip().split()[:max_src])
                
                if max_tgt is not None:
                    d["y"] = ' '.join(d["y"].strip().split()[:max_tgt])
                
                d["x_len"] = len(d["x"].strip().split())
                d["y_len"] = len(d["y"].strip().split())
                data.append(d)
    except Exception as e:
        print(f"❌ Error reading files: {e}")
        return [], 0, 0
    
    max_src_len = max([d["x_len"] for d in data]) if data else 0
    max_tgt_len = max([d["y_len"] for d in data]) if data else 0
    
    print(f"✓ Loaded {len(data)} samples")
    print(f"  Max source length: {max_src_len}")
    print(f"  Max target length: {max_tgt_len}")
    
    return data, max_src_len, max_tgt_len


def get_seq(data):
    """Extract x and y sequences from data."""
    x_seq = [d["x"] for d in data]
    y_seq = [d["y"] for d in data]
    return x_seq, y_seq


# ============ PYTORCH DATASET ============

class CHQSummDataset(Dataset):
    """PyTorch Dataset for CHQ-Summ."""
    
    def __init__(self, data, tokenizer, source_len, summ_len):
        self.tokenizer = tokenizer
        self.source_len = source_len
        self.summ_len = summ_len
        
        if isinstance(data[0], dict) and 'x' in data[0]:
            x_seq, y_seq = get_seq(data)
            self.sources = x_seq
            self.targets = y_seq
        else:
            self.sources = [d['text'] for d in data]
            self.targets = [d['summary'] for d in data]
    
    def __len__(self):
        return len(self.sources)
    
    def __getitem__(self, index):
        source_text = str(self.sources[index])
        target_text = str(self.targets[index])
        
        # Tokenize with modern API
        source = self.tokenizer(
            source_text,
            max_length=self.source_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        target = self.tokenizer(
            target_text,
            max_length=self.summ_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()
        
        return {
            'source_ids': source_ids.to(dtype=torch.long),
            'source_mask': source_mask.to(dtype=torch.long),
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_ids.to(dtype=torch.long),
            'target_mask': target_mask.to(dtype=torch.long)
        }


# Legacy alias
CustomDataset = CHQSummDataset


# ============ UTILITY FUNCTIONS ============

def create_source_target_files(data_dir: str = 'data/'):
    """
    Create .source and .target files from JSON data.
    Useful for training pipeline compatibility.
    """
    train_data, val_data, test_data = load_yahoo_dataset(data_dir)
    
    if not train_data:
        print("❌ No data loaded")
        return
    
    data_dir = Path(data_dir)
    
    for split_name, split_data in [('train', train_data), ('val', val_data), ('test', test_data)]:
        source_file = data_dir / f'{split_name}.source'
        target_file = data_dir / f'{split_name}.target'
        
        with open(source_file, 'w', encoding='utf-8') as f_src, \
             open(target_file, 'w', encoding='utf-8') as f_tgt:
            for item in split_data:
                f_src.write(_clean_text(item['text']) + '\n')
                f_tgt.write(_clean_text(item['summary']) + '\n')
        
        print(f"✓ Created {split_name}.source and {split_name}.target")


if __name__ == "__main__":
    # Test data loading
    train_data, val_data, test_data = load_yahoo_dataset('data/')
    
    if train_data:
        print(f"\n✓ Successfully loaded dataset")
        print(f"  Sample question: {train_data[0]['text'][:100]}...")
        print(f"  Sample summary: {train_data[0]['summary']}")
