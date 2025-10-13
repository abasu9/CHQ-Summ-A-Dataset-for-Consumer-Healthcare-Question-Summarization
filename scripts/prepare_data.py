#!/usr/bin/env python3
"""
Prepare CHQ-Summ Data
Extract Yahoo L6 questions and create .source/.target files
"""

import argparse
import json
import os
from pathlib import Path

try:
    from lxml import etree
    LXML_AVAILABLE = True
except ImportError:
    etree = None
    LXML_AVAILABLE = False
    print("⚠️  Warning: lxml not available, using xml.etree instead")
    import xml.etree.ElementTree as etree


def clean_text(text):
    """Clean text by removing unwanted whitespace."""
    if not text:
        return ""
    text = text.replace('\n', ' ')
    text = text.replace('\t', ' ')
    text = text.replace('  ', ' ')
    text = text.replace('   ', ' ')
    return text.strip()


def read_yahoo_data(yahoo_path):
    """
    Read Yahoo L6 dataset XML file.
    
    Args:
        yahoo_path: Path to FullOct2007.xml or yahool6.xml
        
    Returns:
        Dictionary mapping question IDs to data
    """
    print(f"📂 Reading Yahoo L6 data from: {yahoo_path}")
    
    if not os.path.exists(yahoo_path):
        print(f"❌ File not found: {yahoo_path}")
        return {}
    
    data_items = {}
    ctr = 0
    
    try:
        if LXML_AVAILABLE:
            for event, elem in etree.iterparse(
                yahoo_path, 
                tag="vespaadd", 
                encoding='utf-8', 
                recover=True
            ):
                doc = elem.find('document')
                try:
                    meta_data = {}
                    q_id = doc.findtext('uri')
                    question = doc.findtext('subject')
                    content = doc.findtext('content')
                    
                    if q_id:
                        meta_data['id'] = q_id
                        meta_data['question'] = question or ""
                        meta_data['content'] = content or ""
                        data_items[meta_data['id']] = meta_data
                        ctr += 1
                        
                        if ctr % 1000 == 0:
                            print(f"  Read {ctr} questions...")
                    
                    # Clear element to save memory
                    elem.clear()
                    
                except Exception as e:
                    continue
        else:
            # Fallback to standard xml parser
            tree = etree.parse(yahoo_path)
            root = tree.getroot()
            
            for doc in root.findall('.//document'):
                try:
                    meta_data = {}
                    q_id = doc.findtext('uri') or doc.findtext('id')
                    question = doc.findtext('subject')
                    content = doc.findtext('content')
                    
                    if q_id:
                        meta_data['id'] = q_id
                        meta_data['question'] = question or ""
                        meta_data['content'] = content or ""
                        data_items[meta_data['id']] = meta_data
                        ctr += 1
                        
                        if ctr % 1000 == 0:
                            print(f"  Read {ctr} questions...")
                except:
                    continue
    
    except Exception as e:
        print(f"❌ Error reading XML: {e}")
        import traceback
        traceback.print_exc()
        return {}
    
    print(f"✓ Loaded {len(data_items)} questions from Yahoo L6")
    return data_items


def process_chq_summ(yahoo_data_items, chq_summ_file, mode, output_dir):
    """
    Process CHQ-Summ annotations and create .source/.target files.
    
    Args:
        yahoo_data_items: Dictionary of Yahoo L6 data
        chq_summ_file: Path to CHQ-Summ JSON file (train/val/test)
        mode: Split name (train/val/test)
        output_dir: Directory to save output files
    """
    print(f"\n📝 Processing {mode} split...")
    
    if not os.path.exists(chq_summ_file):
        print(f"❌ File not found: {chq_summ_file}")
        return
    
    with open(chq_summ_file, 'r', encoding='utf-8') as f:
        chq_dataset = json.load(f)
    
    sources = []
    targets = []
    ids = []
    
    missing_count = 0
    for chq_item in chq_dataset:
        item_id = chq_item.get('id')
        
        if not item_id or item_id not in yahoo_data_items:
            missing_count += 1
            continue
        
        meta_data = yahoo_data_items[item_id]
        question = meta_data.get('question', '')
        content = meta_data.get('content', '')
        
        # Combine question and content
        source_text = f"{question} {content}".strip()
        
        # Get summary
        target_text = chq_item.get('human_summary', '')
        
        if source_text and target_text:
            ids.append(item_id)
            sources.append(clean_text(source_text))
            targets.append(clean_text(target_text))
    
    # Save files
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    id_file = output_dir / f'{mode}.id'
    source_file = output_dir / f'{mode}.source'
    target_file = output_dir / f'{mode}.target'
    
    with open(id_file, 'w', encoding='utf-8') as f_id, \
         open(source_file, 'w', encoding='utf-8') as f_src, \
         open(target_file, 'w', encoding='utf-8') as f_tgt:
        
        for doc_id, src, tgt in zip(ids, sources, targets):
            f_id.write(doc_id.strip() + '\n')
            f_src.write(src + '\n')
            f_tgt.write(tgt + '\n')
    
    print(f"✓ Processed {len(sources)} samples")
    if missing_count > 0:
        print(f"  ⚠️  {missing_count} items not found in Yahoo L6 data")
    print(f"✓ Saved to: {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description='Prepare CHQ-Summ dataset from Yahoo L6 XML'
    )
    parser.add_argument(
        '--Yahoo_data_path',
        type=str,
        required=True,
        help='Path to Yahoo L6 XML file (FullOct2007.xml or yahool6.xml)'
    )
    parser.add_argument(
        '--CHQ_summ_path',
        type=str,
        required=True,
        help='Path to directory containing train.json, val.json, test.json'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("🚀 CHQ-SUMM DATA PREPARATION")
    print("="*70)
    
    # Read Yahoo L6 data
    yahoo_data = read_yahoo_data(args.Yahoo_data_path)
    
    if not yahoo_data:
        print("❌ Failed to load Yahoo L6 data")
        return
    
    # Process each split
    chq_path = Path(args.CHQ_summ_path)
    
    for split in ['train', 'val', 'test']:
        json_file = chq_path / f'{split}.json'
        if json_file.exists():
            process_chq_summ(
                yahoo_data,
                str(json_file),
                split,
                args.CHQ_summ_path
            )
        else:
            print(f"⚠️  {split}.json not found, skipping...")
    
    print("\n✅ Data preparation complete!")
    print(f"📁 Files saved to: {args.CHQ_summ_path}/")


if __name__ == '__main__':
    main()
