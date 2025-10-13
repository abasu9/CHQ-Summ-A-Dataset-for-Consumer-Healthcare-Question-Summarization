# Data Directory

This directory contains data loading utilities and the dataset files.

## Required Files

Place these files here after downloading:

- `yahool6.xml` or `FullOct2007.xml` - Yahoo L6 dataset (download from Yahoo Webscope)
- `train.json` - CHQ-Summ training split (download from OSF)
- `val.json` - CHQ-Summ validation split (download from OSF)
- `test.json` - CHQ-Summ test split (download from OSF)

## Generated Files

After running `scripts/prepare_data.py`, these files will be created:

- `train.source` / `train.target`
- `val.source` / `val.target`
- `test.source` / `test.target`
- `train.id` / `val.id` / `test.id`

## Usage
```python
from data.data_loader import load_yahoo_dataset

train_data, val_data, test_data = load_yahoo_dataset('data/')
