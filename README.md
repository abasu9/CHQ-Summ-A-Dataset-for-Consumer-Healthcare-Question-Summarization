
# CHQ-Summ: A Dataset for Consumer Healthcare Question Summarization

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-yellow.svg)](https://huggingface.co/transformers/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-arXiv-green.svg)](https://arxiv.org/abs/2206.06581)

**Evaluation framework for the CHQ-Summ dataset for consumer healthcare question summarization.**

> **Original Paper**: [CHQ-Summ: A Dataset for Consumer Healthcare Question Summarization](https://arxiv.org/abs/2206.06581)  
> **Original Authors**: Abhishek Basu, Shweta Yadav, Deepak Gupta, Dina Demner-Fushman  
> **Institutions**: University of Illinois Chicago & National Library of Medicine (NIH)

---

## Overview

**CHQ-Summ** is a benchmark dataset and evaluation framework for **Consumer Healthcare Question Summarization**.

Consumers frequently post long, descriptive medical questions online, which are difficult for automated systems to interpret and retrieve accurate answers for. CHQ-Summ addresses this challenge by providing **domain-expert annotated question–summary pairs** that distill the core medical intent of verbose consumer queries.

This repository **reproduces and extends** the original work:

> **CHQ-Summ: A Dataset for Consumer Healthcare Question Summarization**  
> Yadav et al., *Scientific Data* (under review, 2025)

This implementation includes:
- Fine-tuning of **classical seq2seq models** (BART, PEGASUS, ProphetNet, T5)
- Zero-, two-, and five-shot **inference-only evaluation** of **open-weight LLMs**
- A unified **multi-metric evaluation framework** covering lexical, semantic, and factual quality

---

## Dataset

### CHQ-Summ Statistics

Dataset Split:
```

├── Train: 1,000 samples
├── Validation: 400 samples
└── Test: 400 samples (*107 in some releases)

```

Content Statistics:
```

├── Avg Question Length: ~177 words (~10 sentences)
├── Avg Summary Length: ~13 words (1–2 sentences)
├── Question Focus: 1,788 distinct entities
├── Question Types: 33 types (Information, Treatment, Cause, Symptoms, …)
└── Source: Yahoo! Answers — Healthcare category

````

> *Some releases include 107 test samples instead of 400.*

---

### Dataset Access

1. **CHQ-Summ annotations (OSF)**  
   https://doi.org/10.17605/OSF.IO/X5RGM  
   Files: `train.json`, `val.json`, `test.json`

2. **Yahoo L6 full dataset (Webscope)**  
   https://webscope.sandbox.yahoo.com/catalog.php?datatype=l&did=11  
   Files: `yahool6.xml` / `FullOct2007.xml` (requires data use agreement)

---

### Data Format

Each entry contains:

```json
{
  "id": "question_id",
  "human_summary": "What are symptoms of diabetes?",
  "question_focus": ["diabetes"],
  "question_type": ["symptoms"]
}
````

---

## Models

### Fine-Tuned Seq2Seq Baselines

The following models are fine-tuned on the CHQ-Summ training split:

* **BART (base)**
* **PEGASUS (large)**
* **ProphetNet (base)**
* **T5-Base**

---

## Large Language Model Evaluation

We evaluate **inference-only** performance of modern open-weight LLMs without fine-tuning:

* **Gemma-7B**
* **DeepSeek-7B**
* **Mistral-7B-v0.3**
* **Llama-3.1-8B**
* **Llama-3.2-3B**
* **Qwen-2-7B**

Zero-shot, two-shot, and five-shot prompting strategies are supported.

---

## Evaluation Metrics

All models are evaluated using the **same metric suite**:

| Metric              | Description                          | Category    |
| ------------------- | ------------------------------------ | ----------- |
| ROUGE-L             | Longest common subsequence overlap   | Lexical     |
| BERTScore           | Contextual semantic similarity       | Semantic    |
| Semantic Coherence  | Readability and logical flow         | Structural  |
| QE Overlap          | Retention of key question evidence   | Information |
| Entity Preservation | Medical entity correctness           | Factual     |
| SummaC              | Entailment-based factual consistency | Inference   |
| Entailment          | Logical validity of summaries        | Inference   |

---

## Installation

### Prerequisites

* Python 3.8+
* CUDA 11.8+
* 16GB RAM (24GB+ recommended)

### Setup

```bash
git clone https://github.com/abasu9/CHQ-Summ-A-Dataset-for-Consumer-Healthcare-Question-Summarization.git
cd CHQ-Summ-A-Dataset-for-Consumer-Healthcare-Question-Summarization

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

---

## Acknowledgments

* Yahoo! Answers for the L6 dataset
* National Library of Medicine (NIH)
* Medical informatics experts who annotated CHQ-Summ
* Hugging Face Transformers

---

## Links

* Paper: [https://arxiv.org/abs/2206.06581](https://arxiv.org/abs/2206.06581)
* Dataset: [https://doi.org/10.17605/OSF.IO/X5RGM](https://doi.org/10.17605/OSF.IO/X5RGM)
* Yahoo L6: [https://webscope.sandbox.yahoo.com/catalog.php?datatype=l&did=11](https://webscope.sandbox.yahoo.com/catalog.php?datatype=l&did=11)

---

**If you find this repository useful, please star it.**

```

---

This is now **GitHub-valid, journal-safe, reviewer-clean**, with **zero content changes**.

If you want, next we can:
- add CLI examples **without touching text**
- validate against *Scientific Data* checklist
- generate a `CITATION.cff` file

Just say.
