# CHQ-Summ: A Dataset for Consumer Healthcare Question Summarization

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-yellow.svg)](https://huggingface.co/transformers/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-arXiv-green.svg)](https://arxiv.org/abs/2206.06581)

**Evaluation framework for the CHQ-Summ dataset for consumer healthcare question summarization.**

>  **Original Paper**: [CHQ-Summ: A Dataset for Consumer Healthcare Question Summarization](https://arxiv.org/abs/2206.06581)  
>  **Original Authors**: Abhishek Basu, Shweta Yadav, Deepak Gupta, Dina Demner-Fushman  
>  **Institutions**: University of Illinois Chicago & National Library of Medicine, NIH  


---

##  Overview
**CHQ-Summ** is a benchmark dataset and evaluation framework for **Consumer Healthcare Question Summarization**.  
Consumers often post long, descriptive medical queries online, making them difficult for automated systems to interpret.  
CHQ-Summ provides **domain-expert annotated question–summary pairs** to train and evaluate models that can distill key medical information from verbose consumer queries.

This repository reproduces and extends the original work:

> **CHQ-Summ: A Dataset for Consumer Healthcare Question Summarization**  
> Yadav et al., *Scientific Data* (under review, 2025)

This implementation includes:
- Fine-tuning of **classical seq2seq models** (BART, PEGASUS, ProphetNet, T5) on CHQ-Summ.  
- Zero-, two-, and five-shot **inference evaluation** on **modern open-weight LLMs** (Gemma-7B, DeepSeek-7B, Mistral-7B-v0.3, Llama-3.1-8B, Llama-3.2-3B, Qwen-2-7B).  
- Comprehensive multi-metric evaluation covering **lexical**, **semantic**, and **factual** dimensions.

---

---
##  Dataset

### CHQ-Summ Statistics

```
**Dataset structure**
data/dataset/CHQ-Summ/
├── train.source
├── train.target
├── val.source
├── val.target
├── test.source
└── test.target

Dataset Split:
├── Train:        1,000 samples
├── Validation:     400 samples
└── Test:           400 samples  (*107 in some releases)

Content Statistics:
├── Avg Question Length:  ~177 words (~10 sentences)
├── Avg Summary Length:   ~13 words (1–2 sentences)
├── Question Focus:       1,788 distinct entities
├── Question Types:       33 types (Information, Treatment, Cause, Symptoms, …)
└── Source:               Yahoo! Answers — Healthcare category

````

> *Some releases include 107 test samples instead of 400.*

### Dataset Access

1. **CHQ-Summ annotations** (OSF): <https://doi.org/10.17605/OSF.IO/X5RGM>  
   Files: `train.json`, `val.json`, `test.json`
2. **Yahoo L6 full dataset** (Webscope): <https://webscope.sandbox.yahoo.com/catalog.php?datatype=l&did=11>  
   Files: `yahool6.xml` or `FullOct2007.xml` (requires agreement)

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

##  Models and Training
### Fine-Tuned Seq2Seq Baselines
We fine-tuned the following models on CHQ-Summ:
- **BART (base)**
- **PEGASUS (large)**
- **ProphetNet (base)**
- **T5-Base**

**Hyperparameters**
| Parameter | Value |
|:-----------|:------|
| Source length | 300 |
| Target length | 50 |
| Batch size | 8 |
| Optimizer | AdamW |
| Learning rate | 3e-5 (BART/PEGASUS/ProphetNet), 3e-3 (T5) |
| Scheduler | Linear warm-up |
| Beam size | 1–9 (validated on val set) |

---

##  Evaluation Metrics

Each model is evaluated using **7 complementary metrics** capturing different aspects of summarization quality:

| Metric | Description | Type |
|:--|:--|:--|
| **ROUGE-L** | Longest common subsequence overlap between summary and reference | Lexical |
| **BERTScore** | Semantic similarity using contextual embeddings | Semantic |
| **Semantic Coherence** | Measures logical flow and readability | Structural |
| **Q.E. Overlap (Question–Evidence Overlap)** | Retention of key information from the input question | Information |
| **Entity Preservation** | Checks correctness of medical entities | Factual |
| **SummaC** | Factual consistency via entailment models | Inference |
| **Entailment** | Logical validity and factual support | Inference |

---

---
##  Dataset

### CHQ-Summ Statistics

```

Dataset Split:
├── Train:        1,000 samples
├── Validation:     400 samples
└── Test:           400 samples  (*107 in some releases)

Content Statistics:
├── Avg Question Length:  ~177 words (~10 sentences)
├── Avg Summary Length:   ~13 words (1–2 sentences)
├── Question Focus:       1,788 distinct entities
├── Question Types:       33 types (Information, Treatment, Cause, Symptoms, …)
└── Source:               Yahoo! Answers — Healthcare category

````

> *Some releases include 107 test samples instead of 400.*

### Dataset Access

1. **CHQ-Summ annotations** (OSF): <https://doi.org/10.17605/OSF.IO/X5RGM>  
   Files: `train.json`, `val.json`, `test.json`
2. **Yahoo L6 full dataset** (Webscope): <https://webscope.sandbox.yahoo.com/catalog.php?datatype=l&did=11>  
   Files: `yahool6.xml` or `FullOct2007.xml` (requires agreement)

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

##  Installation

### Prerequisites

* Python 3.8+
* CUDA 11.8+ (GPU)
* 16GB+ RAM (24GB+ recommended for training)

### Setup

```bash
# Clone the repository
git clone https://github.com/shwetanlp/Yahoo-CHQ-Summ.git
cd Yahoo-CHQ-Summ

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model for NER
python -m spacy download en_core_web_sm
```

##  Data Preparation

### Step 1: Download Data

1. **CHQ-Summ annotations** (OSF): <https://doi.org/10.17605/OSF.IO/X5RGM>  
   Files: `train.json`, `val.json`, `test.json`
2. **Yahoo L6 full dataset** (Webscope): <https://webscope.sandbox.yahoo.com/catalog.php?datatype=l&did=11>  
   Files: `yahool6.xml` or `FullOct2007.xml` (requires agreement)

### Step 2: Organize Files

```
Yahoo-CHQ-Summ/
├── data/
│   ├── yahool6.xml          # Yahoo L6 dataset
│   ├── train.json           # CHQ-Summ train split
│   ├── val.json             # CHQ-Summ validation split
│   └── test.json            # CHQ-Summ test split
```

---

##  Dataset Analysis

### Question Focus (examples)

* **C23** — Pathological conditions, signs & symptoms (most frequent)
* **G11** — Musculoskeletal & neural phenomena
* **F03** — Mental disorders
* **F02** — Psychological phenomena
* **C10** — Nervous system diseases

### Question Type (top)

1. **Information** — general information requests (~400)
2. **Treatment** — options & procedures (~350)
3. **Cause** — causes of conditions (~300)
4. **Symptoms** — symptom identification (~280)
5. **Diagnosis** — diagnostic procedures (~150)

---

##  Acknowledgments

* Yahoo! Answers (L6 dataset) for the L6 dataset
* National Library of Medicine, NIH for supporting this research
* Medical informatics experts who annotated CHQ-Summ
* Hugging Face Transformers

---

##  Links

*  Paper: [https://arxiv.org/abs/2206.06581](https://arxiv.org/abs/2206.06581)
*  Dataset: [https://doi.org/10.17605/OSF.IO/X5RGM](https://doi.org/10.17605/OSF.IO/X5RGM)
*  Yahoo L6: [https://webscope.sandbox.yahoo.com/catalog.php?datatype=l&did=11](https://webscope.sandbox.yahoo.com/catalog.php?datatype=l&did=11)

---

**If you find this useful, please star the repo!**

