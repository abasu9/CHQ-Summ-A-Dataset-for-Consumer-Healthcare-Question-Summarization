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

The **CHQ-Summ** dataset targets summarization of consumer health questions from online forums. Such questions often include peripheral information, complicating downstream understanding. This repository provides a modern, reproducible evaluation framework for automatic question summarization systems.

### Key Features

- **1,507 expert-annotated summaries**  
- **Focus entities** and **33 question types** per item  
- Out-of-the-box support for **BART, PEGASUS, T5, ProphetNet** and custom models  
- Prompting strategies: **Standard, Chain-of-Density, Hierarchical, Element-Aware**  
- Metrics: **ROUGE-L, BERTScore, Semantic Coherence, Entailment (cosine), SummaC-style (cosine), QE Overlap, Entity Preservation**  
- Works with **Transformers ≥4.35** and **PyTorch ≥2.0**

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

