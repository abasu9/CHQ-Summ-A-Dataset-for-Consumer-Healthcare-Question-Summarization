
# CHQ-Summ: A Dataset for Consumer Healthcare Question Summarization

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-yellow.svg)](https://huggingface.co/transformers/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-arXiv-green.svg)](https://arxiv.org/abs/2206.06581)

**Evaluation framework for the CHQ-Summ dataset for consumer healthcare question summarization.**

> 📄 **Original Paper**: [CHQ-Summ: A Dataset for Consumer Healthcare Question Summarization](https://arxiv.org/abs/2206.06581)  
> 👥 **Original Authors**: Shweta Yadav, Deepak Gupta, Dina Demner-Fushman  
> 🏛️ **Institutions**: University of Illinois Chicago & National Library of Medicine, NIH  
> 👨‍💻 **Repository Maintainer**: Abhishek Basu

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Installation](#-installation)
- [Data Preparation](#-data-preparation)
- [Quick Start](#-quick-start)
- [Evaluation Framework](#-evaluation-framework)
- [Results](#-results)
- [Advanced Usage](#-advanced-usage)
- [Dataset Analysis](#-dataset-analysis)
- [Contributing](#-contributing)
- [License](#-license)
- [Citation](#-citation)
- [Acknowledgments](#-acknowledgments)
- [Contact](#-contact)
- [Links](#-links)

---

## 🎯 Overview

The **CHQ-Summ** dataset targets summarization of consumer health questions from online forums. Such questions often include peripheral information, complicating downstream understanding. This repository provides a modern, reproducible evaluation framework for automatic question summarization systems.

### Key Features

- **1,507 expert-annotated summaries**  
- **Focus entities** and **33 question types** per item  
- Out-of-the-box support for **BART, PEGASUS, T5, ProphetNet** and custom models  
- Prompting strategies: **Standard, Chain-of-Density, Hierarchical, Element-Aware**  
- Metrics: **ROUGE-L, BERTScore, Semantic Coherence, Entailment (cosine), SummaC-style (cosine), QE Overlap, Entity Preservation**  
- Works with **Transformers ≥4.35** and **PyTorch ≥2.0**

---

## 📊 Dataset

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

## 🚀 Installation

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

### Requirements (excerpt)

```txt
torch>=2.0.0
transformers>=4.35.0
datasets>=2.14.0
sentencepiece>=0.1.99
sentence-transformers>=2.2.2
bert-score>=0.3.13
rouge-score>=0.1.2
spacy>=3.7.0
lxml>=4.9.0
pandas>=2.0.0
numpy>=1.24.0
tqdm>=4.66.0
pyyaml>=6.0
```

---

## 📁 Data Preparation

### Step 1: Download Data

* CHQ-Summ splits from OSF
* Yahoo L6 XML from Webscope

### Step 2: Organize Files

```
Yahoo-CHQ-Summ/
├── data/
│   ├── yahool6.xml          # Yahoo L6 dataset
│   ├── train.json           # CHQ-Summ train split
│   ├── val.json             # CHQ-Summ validation split
│   └── test.json            # CHQ-Summ test split
```

### Step 3: (Optional) Extract Yahoo Data to parallel files

```bash
python scripts/prepare_data.py \
  --Yahoo_data_path data/yahool6.xml \
  --CHQ_summ_path data/
```

Creates:

```
data/
├── train.source  ─┐
├── train.target   ├─ parallel files for seq2seq
├── val.source     ├─ training (original pipeline)
├── val.target     ├─
├── test.source   ─┘
└── test.target
```

---

## 🎮 Quick Start

### Evaluate Pre-trained Models

```bash
# Evaluate all models (0-shot)
python -m evaluation.evaluate

# Evaluate specific model
python -m evaluation.evaluate --model bart --samples 50

# With a custom config
python -m evaluation.evaluate --config config/eval_config.yaml
```

### Few-Shot Evaluation

```bash
# 0-shot, 2-shot, 5-shot
python -m evaluation.evaluate_fewshot \
  --model bart \
  --model_path facebook/bart-large \
  --shots 0 2 5

# Custom few-shot settings
python -m evaluation.evaluate_fewshot \
  --model pegasus \
  --model_path google/pegasus-large \
  --shots 0 1 3 5 10
```

### Shell Script

```bash
# Quick evaluation of all models
bash scripts/run_evaluation.sh
```

---

## 🔬 Evaluation Framework

### Supported Models

| Model          | Size | HuggingFace ID                       |
| -------------- | ---- | ------------------------------------ |
| **BART**       | 406M | `facebook/bart-large`                |
| **PEGASUS**    | 568M | `google/pegasus-large`               |
| **ProphetNet** | 382M | `microsoft/prophetnet-large-uncased` |
| **T5**         | 770M | `t5-large`                           |

### Prompting Strategies

**1) Standard**

```
Summarize:
Text: [Question + Details + Answer]
Summary:
```

**2) Chain-of-Density (CoD)**

```
Dense summary:
Text: [Question + Details + Answer]
Summary:
```

**3) Hierarchical**

```
Hierarchical summary:
Text: [Question + Details + Answer]
Summary:
```

**4) Element-Aware**

```
WHO/WHAT/WHY:
Text: [Question + Details + Answer]
Summary:
```

### Metrics

| Metric                    | Description                       | Range |
| ------------------------- | --------------------------------- | ----- |
| **ROUGE-L**               | Longest common subsequence        | 0–1   |
| **BERTScore**             | Contextual embedding similarity   | 0–1   |
| **Semantic Coherence**    | Inter-sentence coherence          | 0–1   |
| **Entailment (cosine)**   | Source–summary semantic agreement | 0–1   |
| **SummaC-style (cosine)** | Sentence-level consistency proxy  | 0–1   |
| **QE Overlap**            | Question-entity overlap           | 0–1   |
| **Entity Preservation**   | Named-entity preservation rate    | 0–1   |

---

### Output Files

```
results/
├── bart_eval.csv              # Average metrics per method
├── bart_summaries.csv         # Generated summaries + scores (per sample)
├── pegasus_eval.csv
├── pegasus_summaries.csv
├── t5_eval.csv
├── t5_summaries.csv
├── prophetnet_eval.csv
├── prophetnet_summaries.csv
└── ALL_MODELS_eval.csv        # Combined results
```

**Columns in `{model}_summaries.csv`:**

* `source_text` — original question + answer
* `reference_summary` — human summary
* `generated_summary` — model output
* 7 evaluation metric columns

---

## 🛠️ Advanced Usage

### Custom Model Evaluation

```python
from evaluation.evaluate import ModelEvaluator
from data.data_loader import load_yahoo_dataset

# Load data
train_data, val_data, test_data = load_yahoo_dataset('data/')

# Evaluate your model
evaluator = ModelEvaluator(
    model_path='your-org/your-finetuned-model',
    model_type='bart'  # or 'pegasus', 't5', 'prophetnet'
)

results = evaluator.evaluate(
    test_data=test_data,
    methods=['standard', 'chain_of_density'],
    num_shots=0
)
```

### Adding Custom Metrics

```python
# In evaluation/metrics.py
class EvaluationMetrics:
    def your_custom_metric(self, pred: str, ref: str, src: str) -> float:
        score = calculate_your_metric(pred, ref, src)
        return score
```

---

## 📊 Dataset Analysis

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

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/AmazingFeature`
3. Commit: `git commit -m "Add AmazingFeature"`
4. Push: `git push origin feature/AmazingFeature`
5. Open a Pull Request

**Ways to contribute:** bug reports, new metrics, docs, model support, sharing results.

---

## 📄 License

MIT — see [LICENSE](LICENSE).

---

## 📚 Citation

If you use this dataset or code, please cite:

```bibtex
@article{yadav2022chqsumm,
  title={CHQ-Summ: A Dataset for Consumer Healthcare Question Summarization},
  author={Yadav, Shweta and Gupta, Deepak and Demner-Fushman, Dina},
  journal={arXiv preprint arXiv:2206.06581},
  year={2022}
}
```

**Related Papers:**

```bibtex
@inproceedings{yadav2021reinforcement,
  title={Reinforcement Learning for Abstractive Question Summarization with Question-Aware Semantic Rewards},
  author={Yadav, Shweta and Gupta, Deepak and Abacha, Asma Ben and Demner-Fushman, Dina},
  booktitle={Proceedings of ACL-IJCNLP 2021},
  pages={249--255},
  year={2021}
}

@article{yadav2022question,
  title={Question-Aware Transformer Models for Consumer Health Question Summarization},
  author={Yadav, Shweta and Gupta, Deepak and Abacha, Asma Ben and Demner-Fushman, Dina},
  journal={Journal of Biomedical Informatics},
  pages={104040},
  year={2022}
}
```

---

## 🙏 Acknowledgments

* Yahoo! Answers (L6 dataset) for the L6 dataset
* National Library of Medicine, NIH for supporting this research
* Medical informatics experts who annotated CHQ-Summ
* Hugging Face Transformers

---

## 📧 Contact

* **Repository Maintainer**: Abhishek Basu — [abasu9@uic.edu](mailto:abasu9@uic.edu)
* **Issues**: use [GitHub Issues](../../issues)

---

## 🔗 Links

* 📄 Paper: [https://arxiv.org/abs/2206.06581](https://arxiv.org/abs/2206.06581)
* 💾 Dataset: [https://doi.org/10.17605/OSF.IO/X5RGM](https://doi.org/10.17605/OSF.IO/X5RGM)
* 🌐 Yahoo L6: [https://webscope.sandbox.yahoo.com/catalog.php?datatype=l&did=11](https://webscope.sandbox.yahoo.com/catalog.php?datatype=l&did=11)

---

**⭐️ If you find this useful, please star the repo!**

