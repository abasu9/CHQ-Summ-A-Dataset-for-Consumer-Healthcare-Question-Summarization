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

## Repository Structure

