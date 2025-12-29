# 🧠 Alzheimer's Disease Research Assistant (ALZG)

AI-powered RAG (Retrieval-Augmented Generation) system for Alzheimer's disease research using biomedical literature with advanced evaluation metrics.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.30+-yellow.svg)](https://huggingface.co/transformers)

## 📋 Overview

This project implements a sophisticated RAG pipeline for Alzheimer's disease research that:
- **Retrieves** relevant research articles from a biomedical database
- **Generates** evidence-based answers with source citations
- **Evaluates** system performance using multiple metrics
- **Visualizes** results for analysis

### Key Features
- 🔍 Semantic search with re-ranking (Cross-Encoder)
- 📝 LLM-powered answer generation with source attribution
- 📊 Comprehensive evaluation system (ROUGE, similarity, diversity metrics)
- 📈 Interactive visualizations
- 🏗️ Modular pipeline design
- 🐳 Docker support

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- 8GB+ RAM (16GB recommended)
- Optional: CUDA-capable GPU for faster inference

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Vimp17/alzg-alzheimer-rag.git
cd alzg-alzheimer-rag```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Initialize the database with demo articles:

```bash
python init_database.py
```

4. Run the Streamlit application:

```bash
streamlit run app/streamlit_app_hf.py
```