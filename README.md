# CAST: Conciseness-Attuned Summarization Tactic via Multi-Agent and Self-Refinement Framework

Large Language Models (LLMs) have been widely adopted for code summarization, yet they often generate overly verbose function-level summaries. Such verbosity can obscure the essential purpose of the function and burden developers with unnecessary details. 

This repository contains the official code and dataset for **CAST**, a novel multi-agent self-refinement framework designed to generate concise summaries while preserving the original semantics of the source code. CAST decomposes the summarization process into an initial generation stage and an iterative validation-reflection stage.

**Repository Link**: [https://github.com/zhanzhan5125/CAST](https://github.com/zhanzhan5125/CAST)

---

### 1. CodeSearchNet-Lite
This directory contains our curated, high-quality subset of the CodeSearchNet benchmark. To ensure data quality for empirical analysis, we applied heuristic cleaning steps to the original dataset.
* Contains the refined data samples and feature files.

### 2. Empirical Study
This folder houses the scripts and data for our empirical investigation into the characteristics of concise code summaries, directly addressing our two main Research Questions (RQ1 and RQ2).
* **RQ1**: What are the characteristics of concise general summaries?
* **RQ2**: What components are essential for a concise general summary?

**Key Files:**
* `AST.py`: Parses source code into Abstract Syntax Trees to extract a sequence of statements.
* `Attention.py`: Analyzes multi-head cross-attention weights between code structures and summary words (utilizing model `codebert`).
* `RQ1_cluster.py`: Clusters samples using the extracted features to locate the most concise general summaries.
* `RQ2_structure.py`: Analysis the main focus of summaries with parsed code structures.
* `cluster_3_GOLDEN_data.csv` & `golden_cluster_3_final_layer_stats_extended.csv`: Data and statistics for the identified concise cluster (Cluster 3).

### 3. Approach
Calling LLMs to generate comments (Configure parameters and components via command-line arguments):
```bash
python run.py
```

**Key Files:**
* `run.py`: The main executable script that controls the pipeline with parameters.
* `agent.py`: Defines the multi-agent collaboration.
* `model.py`: Manages the API interactions and configurations for the underlying Large Language Models.
* `rag.py`: Implements the Retrieval-Augmented Generation mechanism with the vector database.
* `tool.py`: Implements the auxiliary tools used by agents.
* `validation.py`: Executes the Reflexion-based validation.

### 4. Evaluation
Contains the evaluation pipeline and analysis tools used to benchmark CAST against baseline approaches.
* `evaluation.py`: The primary script for computing performance across our selected metrics (e.g., BLEU, METEOR, ROUGE-L, BERTScore).
* `g-eval_prompt`: Prompt templates specifically designed for the LLM-based G-Eval reasoning metric.
* `plot_draw.py`: Generates high-quality, academic-style visualizations and distribution graphs.
