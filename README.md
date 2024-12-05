
# Retrieval-Augmented Generation (RAG) Evaluation Framework  

## Table of Contents  
1. [Overview](#overview)  
2. [Modules](#modules)  
   - [1. generate_test_data.py](#1-generate-test-datapython)  
   - [2. loggers.py](#2-loggerspython)  
   - [3. eval_rag.py](#3-eval-ragpython)  
3. [Prerequisites](#prerequisites)  
   - [1. API Key Configuration](#1-api-key-configuration)  
   - [2. Install Requirements](#2-install-requirements)  
4. [How to Run](#how-to-run)  
5. [Output Files](#output-files)  
6. [RAGAS Library](#ragas-library)  
7. [Metrics for Evaluating a RAG Pipeline](#metrics-for-evaluating-a-rag-pipeline)  
8. [Contribution](#contribution)  
9. [License](#license)  

---

## Overview  

This framework evaluates the performance of a **Retrieval-Augmented Generation (RAG)** pipeline. It benchmarks both **retrieved documents** and **generated answers** using multiple metrics. The pipeline consists of **three Python modules**, each handling a specific aspect of the workflow.

---

## Modules  

### 1. `generate_test_data.py`  

#### Purpose:  
This module processes a dataset (`data.csv`) prepared from custom PDF files. It contains the following **three columns**:  
   - **user_input**: The query or input from the user.  
   - **reference_contexts**: The reference documents for retrieval evaluation.  
   - **reference**: The ground truth answers for generated response evaluation.  

#### Workflow:  
- The module uses this data as input and integrates it into the pipeline.  
- It employs **Pinecone retriever**, a dense vector semantic search retriever, to fetch documents based on the user's query.  
- Prepares a prompt by combining the user's query with retrieved documents as context and feeds it into **ChatOpenAI (gpt-4-mini)** to generate answers.  

#### Output:  
- An Excel file named `output.xlsx` is generated, containing **five columns**:  
  1. **query**: User query from `data.csv`.  
  2. **reference_retrieval**: The original reference document for retrieval evaluation.  
  3. **reference**: The ground truth answers for generated response evaluation.  
  4. **retrieval**: Retrieved documents fetched using Pinecone.  
  5. **candidate**: The model-generated answers based on the query and retrieved documents.  

---

### 2. `loggers.py`  

#### Purpose:  
Handles all logging operations for the pipeline, including logging informational messages and errors.  

#### Key Features:  
- Tracks data processing steps for debugging and monitoring.  
- Ensures proper logging of errors for troubleshooting during the execution of the framework.  

---

### 3. `eval_rag.py`  

#### Purpose:  
This module evaluates the **retrieval performance** and **answer generation quality** using a wide range of metrics.  

#### Workflow:  
- Loads the `output.xlsx` file generated from the `generate_test_data.py` module, which contains five columns:  
  1. **query**  
  2. **reference_retrieval**  
  3. **reference**  
  4. **retrieval**  
  5. **candidate**  
- Evaluates four types of **pairs**:  
  1. **Query vs. Retrieval**: Compares the user's query to the retrieved document from Pinecone.  
  2. **Query vs. Candidate Answer**: Compares the query with the model-generated response to assess how well it addresses the query.  
  3. **Reference (Ground Truth) vs. Candidate**: Compares the generated response with the reference (ground truth) to check alignment.  
  4. **Retrieval vs. Candidate**: Compares the retrieved documents with the generated answer to ensure grounding.  

---

## Prerequisites  

### 1. API Key Configuration  
Make sure to configure your OpenAI API key and store it in the `.env` file. You will need it to run the GPT model in the `generate_test_data.py` module.  

### 2. Install Requirements  
Before running the framework, install the required dependencies:  
```bash
pip install -r requirements.txt
```

---

## How to Run  

1. **Prepare Dataset**: Ensure that your dataset (e.g., `data.csv`) is ready with queries and corresponding reference documents.  
2. **Run the Pipeline**:  
   To run the framework, execute the following command:
   ```bash
   python generate_test_data.py
   ```
   This will generate the `output.xlsx` file with the results.

3. **Evaluate Performance**:  
   After generating the `output.xlsx` file, run:
   ```bash
   python eval_rag.py
   ```
   This will evaluate the performance of the RAG pipeline and provide retrieval and generation metrics.

---

## Output Files  

- **`output.xlsx`**: Contains results from the RAG pipeline, including user queries, retrieved documents, and model-generated answers.  
- **`eval_results.json`**: Stores the evaluation metrics (e.g., Recall@k, BLEU, ROUGE) for retrieval and generation.  

---

## RAGAS Library  

The **RAGAS (Retrieval-Augmented Generation Assessment Suite)** library is a specialized tool designed to evaluate RAG pipelines comprehensively. It combines retrieval and generation evaluation metrics to provide a holistic assessment of the pipeline's performance. Here's how to integrate **RAGAS** into the outlined RAG evaluation process:

---

### **Updated Outline for Evaluating a RAG Pipeline**

### **1. Dataset Preparation**
- **Purpose**: Create a controlled and comprehensive dataset for evaluation.
  - Collect **queries** relevant to the domain or task.
  - Annotate:
    - **Ground truth retrievals**: Manually selected documents for each query.
    - **Reference answers**: Human-annotated ground truth answers.

---

### **2. Evaluate the Retrieval Component**

#### **2.1. Automated Retrieval Metrics**
- **Metrics to Compute**:
  - **Recall@k**: Measure the presence of relevant documents in the top `k` results.
  - **Precision@k**: Determine the proportion of relevant documents in the top `k` results.
  - **Mean Reciprocal Rank (MRR)**: Evaluate the ranking quality of the first relevant document.
  - **nDCG (Normalized Discounted Cumulative Gain)**: Assess how relevant documents are positioned in the retrieval results.

#### **2.2. Semantic Similarity Metrics**
- Compare query embeddings to retrieved document embeddings using cosine similarity or sentence-transformer scores.
- Use tools like `sentence-transformers` for embedding-based evaluation.

#### **2.3. Human Evaluation of Retrieval**
- **Qualitative Assessment**:
  - **Relevance**: Are retrieved documents topically aligned with the query?
  - **Completeness**: Do they contain all necessary information to answer the query?
  - **Precision**: Avoidance of irrelevant or verbose content.
  - **Diversity**: No duplicates or overly redundant results.
  - **Timeliness**: Especially for web-based search, are documents up-to-date?

---

### **3. Evaluate the Generation Component**

#### **3.1. Automated Generation Metrics**
- **Text Similarity Metrics**:
  - **Lexical**: BLEU, ROUGE, METEOR.
  - **Semantic**: BERTScore, cosine similarity.
- **Faithfulness to Retrievals**:
  - Attribution scoring: Check if the generated content is grounded in retrieved documents.
  - Tools like **FactCC** or **AttributionEval** can help verify factual consistency.

#### **3.2. Query vs. Candidate Answer**
- Evaluate the degree to which the generated answer addresses the query. Use semantic similarity metrics to compare the query intent with the generated answer.

#### **3.3. Reference (Ground Truth) vs. Candidate**
- Compare generated answers with reference answers using metrics like:
  - BLEU, ROUGE, METEOR for lexical match.
  - Semantic metrics like BERTScore for meaning-based comparison.
- Check alignment in terms of:
  - **Relevance**: Does the answer address the query?
  - **Completeness**: Does it include all key points from the reference?

#### **3.4. Human Evaluation of Generated Answers**
- Assess using the following criteria:
  - **Relevance**: Is the answer aligned with the query?
  - **Accuracy**: Are the facts in the answer correct?
  - **Groundedness**: Are claims in the answer traceable to retrieved documents?
  - **Coherence**: Is the answer well-structured and grammatically correct?
  - **Conciseness**: Is it free from unnecessary details?

---

### **4. Integrated Evaluation with RAGAS**

**RAGAS** can simplify and automate several aspects of RAG pipeline evaluation, particularly retrieval and generation quality. It combines metrics and produces a consolidated evaluation report.

#### **4.1. Key Features of RAGAS**
- Combines retrieval and generation evaluation in one framework.
- Provides both **automated metrics** and **qualitative insights**.
- Supports domain-specific customization of evaluation pipelines.

#### **4.2. Steps to Use RAGAS**
1. **Install RAG

AS**:  
   ```bash
   pip install ragas
   ```

2. **Integrate RAGAS with the pipeline**:  
   - Replace manual metric calculations with RAGAS methods.  
   - Use `ragas.eval_retrieval()` and `ragas.eval_generation()` for comprehensive reports.

3. **Run Evaluation**:  
   After running your retrieval and generation models, use RAGAS to consolidate results and generate actionable insights for performance improvement.

---

## Metrics for Evaluating a RAG Pipeline  

### Retrieval Metrics:  
- **Recall@k**: Measures if the relevant documents are in the top `k` results.
- **Precision@k**: Proportion of relevant documents in top `k` results.
- **nDCG**: Normalized discount for rank position of relevant documents.
- **MRR**: Reciprocal rank of the first relevant document.

### Generation Metrics:  
- **BLEU, ROUGE, METEOR**: Lexical comparison between generated and reference answers.
- **BERTScore**: Semantic matching of embeddings for similarity.
- **Faithfulness**: Ensures generated text is grounded in retrieved content.

---
