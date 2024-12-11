# Retrieval-Augmented Generation (RAG) Evaluation Framework

## Table of Contents
1. [Introduction to the RAG Process](#introduction-to-the-rag-process)
2. [Key Concepts in RAG](#key-concepts-in-rag)
3. [How RAG Works](#how-rag-works)
4. [Why RAG is Important](#why-rag-is-important)
5. [Key Steps in a RAG Pipeline](#key-steps-in-a-rag-pipeline)
6. [Optional Enhancements](#optional-enhancements)
7. [2-Stage Evaluation Process for RAG Pipeline](#2-stage-evaluation-process-for-rag-pipeline)
8. [Evaluation Procedure for RAG Retrieval with Dual Comparison](#evaluation-procedure-for-rag-retrieval-with-dual-comparison)
9. [Key Evaluation Criteria for RAG Retrieval](#key-evaluation-criteria-for-rag-retrieval)
10. [Evaluation of Web-Based Retrieval](#evaluation-of-web-based-retrieval)
11. [RAG Answer Evaluation](#rag-answer-evaluation)
12. [Python-based RAG Evaluation Frameworks](#python-based-rag-evaluation-frameworks)
13. [Overview](#overview)
14. [Modules](#modules)
    - [1. generate_test_data.py](#1-generate_test_datapy)
    - [2. loggers.py](#2-loggerspy)
    - [3. eval_rag.py](#3-eval_ragpy)
15. [Prerequisites](#prerequisites)
    - [1. API Key Configuration](#1-api-key-configuration)
    - [2. Install Requirements](#2-install-requirements)
16. [How to Run](#how-to-run)
17. [Output Files](#output-files)
18. [RAGAS Library](#ragas-library)
19. [Metrics for Evaluating a RAG Pipeline](#metrics-for-evaluating-a-rag-pipeline)
20. [Contribution](#contribution)
21. [License](#license)

---

## Introduction to the RAG Process

RAG (Retrieval-Augmented Generation) is a hybrid approach that combines information retrieval with text generation to provide accurate and contextually relevant answers to user queries. It leverages the power of large language models (LLMs) while grounding their responses in factual, retrieved information from a document or knowledge base.

## Key Concepts in RAG

1. **Retrieval Component**:
   - Focuses on fetching relevant documents, chunks, or data from a knowledge base.
   - Uses methods like vector search, keyword-based search (e.g., BM25), or hybrid approaches to identify content related to the query.

2. **Augmentation**:
   - Combines retrieved documents with the user's query to provide context for the generative model.
   - Prevents the generative model from hallucinating or producing irrelevant responses.

3. **Generation Component**:
   - Uses an LLM (e.g., GPT-4) to generate natural language answers.
   - Relies on the context provided by retrieved documents to enhance accuracy and relevance.

## How RAG Works

The RAG process operates in two main stages:

1. **Retrieve**:
   - Query embedding is generated and matched with document embeddings in a vector database.
   - Retrieved results are ranked based on semantic similarity, contextual relevance, or additional criteria like temporal relevance.

2. **Generate**:
   - The retrieved documents are combined with the query to create a well-structured input (prompt).
   - The generative model produces the final answer, enriched by the factual content from the retrieved documents.

## Why RAG is Important

- **Fact-Based Answers**: Helps reduce hallucination by grounding answers in trusted sources.
- **Scalable Knowledge Integration**: Combines static knowledge bases with real-time data (e.g., web search).
- **Versatile Applications**: Useful in domains like customer support, research, healthcare, and education, where accuracy and context matter.

## Key Steps in a RAG Pipeline

1. **Document Loading**:
   - Use loaders like pypdf, unstructured, or similar tools to parse raw documents into textual content.
   - Supports multiple file types, e.g., PDFs, Word docs, images, etc.

2. **Text Splitting**:
   - Divide content into manageable chunks using tools like RecursiveTextSplitter or advanced semantic chunkers.
   - Splitting methods (character, text, or sentence) ensure chunks are coherent and aligned with query needs.

3. **Embedding Creation**:
   - Convert textual chunks into vector representations using embedding models (e.g., OpenAI embeddings, SentenceTransformers, or other LLM-based embeddings).

4. **Vector Indexing**:
   - Store these embeddings in vector databases like ChromaDB, FAISS, Pinecone, Quadrant, LanceDB, Neo4j, or graph databases.

5. **Query Processing**:
   - When a query is received:
     - Generate the query embedding using the same embedding model as for documents.
     - Perform a vector search to identify the most relevant chunks.

6. **Hybrid Search**:
   - Combine multiple retrieval approaches:
     - Dense Vector Search: Semantic similarity via embeddings.
     - Sparse Search: Keyword-based relevance using BM25, TF-IDF.
     - Optionally, use graph-based search or web search for enhanced coverage.

7. **Re-ranking**:
   - Reorder the retrieved documents using techniques like cross-encoder models or LLM-based scoring for contextual relevance and better-ranked results.

8. **Context Preparation**:
   - Merge retrieved documents and the query into a single structured prompt for the generative model.
   - Perform effective prompt engineering to enhance accuracy and minimize hallucinations.

9. **Answer Generation**:
   - Send the prompt to a generative model (e.g., GPT-4, GPT-4-mini) to produce the final response.

## Optional Enhancements

1. **Web Search Integration**:
   - Supplement local data by fetching external information via web-based searches when required.

2. **Adaptive Querying**:
   - Dynamically adapt between retrieval methods (vector, hybrid, graph, or web search) based on query complexity.

## 2-Stage Evaluation Process for RAG Pipeline

The 2-stage evaluation process for a RAG pipeline focuses on ensuring both the retrieval and generation components perform effectively.

1. **Retrieval Evaluation Stage**:
   - Assess the relevance and quality of the documents retrieved in response to the user query.
   - Metrics: Precision, recall, relevance, cosine similarity, re-ranking performance analysis.

2. **RAG Answer Evaluation**:
   - Examine the accuracy, coherence, and completeness of the answers generated by the model.
   - Metrics: BLEU, ROUGE, factual faithfulness tests.

## Evaluation Procedure for RAG Retrieval with Dual Comparison

### Dual Comparison Framework

This framework compares the retrieved documents with both the user query and the ground truth to assess the performance of the retrieval system.

1. **User Query ↔ Retrieved Document**:
   - **Purpose**: Assess semantic alignment and contextual relevance.
   - **Metrics**: Cosine Similarity, nDCG (Normalized Discounted Cumulative Gain).

2. **Ground Truth ↔ Retrieved Document**:
   - **Purpose**: Measure the factual and contextual accuracy of the retrieved documents.
   - **Metrics**: Cosine Similarity, Precision, Recall, F1-Score, BLEU/ROUGE.

## Key Evaluation Criteria for RAG Retrieval

1. **Relevance**:
   - **Comparison**: Query ↔ Retrieved Document, Ground Truth ↔ Retrieved Document.
   - **Metrics**: Cosine Similarity, nDCG.

2. **Accuracy**:
   - **Comparison**: Retrieved Document ↔ Ground Truth.
   - **Metrics**: Precision, Recall, F1-Score.

3. **Coverage**:
   - **Comparison**: Query ↔ Retrieved Document.
   - **Metrics**: Recall.

4. **Diversity**:
   - **Comparison**: Retrieved Document ↔ Retrieved Document (within the same retrieval batch).
   - **Metrics**: Clustering-Based Diversity Scoring.

5. **Sentiment Alignment**:
   - **Comparison**: Query ↔ Retrieved Document (for sentiment-driven queries).
   - **Metrics**: Sentiment Analysis Scores.

6. **Specificity**:
   - **Comparison**: Retrieved Document ↔ Ground Truth.
   - **Metrics**: Precision.

7. **Temporal Relevance**:
   - **Comparison**: Retrieved Document ↔ Timestamp.
   - **Metrics**: Timestamp Analysis.

8. **Language and Readability**:
   - **Comparison**: Retrieved Document ↔ Query.
   - **Metrics**: Flesch Reading Ease.

9. **Handling Ambiguity**:
   - **Comparison**: Query ↔ Retrieved Documents (for ambiguous queries).
   - **Metrics**: Diversity Scoring.

10. **Novelty**:
    - **Comparison**: Retrieved Document ↔ Retrieved Document (within the same retrieval batch).
    - **Metrics**: Cosine Similarity.

## Evaluation of Web-Based Retrieval

Evaluating the retrieval from web-based search tools involves a slightly different approach because the retrieved data is often dynamic, unstructured, and context-sensitive.

### Key Steps to Evaluate Web-Based Retrieval

1. **Relevance**:
   - **Comparison**: Query ↔ Retrieved Snippets.
   - **Metrics**: Cosine Similarity, Human Judgment.

2. **Coverage**:
   - **Comparison**: Query ↔ Retrieved Snippets.
   - **Metrics**: Recall, Top-k Document Analysis.

3. **Diversity**:
   - **Comparison**: Snippet ↔ Snippet (within the same batch).
   - **Metrics**: Jaccard Similarity, Clustering-Based Diversity Scoring.

4. **Specificity**:
   - **Comparison**: Snippet ↔ Query.
   - **Metrics**: Precision, Human Annotation.

5. **Temporal Relevance**:
   - **Comparison**: Snippet Timestamp ↔ Query Context.
   - **Metrics**: Timestamp Analysis, Temporal Coverage.

6. **Authority and Credibility**:
   - **Comparison**: Retrieved Source ↔ Trusted Sources List.
   - **Metrics**: Domain Credibility Score, Fact-Check Pass.

7. **Readability**:
   - **Comparison**: Snippet ↔ User.
   - **Metrics**: Flesch Reading Ease, Sentence Length Analysis.

8. **Sentiment Alignment (Optional)**:
   - For sentiment-based queries, measure how well the tone of the retrieved web snippets aligns with the sentiment of the query.

## RAG Answer Evaluation

### Evaluation Framework for RAG Answer Validation

In this stage, the query, ground truth answer (reference), and model-generated answer (candidate) are compared to ensure the answer is accurate, relevant, and linguistically coherent.

1. **Correctness and Factual Accuracy**:
   - **Metrics**: Exact Match, F1-Score, ROUGE, BLEU.

2. **Semantic and Contextual Relevance**:
   - **Metrics**: METEOR, Cosine Similarity, nDCG.

3. **Linguistic Fluency and Coherence**:
   - **Metrics**: Perplexity, Flesch Reading Ease, Grammar Checks.

4. **Structural Alignment**:
   - **Metrics**: TER (Translation Edit Rate), Levenshtein Distance.

5. **Coverage**:
   - **Metrics**: Recall, Top-k Coverage Analysis.

6. **Diversity and Specificity**:
   - **Metrics**: Precision, Jaccard Similarity.

7. **Temporal and Contextual Validity**:
   - **Metrics**: Temporal Relevance, Contextual Validity Score.

8. **Authority and Credibility**:
   - **Metrics**: Source Validation, Credibility Score.

## Python-based RAG Evaluation Frameworks

1. **RAGAs Framework for Performance Evaluation of RAG Applications**:
   - Key metrics: Context Relevancy, Context Recall, Faithfulness, Answer Relevancy, Response Precision, Contextual Consistency, Factual Correctness, Context Entities Recall, Semantic Similarity, BLEU Score, ROUGE Score, Exact Match.

2. **RAGChecker: A Fine-grained Framework for Diagnosing RAG**:
   - Key features: Holistic Evaluation, Diagnostic Metrics, Fine-grained Analysis, Benchmark Dataset, Meta-Evaluation.

3. **LLM as Judge for RAG Evaluation**:
   - Key metrics: Accuracy and Relevance, Coherence and Fluency, Contextual Understanding, Diversity and Creativity, Helpfulness and User Satisfaction, Win Rate and Comparison with Other Models.

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
