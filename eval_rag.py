import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import evaluate
import bert_score
from logger import setup_logger

# Initialize logger
logger = setup_logger('metrics_logger', 'metrics_log.log')

# Function to get embeddings from the bge-m3 model
def get_embeddings(sentence, model_name='BAAI/bge-m3'):
    logger.info(f"Getting embeddings for the sentence: {sentence}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Tokenize the input sentence
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True)

    # Get the model output
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the embeddings (mean pooling of the last hidden state)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    logger.debug(f"Embeddings generated: {embeddings[:5]}...")  # Printing the first few values of the embeddings
    return embeddings

# Function to calculate Cosine Similarity
def cosine_similarity_score(embedding1, embedding2):
    logger.info(f"Calculating Cosine Similarity between embeddings")
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    cosine_sim = dot_product / (norm1 * norm2)
    logger.debug(f"Cosine Similarity: {cosine_sim}")
    return cosine_sim

# Function to calculate Jaccard Similarity
def jaccard_similarity(sentence1, sentence2):
    logger.info(f"Calculating Jaccard Similarity between '{sentence1}' and '{sentence2}'")
    set1 = set(sentence1.split())
    set2 = set(sentence2.split())
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    jaccard_sim = intersection / union if union != 0 else 0
    logger.debug(f"Jaccard Similarity: {jaccard_sim}")
    return jaccard_sim

# Function to calculate Coverage Score
def coverage_score(reference_sentence, candidate_sentence):
    logger.info(f"Calculating Coverage Score between '{reference_sentence}' and '{candidate_sentence}'")
    reference_set = set(reference_sentence.split())
    candidate_set = set(candidate_sentence.split())
    intersection = len(reference_set.intersection(candidate_set))
    coverage = intersection / len(reference_set) if len(reference_set) != 0 else 0
    logger.debug(f"Coverage Score: {coverage}")
    return coverage

# Function to calculate Relevance Score
def relevance_score(reference_sentence, candidate_sentence):
    logger.info(f"Calculating Relevance Score between '{reference_sentence}' and '{candidate_sentence}'")
    reference_set = set(reference_sentence.split())
    candidate_set = set(candidate_sentence.split())
    relevance = len(reference_set.intersection(candidate_set)) / len(candidate_set) if len(candidate_set) != 0 else 0
    logger.debug(f"Relevance Score: {relevance}")
    return relevance

# Unified function to calculate all metrics
def calculate_metrics(reference, candidate, model_name='BAAI/bge-m3', lm_model_name='gpt2'):
    logger.info(f"Calculating all metrics for reference: '{reference}' and candidate: '{candidate}'")
    # Get embeddings
    embedding_ref = get_embeddings(reference, model_name)
    embedding_cand = get_embeddings(candidate, model_name)
    
    # Initialize the evaluate library metrics
    rouge = evaluate.load('rouge')
    meteor = evaluate.load('meteor')
    ter = evaluate.load('ter')
    perplexity = evaluate.load('perplexity')
    bleu = evaluate.load('bleu')  # For BLEU Score

    # Calculate Cosine Similarity
    cosine_sim = cosine_similarity_score(embedding_ref, embedding_cand)

    # Calculate Jaccard Similarity
    jaccard_sim = jaccard_similarity(reference, candidate)

    # Calculate Coverage Score
    coverage = coverage_score(reference, candidate)

    # Calculate Relevance Score
    relevance = relevance_score(reference, candidate)

    # Calculate ROUGE score
    rouge_score = rouge.compute(predictions=[candidate], references=[reference])
    logger.debug(f"ROUGE Score: {rouge_score}")

    # Calculate METEOR score
    meteor_score = meteor.compute(predictions=[candidate], references=[reference])
    logger.debug(f"METEOR Score: {meteor_score}")

    # Calculate TER score
    ter_score = ter.compute(predictions=[candidate], references=[reference])
    logger.debug(f"TER Score: {ter_score}")

    # Load the language model for perplexity calculation
    lm_tokenizer = AutoTokenizer.from_pretrained(lm_model_name)
    lm_model = AutoModelForCausalLM.from_pretrained(lm_model_name)

    # Calculate Perplexity using the model and tokenizer
    perplexity_score = perplexity.compute(predictions=[candidate], model_id=lm_model_name)
    logger.debug(f"Perplexity Score: {perplexity_score}")

    # Calculate BLEU Score using the evaluate library
    bleu_score = bleu.compute(predictions=[candidate], references=[[reference]])
    logger.debug(f"BLEU Score: {bleu_score}")

    # Calculate BERTScore
    bert_score_results = bert_score.score([candidate], [reference], lang="en")
    bert_f1 = bert_score_results[2].item()  # F1 score of BERTScore
    logger.debug(f"BERTScore F1: {bert_f1}")

    # Return all the calculated metrics as a dictionary
    return {
        'Cosine Similarity': cosine_sim,
        'Jaccard Similarity': jaccard_sim,
        'Coverage Score': coverage,
        'Relevance Score': relevance,
        'ROUGE Score': rouge_score,
        'METEOR Score': meteor_score,
        'TER Score': ter_score,
        'Perplexity Score': perplexity_score,
        'BLEU Score': bleu_score,
        'BERTScore': bert_f1
    }

# Function to calculate metrics for each pair and return DataFrame
def calculate_for_pair(df, pair1, pair2):
    logger.info(f"Calculating metrics for pairs: '{pair1}' and '{pair2}'")
    metrics_list = []
    for idx, row in df.iterrows():
        reference = row[pair1]
        candidate = row[pair2]
        metrics = calculate_metrics(reference, candidate)
        metrics_list.append(metrics)
    
    return pd.DataFrame(metrics_list)

# Load the Excel file
file_path = 'output.xlsx'
logger.info(f"Loading Excel file from {file_path}")
df = pd.read_excel(file_path, sheet_name='Sheet1')

# Calculate metrics for each pair
df_query_retrieval = calculate_for_pair(df, 'query', 'retrieval')
df_reference_retrieval = calculate_for_pair(df, 'reference_retrieval', 'retrieval')
df_query_candidate = calculate_for_pair(df, 'query', 'candidate')
df_reference_candidate = calculate_for_pair(df, 'reference', 'candidate')

# Save all DataFrames in separate sheets with shortened names
output_file = 'output_metrics_parallel.xlsx'
with pd.ExcelWriter(output_file) as writer:
    df_query_retrieval.to_excel(writer, sheet_name='Query_vs_Retrieval', index=False)
    df_reference_retrieval.to_excel(writer, sheet_name='Ref_Retrieval_vs_Retrieval', index=False)
    df_query_candidate.to_excel(writer, sheet_name='Query_vs_Candidate', index=False)
    df_reference_candidate.to_excel(writer, sheet_name='Ref_vs_Candidate', index=False)

logger.info("Metrics saved successfully to 'output_metrics_parallel.xlsx'")


logger.info("Metrics saved successfully to 'output_metrics.xlsx'")
