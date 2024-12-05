import os
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI

# Disable tracing
os.environ["LANGCHAIN_TRACING_V2"] = "false"

# Load environment variables
load_dotenv()

# Setting up LLM
llm = ChatOpenAI(model="gpt-4", temperature=0, max_retries=2)

# Initialize embeddings
print("Initializing OpenAI Embeddings...")
openai_embedder = OpenAIEmbeddings(model="text-embedding-3-large", api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Pinecone
print("Connecting to Pinecone...")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("chatbot")

# Vector store and retriever
print("Setting up vector store...")
vector_store = PineconeVectorStore(index=index, embedding=openai_embedder)

# Function to take a query and get retrieved documents
def get_retrieved_documents(query, k=1):
    print(f"Retrieving documents for query: {query}")
    results = vector_store.similarity_search_with_score(query, k)
    return results[0][0].page_content

# Function to generate the RAG prompt
def generate_rag_prompt():
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Your primary responsibility is to answer questions based on the retrieved document provided. "
                "If the answer cannot be found in the retrieved document, simply respond with 'Answer is not available in the provided context.' "
                "Do not generate any random or assumed information. Ensure that your response is based entirely on the context provided, "
                "and do not try to answer the question if the information is missing. For each response, clearly indicate if the information comes from the retrieved document. "
                "Keep your answers concise, accurate, and contextually relevant."
            ),
            ("human", "{query}"),
            ("assistant", "{context}")
        ]
    )

# Getting answer
def get_answer(query):
    # Retrieve documents
    docs = get_retrieved_documents(query)

    # Generate prompt
    prompt = generate_rag_prompt()

    # Create a chain and invoke
    chain = prompt | llm
    response = chain.invoke({"query": query, "context": docs})
    return docs, response.content

# Load and process data
def process_dataframe(filepath):
    """
    Processes the CSV file to:
    1. Remove the 'Unnamed: 0' and 'synthesizer_name' columns.
    2. Clean the 'reference_contexts' column by splitting and slicing.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Processed dataframe.
    """
    print(f"Loading and processing data from {filepath}...")
    data = pd.read_csv(filepath)

    # Remove the 'Unnamed: 0' and 'synthesizer_name' columns
    columns_to_remove = ['Unnamed: 0', 'synthesizer_name']
    data = data.drop(columns=[col for col in columns_to_remove if col in data.columns], errors='ignore')

    # Process the 'reference_contexts' column
    if 'reference_contexts' in data.columns:
        def clean_reference_context(value):
            try:
                return value.split("\\n\\n")[1][2:-2]
            except (IndexError, TypeError):
                return value
        data['reference_contexts'] = data['reference_contexts'].apply(clean_reference_context)

    return data

# Function to process the DataFrame and get answers
def process_dataframe_with_answers(filepath="data.csv"):
    print("Processing the DataFrame and retrieving answers...")
    data = process_dataframe(filepath)

    retrievals = []
    candidates = []

    # Iterate through the user_input column
    for query in tqdm(data['user_input'], desc="Processing queries"):
        try:
            retrieved_doc, answer = get_answer(query)
        except Exception as e:
            retrieved_doc = "Error retrieving document"
            answer = f"Error: {str(e)}"
            print(f"Error processing query: {query} - {e}")

        retrievals.append(retrieved_doc)
        candidates.append(answer)

    # Add the results to the DataFrame
    data['retrievals'] = retrievals
    data['candidates'] = candidates

    return data

# Driver code
if __name__ == "__main__":
    print("Starting process...")
    output_data = process_dataframe_with_answers("data.csv")
    print("Saving results to output.csv...")
    output_data.to_csv("output.csv", index=False)
    print("Process completed!")
