import os
import fitz
import qdrant_client
from dotenv import load_dotenv
from langchain.agents import initialize_agent, Tool, AgentType
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

# Load environment variables
load_dotenv()

# Set Hugging Face Access Token
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

# Load the Llama 3.2 model using Hugging Face's transformers library
llm_pipeline = pipeline(
    "text-generation",
    model="meta-llama/Llama-3.2-1B",
    tokenizer="meta-llama/Llama-3.2-1B",  # Ensure tokenizer matches the model
    device=0  # Use GPU if available, else switch to -1 for CPU
)

# Wrap the Llama pipeline into LangChain
llm = HuggingFacePipeline(pipeline=llm_pipeline)

# Initialize Embeddings with Hugging Face
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")

# Connect to Qdrant
qdrant_client = QdrantClient(
    url="https://8db3766a-65c1-409f-aaa9-79d79321e863.europe-west3-0.gcp.cloud.qdrant.io",
    api_key="xyB4yWC3aXxMfv8VET4cLZdpwMziZ0EM2j3BMQ6gUy--t6G7jVmx8w"
)

# Ensure the Qdrant collection is set up
collection_name = "google_doc_embeddings"
if not qdrant_client.collection_exists(collection_name):
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        # Adjust vector size if needed
    )

# Load and process the PDF
pdf_path = "Insurance Industry _ Report.pdf"  # Path to the PDF in the current directory
doc = fitz.open(pdf_path)


# Function to split text into chunks

def split_into_chunks_by_words(text, words_per_chunk=500):
    words = text.split()
    chunks = []
    current_chunk = []
    for word in words:
        current_chunk.append(word)
        if len(current_chunk) == words_per_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks


# Extract and split the PDF content
doc_content = ""
for page_num in range(len(doc)):
    page = doc.load_page(page_num)
    doc_content += page.get_text()

chunks = split_into_chunks_by_words(doc_content, words_per_chunk=500)

# Generate embeddings for each chunk
chunk_embeddings = embeddings.embed_documents(chunks)

# Upload the embeddings to Qdrant
points = [
    {"id": idx + 1, "vector": embedding, "payload": {"text": chunk}}
    for idx, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings))
]
qdrant_client.upsert(collection_name=collection_name, points=points)


# Function to generate a hypothetical document using HyDE
def generate_hypothetical_doc(question, llm):
    hypothetical_doc = llm.invoke(
        f"Question: {question}\n\nWrite a hypothetical context or document:")
    return hypothetical_doc


# Function for sparse retrieval (simple keyword-based search)
def keyword_search(query, chunks):
    return [chunk for chunk in chunks if query.lower() in chunk.lower()]


# Perform Hybrid Retrieval with HyDE
def hybrid_retrieval_with_hyde(query, chunks, embeddings, llm, k=5):
    # Generate a hypothetical document for the query
    hypothetical_doc = generate_hypothetical_doc(query, llm)

    # Embed the hypothetical document
    hypothetical_embedding = embeddings.embed_documents([hypothetical_doc])[0]

    # Embed the query
    query_embedding = embeddings.embed_documents([query])[0]

    # Perform dense retrieval using both embeddings (query + hypothetical)
    dense_results_query = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=k
    )
    dense_results_hypothetical = qdrant_client.search(
        collection_name=collection_name,
        query_vector=hypothetical_embedding,
        limit=k
    )

    # Perform sparse retrieval (keyword-based)
    sparse_results = keyword_search(query, chunks)

    # Combine results (dense from query + hypothetical + sparse)
    combined_results = dense_results_query + dense_results_hypothetical + sparse_results
    return combined_results[:k]  # Limit to top k results


# Reranking function based on cosine similarity
def rerank_results(query_embedding, results, embeddings):
    # Extract the texts from the retrieved results
    result_texts = [result.payload["text"] for result in results]

    # Generate embeddings for the retrieved documents
    result_embeddings = embeddings.embed_documents(result_texts)

    # Calculate cosine similarity
    similarities = cosine_similarity([query_embedding], result_embeddings)

    # Rank the results based on cosine similarity
    ranked_results = [result for _, result in
                      sorted(zip(similarities[0], results), reverse=True)]
    return ranked_results


# Define the tools for the agent
tools = [
    Tool(
        name="Retrieve Document",
        func=hybrid_retrieval_with_hyde,
        description="Retrieve relevant documents using hybrid dense and sparse retrieval with HyDE."
    ),
    Tool(
        name="Generate Answer",
        func=llm.invoke,
        description="Generate an answer to the question based on retrieved documents."
    )
]

# Initialize the agent
agent = initialize_agent(
    tools,
    llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Perform Retrieval-Augmented Generation (RAG)
question = "What are Opportunities and Challenges in the Insurance Industry?"

# Step 1: Perform Hybrid Retrieval with HyDE
retrieved_docs = hybrid_retrieval_with_hyde(question, chunks, embeddings, llm,
                                            k=5)

# Step 2: Rerank results based on relevance
query_embedding = embeddings.embed_documents([question])[0]
reranked_docs = rerank_results(query_embedding, retrieved_docs, embeddings)

# Step 3: Generate Answer based on reranked documents
context = " ".join([doc.payload["text"] for doc in reranked_docs])
response = llm.invoke(f"Context: {context}\n\nQuestion: {question}\n\nAnswer:")

# Step 4: Structured Output Extraction (formatted)
structured_output = {
    "Question": question,
    "Answer": response,
    "Relevant Documents": [doc.payload["text"] for doc in reranked_docs],
    "Hypothetical Document": generate_hypothetical_doc(question, llm),
}

# Print the structured output in a formatted way
formatted_output = f"""
### Question:
**{structured_output['Question']}**

---

### Answer:
{structured_output['Answer']}

---

### Relevant Documents:
"""
# Loop through relevant documents and add them to the formatted output
for i, doc in enumerate(structured_output['Relevant Documents'], start=1):
    formatted_output += f"{i}. {doc}\n"

formatted_output += f"""

---

### Hypothetical Document:
{structured_output['Hypothetical Document']}
"""

# Print the formatted output
print(formatted_output)
