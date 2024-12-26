import os
import fitz
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

# Load environment variables
load_dotenv()

# Initialize global variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or "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"
GROQ_MODEL = "llama-3.1-70b-versatile"
client = Groq(api_key=GROQ_API_KEY)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
qdrant_client = QdrantClient(
    url="https://8db3766a-65c1-409f-aaa9-79d79321e863.europe-west3-0.gcp.cloud.qdrant.io",
    api_key="xyB4yWC3aXxMfv8VET4cLZdpwMziZ0EM2j3BMQ6gUy--t6G7jVmx8w"
)
collection_name = "google_doc_embeddings"

# Ensure the Qdrant collection exists
def setup_qdrant_collection():
    if not qdrant_client.collection_exists(collection_name):
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )

# Load and process a PDF
def load_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    content = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        content += page.get_text()
    return content

# Split text into chunks
def split_into_chunks(text, words_per_chunk=500):
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

# Generate and upload embeddings to Qdrant
def upload_embeddings(chunks):
    chunk_embeddings = embeddings.embed_documents(chunks)
    points = [
        {"id": idx + 1, "vector": embedding, "payload": {"text": chunk}}
        for idx, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings))
    ]
    qdrant_client.upsert(collection_name=collection_name, points=points)

# Call the Groq API for generating responses
def get_groq_response(prompt):
    refined_prompt = f"You are a domain-specific expert chatbot. Answer only if the question pertains to the provided context; otherwise, politely decline. {prompt}"
    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": refined_prompt}],
        temperature=0.7,
        max_tokens=512,
        top_p=0.9,
        stream=False,
        stop=None,
    )

    # Handle the response structure
    try:
        response = completion.choices[0].message.content
    except AttributeError:
        response = "Error: Unexpected response format. Please check the completion object."

    return response


# Generate a hypothetical document using Groq
def generate_hypothetical_doc(question):
    prompt = f"Write a comprehensive, hypothetical context based on the question: {question}"
    return get_groq_response(prompt)

# Perform sparse retrieval (keyword-based search)
def keyword_search(query, chunks):
    return [chunk for chunk in chunks if query.lower() in chunk.lower()]

# Perform Hybrid Retrieval with HyDE
def hybrid_retrieval_with_hyde(query, chunks, k=5):
    hypothetical_doc = generate_hypothetical_doc(query)
    hypothetical_embedding = embeddings.embed_documents([hypothetical_doc])[0]
    query_embedding = embeddings.embed_documents([query])[0]

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
    sparse_results = keyword_search(query, chunks)

    combined_results = dense_results_query + dense_results_hypothetical + sparse_results
    return combined_results[:k]

# Rerank results based on cosine similarity
def rerank_results(query_embedding, results):
    result_texts = [result.payload["text"] for result in results]
    result_embeddings = embeddings.embed_documents(result_texts)
    similarities = cosine_similarity([query_embedding], result_embeddings)
    ranked_results = [result for _, result in sorted(zip(similarities[0], results), reverse=True)]
    return ranked_results

# Generate the final answer using retrieved and reranked documents
def generate_answer(question, reranked_docs):
    context = " ".join([doc.payload["text"] for doc in reranked_docs])
    prompt = (
        f"Context: {context}\n\n"
        f"Question: {question}\n\n"
        "Answer as a helpful, knowledgeable assistant. If the question is irrelevant, respond with: "
        "'This question does not relate to the provided content.'"
    )
    return get_groq_response(prompt)

# Main function to execute the RAG pipeline
def execute_rag_pipeline(pdf_path, question, k=5):
    setup_qdrant_collection()
    doc_content = load_pdf(pdf_path)
    chunks = split_into_chunks(doc_content, words_per_chunk=500)
    upload_embeddings(chunks)

    retrieved_docs = hybrid_retrieval_with_hyde(question, chunks, k=k)
    query_embedding = embeddings.embed_documents([question])[0]
    reranked_docs = rerank_results(query_embedding, retrieved_docs)

    answer = generate_answer(question, reranked_docs)

    structured_output = {
        "Question": question,
        "Answer": answer,
        "Relevant Documents": [doc.payload["text"] for doc in reranked_docs],
        "Hypothetical Document": generate_hypothetical_doc(question),
    }

    return structured_output

# Example usage
if __name__ == "__main__":
    pdf_path = "Insurance Industry _ Report.pdf"
    question = "Who is Nikhil Dendeti?"
    result = execute_rag_pipeline(pdf_path, question)
    print(result["Answer"])
