import os
import fitz
import streamlit as st
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

from main import generate_answer, rerank_results

# Load environment variables
load_dotenv()

# Global Variables
GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY") or "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"
GROQ_MODEL = "llama-3.1-70b-versatile"
client = Groq(api_key=GROQ_API_KEY)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")
qdrant_client = QdrantClient(
    url="https://8db3766a-65c1-409f-aaa9-79d79321e863.europe-west3-0.gcp.cloud.qdrant.io",
    api_key="xyB4yWC3aXxMfv8VET4cLZdpwMziZ0EM2j3BMQ6gUy--t6G7jVmx8w"
)
collection_name = "google_doc_embeddings"


# Qdrant Setup
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


# Upload embeddings
def upload_embeddings(chunks):
    chunk_embeddings = embeddings.embed_documents(chunks)
    points = [
        {"id": idx + 1, "vector": embedding, "payload": {"text": chunk}}
        for idx, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings))
    ]
    qdrant_client.upsert(collection_name=collection_name, points=points)


# Generate hypothetical document
def generate_hypothetical_doc(question):
    prompt = f"Write a comprehensive, hypothetical context based on the question: {question}"
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
    return completion.choices[0].message.content


# Hybrid Retrieval with HyDE
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
    return dense_results_query + dense_results_hypothetical


# Main Streamlit Interface
def main():
    st.title("Context-Aware Chatbot with Retrieval-Augmented Generation (RAG)")
    st.write("Upload a PDF and ask a question to get a relevant answer.")

    # Upload PDF
    uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])
    question = st.text_input("Enter your question:")
    k = st.slider("Number of relevant documents to retrieve:", 1, 10, 5)

    if uploaded_file and question:
        with st.spinner("Processing your PDF..."):
            setup_qdrant_collection()
            pdf_content = load_pdf(uploaded_file)
            chunks = split_into_chunks(pdf_content)
            upload_embeddings(chunks)

        with st.spinner("Retrieving documents..."):
            retrieved_docs = hybrid_retrieval_with_hyde(question, chunks, k=k)
            query_embedding = embeddings.embed_documents([question])[0]
            reranked_docs = rerank_results(query_embedding, retrieved_docs)

        with st.spinner("Generating answer..."):
            answer = generate_answer(question, reranked_docs)

        # Display results
        st.subheader("Answer:")
        st.write(answer)

        st.subheader("Relevant Documents:")
        for doc in reranked_docs:
            st.write(doc.payload["text"])


if __name__ == "__main__":
    main()
