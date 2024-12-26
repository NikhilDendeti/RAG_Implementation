import fitz
from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Function to process the PDF and generate chunks and embeddings
def process_pdf(pdf_file):
    # Open the uploaded PDF
    doc = fitz.open(pdf_file)

    # Extract and split the PDF content into chunks
    doc_content = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        doc_content += page.get_text()

    # Split text into chunks
    chunks = split_into_chunks_by_words(doc_content)

    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Generate embeddings for each chunk
    chunk_embeddings = embeddings.embed_documents(chunks)

    return chunks, chunk_embeddings


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


