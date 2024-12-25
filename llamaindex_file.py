import os
import fitz
from dotenv import load_dotenv
import llama_index
print(llama_index.__file__)


from llama_index.core import VectorStoreIndex, ServiceContext
from llama_index import LLMPredictor, StorageContext, Document
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.llms import HuggingFaceLLM
from llama_index.vector_stores import QdrantVectorStore
from transformers import pipeline

# Load environment variables
load_dotenv()

# Set Hugging Face Access Token
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("Hugging Face token (HF_TOKEN) is missing.")
os.environ["HF_TOKEN"] = HF_TOKEN

# Load Llama model
try:
    llm_pipeline = pipeline(
        "text-generation",
        model="meta-llama/Llama-2-7b-hf",
        tokenizer="meta-llama/Llama-2-7b-hf",
        device=0
    )
except Exception as e:
    raise ValueError(f"Failed to load model: {e}")

llm_predictor = LLMPredictor(llm=HuggingFaceLLM(pipeline=llm_pipeline))

# Initialize embeddings
embedding_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Connect to Qdrant
try:
    qdrant_vector_store = QdrantVectorStore(
        location="https://8db3766a-65c1-409f-aaa9-79d79321e863.europe-west3-0.gcp.cloud.qdrant.io",
        api_key="xyB4yWC3aXxMfv8VET4cLZdpwMziZ0EM2j3BMQ6gUy--t6G7jVmx8w",
        collection_name="google_doc_embeddings",
        distance_metric="COSINE",
        vector_dim=384
    )
except Exception as e:
    raise ValueError(f"Failed to connect to Qdrant: {e}")

service_context = ServiceContext.from_defaults(
    llm_predictor=llm_predictor,
    embed_model=embedding_model
)

# Load and process PDF
def load_pdf_content(pdf_path):
    doc = fitz.open(pdf_path)
    content = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        content += page.get_text()
    return content

pdf_path = "Insurance Industry _ Report.pdf"
pdf_content = load_pdf_content(pdf_path)
if not pdf_content.strip():
    raise ValueError("PDF content is empty.")

chunks = split_into_chunks_by_words(pdf_content, words_per_chunk=300)
documents = [Document(text=chunk) for chunk in chunks]

storage_context = StorageContext.from_defaults(vector_store=qdrant_vector_store)
index = VectorStoreIndex.from_documents(documents, service_context=service_context, storage_context=storage_context)

# Query and HyDE
question = "What are Opportunities and Challenges in the Insurance Industry?"
retrieved_docs, hypothetical_doc = query_with_hyde(index, question, service_context, k=5)

context = "\n".join([doc.node.text for doc in retrieved_docs])
response = service_context.llm_predictor.predict(
    f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
)

print(f"Question: {question}\nAnswer: {response}")
