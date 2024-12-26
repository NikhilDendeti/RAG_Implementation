import streamlit as st
from utils.pdf_utils import process_pdf, generate_answer, retrieve_documents
from langchain.agents import initialize_agent
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from transformers import pipeline

# Streamlit app title
st.title("Insurance Industry Chatbot")

# Upload PDF
pdf_file = st.file_uploader("Upload PDF", type="pdf")

if pdf_file:
    # Process the PDF when uploaded
    st.write("Processing the PDF...")

    # Extract text and process it (you will need to create a PDF processing function)
    chunks, embeddings = process_pdf(pdf_file)

    # Initialize the model and agent
    llm_pipeline = pipeline(
        "text-generation",
        model="meta-llama/Llama-3.2-1B",
        tokenizer="meta-llama/Llama-3.2-1B",
        device=0  # Use GPU if available
    )
    llm = HuggingFacePipeline(pipeline=llm_pipeline)
    agent = initialize_agent(
        tools=[{
            'name': "Retrieve Document",
            'func': retrieve_documents,
            'description': "Retrieve relevant documents"
        }],
        llm=llm,
        agent_type="ZERO_SHOT_REACT_DESCRIPTION"
    )

    # User input
    question = st.text_input("Ask a question:")

    if question:
        # Generate the answer using the agent
        answer = generate_answer(question, agent, chunks, embeddings)
        st.write("Answer:", answer)


from langchain.agents import Tool

# Function to retrieve relevant documents
def retrieve_documents(query, chunks, embeddings, llm, k=5):
    # Similar retrieval logic to your original code
    results = hybrid_retrieval_with_hyde(query, chunks, embeddings, llm, k)
    return results

# Function to generate an answer based on retrieved documents
def generate_answer(query, agent, chunks, embeddings):
    # Use the agent to get an answer
    context = " ".join([doc.payload["text"] for doc in retrieve_documents(query, chunks, embeddings, agent.llm)])
    answer = agent.invoke(f"Context: {context}\n\nQuestion: {query}\n\nAnswer:")
    return answer
