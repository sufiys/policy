import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader

# Load OpenAI API Key from Streamlit Secrets
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

st.set_page_config(page_title="Policy Q&A AI", layout="wide")
st.title("📜 AI Policy Q&A Assistant 🤖")

# Upload Policy Document
uploaded_file = st.file_uploader("📂 Upload Policy Document (TXT only)", type=["txt"])

if uploaded_file:
    file_path = "uploaded_policy.txt"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success("✅ Policy uploaded successfully! Processing...")

    # Function to process the policy and store it
    def process_policy(file_path):
        loader = TextLoader(file_path)
        documents = loader.load()

        # Split text into chunks for retrieval
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = text_splitter.split_documents(documents)

        # Create embeddings and vector store
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vectorstore = FAISS.from_documents(docs, embeddings)

        # Save the vector database
        vectorstore.save_local("policy_vector_db")

    process_policy(file_path)
    st.success("✅ Policy document processed successfully!")

# Question input
query = st.text_input("🔍 Ask a question related to the policy")

if st.button("Get Answer"):
    if query:
        # Function to fetch answer from policy
        def ask_question(query):
            embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
            vectorstore = FAISS.load_local("policy_vector_db", embeddings)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=OPENAI_API_KEY), retriever=retriever)
            return qa.run(query)

        response = ask_question(query)
        st.markdown("### 📢 Answer:")
        st.write(response)
    else:
        st.warning("⚠️ Please enter a question.")
