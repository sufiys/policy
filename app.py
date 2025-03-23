import os
import time
import streamlit as st
import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI
from langchain.document_loaders import TextLoader

# Load OpenAI API Key securely from Streamlit Secrets
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
VECTORSTORE_DB_PATH = st.secrets.get("VECTORSTORE_DB_PATH", "policy_vector_db")

st.set_page_config(page_title="Policy Q&A AI", layout="wide")
st.title("📜 AI Policy Q&A Assistant 🤖")

# 📂 Upload Policy Document
uploaded_file = st.file_uploader("📂 Upload Policy Document (TXT only)", type=["txt"])

if uploaded_file:
    file_path = "uploaded_policy.txt"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success("✅ Policy uploaded successfully! Processing...")

    def process_policy(file_path):
        loader = TextLoader(file_path)
        documents = loader.load()

        # 🔹 Optimize text chunking to reduce API calls
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)

        # 🔹 Use OpenAI Embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-ada-002")
        vectorstore = FAISS.from_documents(docs, embeddings)

        # 🔹 Save vector database locally
        vectorstore.save_local(VECTORSTORE_DB_PATH)

    process_policy(file_path)
    st.success("✅ Policy document processed successfully!")

# 🔍 User Question Input
query = st.text_input("🔍 Ask a question related to the policy")

if st.button("Get Answer"):
    if query:
        def ask_question(query):
            embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-ada-002")
            vectorstore = FAISS.load_local(VECTORSTORE_DB_PATH, embeddings)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=OPENAI_API_KEY), retriever=retriever)

            max_retries = 5
            for attempt in range(max_retries):
                try:
                    return qa.run(query)
                except openai.error.RateLimitError:
                    wait_time = (2 ** attempt)  # Exponential backoff (2, 4, 8, 16 sec)
                    st.warning(f"⚠️ Rate limit reached. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                except openai.error.OpenAIError as e:
                    st.error(f"⚠️ OpenAI API error: {str(e)}")
                    break

        response = ask_question(query)
        if response:
            st.markdown("### 📢 Answer:")
            st.write(response)
    else:
        st.warning("⚠️ Please enter a question.")
