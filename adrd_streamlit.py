import os
import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import AzureOpenAIEmbeddings
from langchain.chat_models import AzureChatOpenAI
from langchain.vectorstores import Chroma
from langchain.schema import Document
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate


# Load environment variables
API_KEY = st.secrets["API_KEY_OPEN_AI"]
API_VERSION = st.secrets["API_VERSION"]
RESOURCE_ENDPOINT = st.secrets["RESOURCE_ENDPOINT"]
ACCESS_CODE = st.secrets["APP_ACCESS_CODE"]

# Load from secrets or fallback to env
def check_access():
    code = st.text_input("🔒 Enter access code to continue:", type="password")
    if code != ACCESS_CODE:
        st.warning("Incorrect code. Please try again.")
        st.stop()

check_access()

@st.cache_resource
def load_vectorstore():
    persist_path = "chroma_index"

    embeddings = AzureOpenAIEmbeddings(
        openai_api_key=API_KEY,
        openai_api_base=RESOURCE_ENDPOINT,
        openai_api_version=API_VERSION,
        openai_api_type="azure",  # Required for Azure
        chunk_size=1000,           # Must be ≤ 2048; pick a value based on your model context
        validate_base_url=False   # ✅ add this line
    )    

    if os.path.exists(persist_path) and os.listdir(persist_path):
        return Chroma(persist_directory=persist_path, embedding_function=embeddings)

    # Load and chunk transcripts
    with open("data/clean_transcript1.txt", "r", encoding="utf-8") as f:
        transcript1_2022 = f.read()
    with open("data/clean_transcript2.txt", "r", encoding="utf-8") as f:
        transcript2_2022 = f.read()
    with open("data/adrd042925.txt", "r", encoding="utf-8") as f:
        transcript1_2025 = f.read()
    with open("data/adrd043025.txt", "r", encoding="utf-8") as f:
        transcript2_2025 = f.read()
    with open("data/adrd060225.txt", "r", encoding="latin1") as f:
        transcript3_2025 = f.read()

    transcript2022 = "[SYMPOSIUM_2022]\n" + transcript1_2022 + transcript2_2022
    transcript2025 = "[SYMPOSIUM_2025]\n" + transcript1_2025 + transcript2_2025 + transcript3_2025

    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks_2022 = splitter.split_text(transcript2022)
    chunks_2025 = splitter.split_text(transcript2025)

    labeled_2022 = [Document(page_content=c, metadata={"source": "2022"}) for c in chunks_2022]
    labeled_2025 = [Document(page_content=c, metadata={"source": "2025"}) for c in chunks_2025]
    all_chunks = labeled_2022 + labeled_2025

    vectorstore = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory=persist_path
    )
    vectorstore.persist()
    return vectorstore

# Load vectorstore and model
vectorstore = load_vectorstore()
chat_llm = AzureChatOpenAI(
    deployment_name="gpt-4-turbo-128k",
    openai_api_key=API_KEY,
    openai_api_base=RESOURCE_ENDPOINT,
    openai_api_version=API_VERSION,
    temperature=0,
)

# Prompt for summarization
summary_prompt = PromptTemplate(
    input_variables=["text"],
    template="""
You are an expert summarizer. Given the following text extracted from a dementia conference transcript,
identify and summarize the key themes discussed. Present your findings as bullet points with a brief explanation each.

{text}
"""
)
summary_chain = LLMChain(llm=chat_llm, prompt=summary_prompt)

# Streamlit UI
st.title("\U0001F9E0 ADRD Symposium LLM Explorer")
st.write("Compare key themes and insights from the 2022 and 2025 ADRD Symposiums.")

year_selection = st.selectbox("Select Symposium Year", ["All", "2022", "2025"])
question = st.text_input("Ask a question (e.g., 'What did they say about caregiver burden?')")

if st.button("Run Q&A") and question:
    filter_kwargs = {"k": 5}
    if year_selection in ["2022", "2025"]:
        filter_kwargs["filter"] = {"source": year_selection}

    retriever = vectorstore.as_retriever(search_kwargs=filter_kwargs)
    qa = RetrievalQA.from_chain_type(llm=chat_llm, chain_type="stuff", retriever=retriever)
    answer = qa.run(question)
    st.subheader("Answer")
    st.write(answer)

if st.button("Summarize Key Themes"):
    filter_kwargs = {"k": 10}
    if year_selection in ["2022", "2025"]:
        filter_kwargs["filter"] = {"source": year_selection}
    relevant_docs = vectorstore.similarity_search("overview of key themes in the dementia conference", **filter_kwargs)
    combined_text = "\n\n".join([doc.page_content for doc in relevant_docs])
    summary = summary_chain.run(text=combined_text)
    st.subheader("Summary of Key Themes")
    st.write(summary)
