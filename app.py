import os
import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain_groq import ChatGroq

# Get absolute path to data directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "data", "faiss_index")


# Download FAISS index files from Hugging Face if missing
import requests
def download_file(url, dest_path):
    if not os.path.exists(dest_path):
        print(f"Downloading {url} to {dest_path}...")
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")

os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
faiss_url = "https://huggingface.co/datasets/zubiarnaeem00/encyclopedias/resolve/main/index.faiss"
pkl_url = "https://huggingface.co/datasets/zubiarnaeem00/encyclopedias/resolve/main/index.pkl"
download_file(faiss_url, os.path.join(FAISS_INDEX_PATH, "index.faiss"))
download_file(pkl_url, os.path.join(FAISS_INDEX_PATH, "index.pkl"))

# Cache embeddings to avoid repeated downloads
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_folder=os.path.join(BASE_DIR, ".cache")
    )

# Page configuration
st.set_page_config(
    page_title="Medical Information Chatbot",
    page_icon="🏥",
    layout="wide",
)

st.title("🏥 Medical Information Chatbot")
st.markdown(
    "*Educational use only. Always consult a healthcare professional for medical advice.*"
)

# Sidebar for API key input
with st.sidebar:
    st.header("⚙️ Configuration")
    grok_api_key = st.text_input(
        "Enter Groq API Key:", type="password"
    )
    st.markdown("---")
    st.info("RAG + Wikipedia tool. Index at data/faiss_index.")

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "agent" not in st.session_state:
    st.session_state.agent = None


@st.cache_resource

def build_rag(api_key: str):
    embeddings = get_embeddings()
    try:
        vectorstore = FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
        # Correctly check number of docs in vectorstore
        num_docs = None
        if hasattr(vectorstore, 'docstore') and hasattr(vectorstore.docstore, 'docs'):
            num_docs = len(vectorstore.docstore.docs)
        st.write(f"FAISS index loaded. Number of docs: {num_docs if num_docs is not None else 'unknown'}")
        if num_docs == 0 or num_docs is None:
            st.error("FAISS index loaded but contains no documents! Check your index files.")
    except Exception as e:
        st.error(f"Failed to load FAISS index: {e}")
        raise
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    )
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.5, api_key=api_key)
    # Use simple memory without return_messages
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    qa = ConversationalRetrievalChain.from_llm(
        llm, retriever=retriever, memory=memory, verbose=False
    )
    return qa, llm


def build_agent(api_key: str):
    qa_chain, llm = build_rag(api_key)   # make sure you defined build_rag()

    # Wikipedia tool
    wiki_tool = Tool(
        name="Wikipedia Search",
        func=WikipediaAPIWrapper().run,
        description="Use this to look up medical info on Wikipedia"
    )

    # Agent setup
    agent = initialize_agent(
        tools=[wiki_tool],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False
    )

    return agent, qa_chain


# Initialize chains once API key is provided
if grok_api_key:
    if st.session_state.qa_chain is None:
        with st.spinner("Loading RAG index and tools..."):
            try:
                agent, qa_chain = build_agent(grok_api_key)
                st.session_state.agent = agent
                st.session_state.qa_chain = qa_chain
                st.success("Ready!")
            except Exception as e:
                st.error(f"Failed to load: {e}")
else:
    st.warning("Enter your Groq API key to start.")



# Chat UI
prompt = st.chat_input("Ask your medical question...")
if prompt and grok_api_key and st.session_state.qa_chain:
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.spinner("Thinking..."):
        try:
            response = st.session_state.qa_chain.invoke({"question": prompt})
            answer = response.get("answer", "") if isinstance(response, dict) else str(response)
        except Exception as e:
            answer = f"Error: {e}"
    st.session_state.chat_history.append({"role": "assistant", "content": answer})

# Display prior chat
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

with st.sidebar:
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()
        
