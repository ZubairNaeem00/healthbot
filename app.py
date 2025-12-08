import os
import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain_groq import ChatGroq


GROK_KEY = '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'

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

# Sidebar for API key input (falls back to seckret_keys if provided)
with st.sidebar:
    st.header("⚙️ Configuration")
    grok_api_key = st.text_input(
        "Enter Groq API Key:", type="password", value=GROK_KEY or ""
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
    vectorstore = FAISS.load_local(
        "data/faiss_index",
        embeddings=HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        ),
        allow_dangerous_deserialization=True,
    )
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.1, api_key=api_key)
    qa = ConversationalRetrievalChain.from_llm(
        llm, retriever=retriever, memory=memory
    )
    return qa, llm


def build_agent(api_key: str):
    qa_chain, llm = build_rag(api_key)
    wiki_tool = Tool(
        name="Wikipedia Search",
        func=WikipediaAPIWrapper().run,
        description="Use this to look up medical info on Wikipedia",
    )
    agent = initialize_agent(
        tools=[wiki_tool],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,
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
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.qa_chain.invoke({"question": prompt})
                answer = response.get("answer", "") if isinstance(response, dict) else str(response)
                st.markdown(answer)
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": answer}
                )
            except Exception as e:
                st.error(f"Error: {e}")

# Display prior chat
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

with st.sidebar:
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()
