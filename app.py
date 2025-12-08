import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os

# Page configuration
st.set_page_config(
    page_title="Medical Information Chatbot",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTextInput > div > div > input {
        background-color: #f0f2f6;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("🏥 Medical Information Chatbot")
st.markdown("### RAG-powered AI Assistant for Medical Queries")
st.markdown("*This is for educational purposes only. Always consult a healthcare professional for medical advice.*")

# Sidebar for API key input
with st.sidebar:
    st.header("⚙️ Configuration")
    grok_api_key = st.text_input("Enter Groq API Key:", type="password", key="grok_key")
    
    st.markdown("---")
    st.markdown("### 📝 About")
    st.info(
        "This chatbot uses Retrieval-Augmented Generation (RAG) "
        "to provide accurate medical information from a curated knowledge base."
    )
    
    st.markdown("### 🔍 Example Questions")
    st.markdown("""
    - What are the symptoms of diabetes?
    - How to treat high blood pressure?
    - What causes heart disease?
    - Tell me about asthma symptoms
    """)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

# Function to load RAG chain
@st.cache_resource
def load_rag_chain(api_key):
    """Initialize and load the RAG chain"""
    try:
        # Initialize embeddings model
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )
        
        # Load the FAISS vector store
        vectorstore = FAISS.load_local(
            "data/faiss_index", 
            embeddings=embeddings, 
            allow_dangerous_deserialization=True
        )
        
        # Create a retriever
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        
        # Initialize the LLM
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.3,
            api_key=api_key
        )
        
        # Create a prompt template
        template = """You are a helpful medical information assistant. Use the following context from medical documents to answer the user's question accurately and comprehensively.

If the answer is not in the context, say "I don't have enough information in my knowledge base to answer this question accurately. Please consult a healthcare professional."

Context:
{context}

Question: {question}

Answer:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # Helper function to format retrieved documents
        def format_docs(docs):
            return "\n\n".join([doc.page_content for doc in docs])
        
        # Create the RAG chain
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        return rag_chain, True
    except Exception as e:
        return None, str(e)

# Check if API key is provided
if grok_api_key:
    if st.session_state.rag_chain is None:
        with st.spinner("Loading medical knowledge base..."):
            rag_chain, status = load_rag_chain(grok_api_key)
            if rag_chain:
                st.session_state.rag_chain = rag_chain
                st.success("✅ System ready!")
            else:
                st.error(f"❌ Error loading system: {status}")
else:
    st.warning("⚠️ Please enter your Groq API Key in the sidebar to start chatting.")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask your medical question here..."):
    if not grok_api_key:
        st.error("Please enter your Groq API Key in the sidebar first!")
    elif st.session_state.rag_chain is None:
        st.error("Please wait for the system to initialize!")
    else:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get bot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.rag_chain.invoke(prompt)
                    st.markdown(response)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Clear chat button in sidebar
with st.sidebar:
    st.markdown("---")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
