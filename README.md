# 🏥 Medical Information Chatbot

A RAG-powered (Retrieval-Augmented Generation) medical information chatbot built with LangChain, FAISS, and Streamlit. This chatbot provides accurate medical information by retrieving relevant content from a curated knowledge base.

## ⚠️ Disclaimer

**This application is for educational purposes only. Always consult qualified healthcare professionals for medical advice, diagnosis, or treatment.**

## 🌟 Features

- **RAG Architecture**: Uses FAISS vector database for efficient similarity search
- **Accurate Responses**: Retrieves information from medical encyclopedias and documents
- **User-Friendly Interface**: Built with Streamlit for easy interaction
- **Conversational AI**: Powered by Groq's LLaMA 3.1 model
- **Chat History**: Maintains conversation context

## 🚀 Deployment on Streamlit Cloud

### Prerequisites

1. A [Groq API Key](https://console.groq.com/) (free tier available)
2. A GitHub account
3. A Streamlit Cloud account (sign up at [streamlit.io/cloud](https://streamlit.io/cloud))

### Steps to Deploy

1. **Push to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin <your-github-repo-url>
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Select your GitHub repository
   - Set main file path: `app.py`
   - Click "Deploy"

3. **Configure API Key**:
   - In Streamlit Cloud dashboard, go to your app settings
   - Add your Groq API key in the "Secrets" section:
     ```toml
     GROK_KEY = "your-groq-api-key-here"
     ```
   - Or enter it directly in the sidebar when using the app

## 🏃 Running Locally

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd Health-Agent
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   # source venv/bin/activate  # On Mac/Linux
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app**:
   ```bash
   streamlit run app.py
   ```

5. **Enter your Groq API key** in the sidebar

## 📁 Project Structure

```
Health-Agent/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── .gitignore                  # Git ignore file
├── data/
│   └── faiss_index/           # FAISS vector database
│       ├── index.faiss
│       └── index.pkl
└── .streamlit/
    └── config.toml            # Streamlit configuration
```

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **LLM**: Groq (LLaMA 3.1)
- **Vector Database**: FAISS
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Framework**: LangChain

## 📝 How It Works

1. User enters a medical question
2. The question is embedded using HuggingFace embeddings
3. FAISS retrieves the top 3 most relevant document chunks
4. The retrieved context and question are sent to Groq's LLM
5. The LLM generates an accurate response based on the context
6. The response is displayed to the user

## 🔐 Security Notes

- Never commit API keys to GitHub
- Use Streamlit Cloud's secrets management for production
- The `.gitignore` file excludes sensitive files

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 📧 Contact

For questions or support, please open an issue on GitHub.
