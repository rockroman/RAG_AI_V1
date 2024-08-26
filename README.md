## Groq RAG AI Chat Application

This is a simple chat application built using the Groq RAG AI model from Langchain. The application allows you to input a prompt and get a response based on the documents loaded in the application.

### Prerequisites

Before running the application, make sure you have the following installed:

- Python 3.x
- Streamlit
- Langchain
- Hugging Face Transformers
- dotenv

### Getting Started

- Clone the repository
- Create a .env file in the root directory of the project and add your Groq API key:

- Run the application:
  streamlit run app.py

## How it Works

The application loads a document from the web using the WebBaseLoader class from Langchain. The document is then split into smaller chunks using the RecursiveCharacterTextSplitter class. The chunks are then embedded using the HuggingFaceEmbeddings class and stored in a FAISS vector database.

When you input a prompt, the application creates a prompt template using the ChatPromptTemplate class from Langchain. The prompt template includes the context if it is relevant to the question. The application then creates a document chain and retrieval chain using the create_stuff_documents_chain and create_retrieval_chain functions from Langchain. The retrieval chain is then used to get a response to the prompt.

### FAQ

- Q: Can I load my own documents?
- A: Yes, you can modify the WebBaseLoader class to load your own documents.

---

- Q: Can I use a different model?
- A: Yes, you can modify the llm variable to use a different model.

---

- Q: Can I change the chunk size and overlap?
- A: Yes, you can modify the RecursiveCharacterTextSplitter class to change the chunk size and overlap.

---

- Q: Can I change the vector database?
- A: Yes, you can modify the FAISS class to use a different vector database.

---
