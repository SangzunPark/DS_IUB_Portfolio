#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import langchain as lc
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv


# Set paths for multiple PDF files
pdf_paths = ["Contents/eda_ceds_guidelines_2023.pdf"]

# Load PDF files
documents = []
for path in pdf_paths:
    loader = PyPDFLoader(path)
    documents.extend(loader.load())

# Split text into appropriate chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# Initialize embedding model
embeddings = HuggingFaceEmbeddings()

# Create and index the Chroma vector store
vectorstore = Chroma.from_documents(texts, embeddings)

# Create a retriever from the vector store
retriever = vectorstore.as_retriever()

# Initialize the LLM model
load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")
llm = ChatMistralAI(api_key=api_key)

# Create a RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Define the query for summarization
query = "Summarize this document in about 5000 characters"

# Get the summary result
summary = qa_chain.run(query)
print(summary)

