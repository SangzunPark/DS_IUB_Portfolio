#!/usr/bin/env python
# coding: utf-8

# # LLM_NoCustomizing

# In[10]:


import os
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from nltk.tokenize import sent_tokenize
from dotenv import load_dotenv
import openai
import nltk

nltk.download('punkt')

# Load environment variables
load_dotenv()

# Set paths for multiple PDF files
pdf_paths = ["Contents/eda_ceds_guidelines_2023.pdf"]

# Load PDF files
documents = []
for path in pdf_paths:
    loader = PyPDFLoader(path)
    documents.extend(loader.load())

# --- GPT-3.5 Based Chunking ---
def gpt_chunking(documents, chunk_size=1000):
    """
    Use GPT-3.5 to semantically chunk the text from documents.
    Each chunk is generated based on GPT's understanding of context.
    """
    chunks = []

    for doc in documents:
        sentences = sent_tokenize(doc.page_content)
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            # Add sentence to the current chunk if it doesn't exceed the chunk size
            if current_length + len(sentence) <= chunk_size:
                current_chunk.append(sentence)
                current_length += len(sentence)
            else:
                # Process the current chunk with GPT to refine it
                refined_chunk = process_with_gpt(" ".join(current_chunk))
                chunks.append(Document(page_content=refined_chunk))
                
                # Start a new chunk
                current_chunk = [sentence]
                current_length = len(sentence)

        # Process the final chunk
        if current_chunk:
            refined_chunk = process_with_gpt(" ".join(current_chunk))
            chunks.append(Document(page_content=refined_chunk))

    return chunks

def process_with_gpt(text):
    """
    Use GPT-3.5 to refine and validate a chunk of text.
    """
    prompt = (
        f"Given the following text:\n\n{text}\n\n"
        "Split this text into a coherent chunk that makes sense contextually. "
        "Make sure the chunk captures the main idea and is self-contained."
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500
        )
        chunk = response['choices'][0]['message']['content'].strip()
        return chunk
    except Exception as e:
        print(f"Error processing with GPT: {e}")
        return text  # Fallback to original text if GPT processing fails

# Perform GPT-based chunking
texts = gpt_chunking(documents)

# Initialize embedding model for LangChain
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vectorstore = Chroma.from_documents(texts, embeddings)
retriever = vectorstore.as_retriever()

# Initialize the LLM model
api_key = os.getenv("MISTRAL_API_KEY")
llm = ChatMistralAI(api_key=api_key)

# Create a RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Define the query for summarization
query = "Summarize this document in about 10000 characters"
summary = qa_chain.run(query)

print("Generated Summary:")
print(summary)


openai.api_key = os.getenv("OPENAI_API_KEY")

# --- OpenAI GPT Based Hallucination ---
def detect_hallucinations_with_gpt(summary, retriever, max_docs=3):
    """
    Use OpenAI GPT to evaluate if each sentence in the summary aligns with the source documents.
    """
    sentences = sent_tokenize(summary)
    unsupported_sentences = []
    supported_sentences = []

    for sentence in sentences:
        # Retrieve relevant documents
        relevant_docs = retriever.get_relevant_documents(sentence)[:max_docs]
        context = " ".join([doc.page_content for doc in relevant_docs])

        # Query OpenAI GPT to check alignment
        prompt = (
            f"Given the following context:\n\n{context}\n\n"
            f"Does this sentence align with the context? Answer 'Yes' or 'No', and briefly explain:\n\n"
            f"Sentence: \"{sentence}\""
        )

        try:
            # Updated OpenAI API usage
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            answer = response['choices'][0]['message']['content'].strip().lower()
            
            if "yes" in answer:
                supported_sentences.append((sentence, answer))
            else:
                unsupported_sentences.append((sentence, answer))
        
        except Exception as e:
            print(f"Error querying OpenAI GPT: {e}")
            unsupported_sentences.append((sentence, "Error: Unable to evaluate"))

    hallucination_rate = len(unsupported_sentences) / len(sentences) * 100
    return hallucination_rate, unsupported_sentences, supported_sentences

# Detect hallucinations in the generated summary
hallucination_rate, unsupported_sentences, supported_sentences = detect_hallucinations_with_gpt(
    summary, retriever
)

print(f"\nHallucination Rate: {hallucination_rate:.2f}%")

if unsupported_sentences:
    print("\nUnsupported Sentences:")
    for idx, (sent, explanation) in enumerate(unsupported_sentences, 1):
        print(f"{idx}. {sent}\n   Explanation: {explanation}")
else:
    print("\nAll sentences are supported by the source documents.")

