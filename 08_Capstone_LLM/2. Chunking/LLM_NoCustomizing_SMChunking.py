#!/usr/bin/env python
# coding: utf-8

# # LLM_NoCustomizing

# In[4]:


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
openai.api_key = os.getenv("OPENAI_API_KEY")

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
query = "Summarize the CEDS(comprehensive Economic Development Strategy) in about 10000 characters"
summary = qa_chain.run(query)

print("Generated Summary:")
print(summary)


# ## Cosine Similarity Evaluation

# In[9]:


from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from nltk.tokenize import sent_tokenize

def detect_hallucinations_with_cosine(summary, retriever, embeddings, max_docs=3, similarity_threshold=0.7):
    """
    Measure hallucination rate by comparing each sentence in the summary
    with the top retrieved documents using cosine similarity on embeddings.
    If the average similarity to retrieved docs is below a threshold, 
    the sentence is considered a hallucination.
    """
    sentences = sent_tokenize(summary)
    unsupported_sentences = []
    supported_sentences = []

    for sentence in sentences:
        # Retrieve top relevant documents
        relevant_docs = retriever.get_relevant_documents(sentence)[:max_docs]
        
        # If no documents retrieved, consider it unsupported
        if not relevant_docs:
            unsupported_sentences.append((sentence, "No relevant documents retrieved"))
            continue
        
        # Embed sentence and documents
        sentence_embedding = embeddings.embed_documents([sentence])
        doc_embeddings = embeddings.embed_documents([doc.page_content for doc in relevant_docs])
        
        # Calculate cosine similarities and take the average
        sims = [cosine_similarity([sentence_embedding[0]], [de])[0][0] for de in doc_embeddings]
        avg_sim = np.mean(sims)
        
        if avg_sim < similarity_threshold:
            unsupported_sentences.append((sentence, f"Average similarity {avg_sim:.2f} < {similarity_threshold}"))
        else:
            supported_sentences.append((sentence, f"Average similarity {avg_sim:.2f} >= {similarity_threshold}"))

    hallucination_rate = len(unsupported_sentences) / len(sentences) * 100 if sentences else 0
    return hallucination_rate, unsupported_sentences, supported_sentences

cosine_hallucination_rate, cosine_unsupported, cosine_supported = detect_hallucinations_with_cosine(
    summary, retriever, embeddings
)

print(f"\n[Cosine Similarity Based] Hallucination Rate: {cosine_hallucination_rate:.2f}%")

if cosine_unsupported:
    print("\nUnsupported Sentences (Cosine):")
    for idx, (sent, explanation) in enumerate(cosine_unsupported, 1):
        print(f"{idx}. {sent}\n   Explanation: {explanation}")
else:
    print("\nAll sentences are supported by the source documents (Cosine).")


# ## Rouge Evaluation

# In[6]:


from rouge_score import rouge_scorer

def detect_hallucinations_with_rouge(summary, retriever, max_docs=3, rouge_threshold=0.2):
    """
    Measure hallucination rate by comparing each sentence in the summary
    with retrieved documents using ROUGE-L scores. If the best ROUGE-L recall score
    against the retrieved docs is below a certain threshold, consider it a hallucination.
    """
    sentences = sent_tokenize(summary)
    unsupported_sentences = []
    supported_sentences = []
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    for sentence in sentences:
        relevant_docs = retriever.get_relevant_documents(sentence)[:max_docs]

        if not relevant_docs:
            unsupported_sentences.append((sentence, "No relevant documents retrieved"))
            continue

        # Calculate ROUGE-L scores against each retrieved doc, take the best
        rouge_scores = []
        for doc in relevant_docs:
            scores = scorer.score(doc.page_content, sentence)
            rouge_scores.append(scores['rougeL'].recall)

        best_rouge_l = max(rouge_scores) if rouge_scores else 0.0

        if best_rouge_l < rouge_threshold:
            unsupported_sentences.append((sentence, f"Best ROUGE-L recall {best_rouge_l:.2f} < {rouge_threshold}"))
        else:
            supported_sentences.append((sentence, f"Best ROUGE-L recall {best_rouge_l:.2f} >= {rouge_threshold}"))

    hallucination_rate = len(unsupported_sentences) / len(sentences) * 100 if sentences else 0
    return hallucination_rate, unsupported_sentences, supported_sentences

rouge_hallucination_rate, rouge_unsupported, rouge_supported = detect_hallucinations_with_rouge(
    summary, retriever
)

print(f"\n[ROUGE-Based Evaluation] Hallucination Rate: {rouge_hallucination_rate:.2f}%")

if rouge_unsupported:
    print("\nUnsupported Sentences (ROUGE):")
    for idx, (sent, explanation) in enumerate(rouge_unsupported, 1):
        print(f"{idx}. {sent}\n   Explanation: {explanation}")
else:
    print("\nAll sentences are supported by the source documents (ROUGE).")


# ## NLI Evaluation

# In[7]:


from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from nltk.tokenize import sent_tokenize

def detect_hallucinations_with_sliding_nli(summary, retriever, max_docs=3, window_size=300, step_size=150, nli_model="facebook/bart-large-mnli"):
    """
    Measure hallucination by:
    1. Splitting retrieved document context into sliding windows.
    2. Using an NLI model to check if the summary sentence is entailed by any window.

    If no window of the retrieved documents provides 'entailment', 
    consider the sentence hallucinated.
    """

    # Load NLI model and tokenizer
    nli_classifier = pipeline("text-classification", model=nli_model, tokenizer=nli_model, return_all_scores=True)

    sentences = sent_tokenize(summary)
    unsupported_sentences = []
    supported_sentences = []

    for sentence in sentences:
        # Retrieve top relevant documents
        relevant_docs = retriever.get_relevant_documents(sentence)[:max_docs]
        doc_text = " ".join([doc.page_content for doc in relevant_docs])

        if not doc_text.strip():
            unsupported_sentences.append((sentence, "No relevant context retrieved"))
            continue

        # Sliding window over the doc_text
        # Note: window_size and step_size are character-based splits here for simplicity
        # You can adapt to token-based splits if necessary.
        doc_length = len(doc_text)
        entailment_found = False

        for start_idx in range(0, doc_length, step_size):
            end_idx = start_idx + window_size
            window_text = doc_text[start_idx:end_idx]

            # For NLI, we use premise = window_text, hypothesis = sentence
            # NLI model typically expects inputs in premise-hypothesis form.
            results = nli_classifier(f"{window_text} </s> {sentence}")

            # results looks like: [[{'label': 'ENTAILMENT', 'score': ...}, {'label':'NEUTRAL',...}, ...]]
            # Find the entailment score
            for r in results[0]:
                if r['label'].lower() == 'entailment' and r['score'] > 0.5:
                    entailment_found = True
                    break

            if entailment_found:
                break

        if entailment_found:
            supported_sentences.append((sentence, "Entailment found in at least one window"))
        else:
            unsupported_sentences.append((sentence, "No entailment found across all windows"))

    hallucination_rate = (len(unsupported_sentences) / len(sentences) * 100) if sentences else 0
    return hallucination_rate, unsupported_sentences, supported_sentences

nli_hallucination_rate, nli_unsupported, nli_supported = detect_hallucinations_with_sliding_nli(
    summary, retriever
)

print(f"\n[Sliding Window + NLI Based] Hallucination Rate: {nli_hallucination_rate:.2f}%")

if nli_unsupported:
    print("\nUnsupported Sentences (NLI):")
    for idx, (sent, explanation) in enumerate(nli_unsupported, 1):
        print(f"{idx}. {sent}\n   Explanation: {explanation}")
else:
    print("\nAll sentences are supported by the source documents (NLI).")


# ## LLM Evaluation

# In[8]:


import os
import openai
from dotenv import load_dotenv
from nltk.tokenize import sent_tokenize

# Load environment variables
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

def detect_hallucinations_with_gpt_direct(summary, retriever, max_docs=3):
    """
    Use GPT (e.g., GPT-3.5-turbo) to evaluate each sentence.
    The prompt directly asks GPT if the sentence is supported by the provided context.
    """
    sentences = sent_tokenize(summary)
    unsupported_sentences = []
    supported_sentences = []

    for sentence in sentences:
        relevant_docs = retriever.get_relevant_documents(sentence)[:max_docs]
        context = " ".join([doc.page_content for doc in relevant_docs])

        if not context.strip():
            unsupported_sentences.append((sentence, "No context returned by retriever"))
            continue

        prompt = (
            f"Below is a context extracted from source documents:\n\n{context}\n\n"
            f"Check if the following sentence is directly supported by the provided context. "
            f"If yes, answer 'Yes' and provide a brief reasoning. If no, answer 'No' and explain why.\n\n"
            f"Sentence: \"{sentence}\""
        )

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant who verifies factual alignment."},
                    {"role": "user", "content": prompt}
                ]
            )

            answer = response['choices'][0]['message']['content'].lower()

            if "yes" in answer:
                supported_sentences.append((sentence, answer))
            else:
                unsupported_sentences.append((sentence, answer))

        except Exception as e:
            print(f"Error querying GPT: {e}")
            unsupported_sentences.append((sentence, "Error: Unable to evaluate with GPT"))

    hallucination_rate = len(unsupported_sentences) / len(sentences) * 100 if sentences else 0
    return hallucination_rate, unsupported_sentences, supported_sentences

gpt_hallucination_rate, gpt_unsupported, gpt_supported = detect_hallucinations_with_gpt_direct(
    summary, retriever
)

print(f"\n[GPT-Based Evaluation] Hallucination Rate: {gpt_hallucination_rate:.2f}%")

if gpt_unsupported:
    print("\nUnsupported Sentences (GPT):")
    for idx, (sent, explanation) in enumerate(gpt_unsupported, 1):
        print(f"{idx}. {sent}\n   Explanation: {explanation}")
else:
    print("\nAll sentences are supported by the source documents (GPT).")

