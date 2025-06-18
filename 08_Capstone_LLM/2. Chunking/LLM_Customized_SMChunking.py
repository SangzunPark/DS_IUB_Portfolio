#!/usr/bin/env python
# coding: utf-8

# ##  LLM Customized

# In[7]:


import os
import uuid
import nltk
import time
import httpx
from typing import Optional
from dotenv import load_dotenv
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from transformers import GPT2TokenizerFast

class AgenticChunker:
    def __init__(self, mistral_api_key=None, max_tokens=1023):
        self.chunks = {}
        self.id_truncate_limit = 5
        self.max_tokens = max_tokens  # Set the maximum number of tokens per chunk
        self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')  # Select the appropriate tokenizer

        # Whether to update summary and title
        self.generate_new_metadata_ind = True
        self.print_logging = True

        if mistral_api_key is None:
            mistral_api_key = os.getenv("MISTRAL_API_KEY")

        if mistral_api_key is None:
            raise ValueError("API key is not provided and not found in environment variables")

        self.llm = ChatMistralAI(api_key=mistral_api_key, temperature=0)

    def add_propositions(self, propositions):
        for proposition in propositions:
            retry_count = 0
            while retry_count < 5:
                try:
                    self.add_proposition(proposition)
                    time.sleep(1)  # Add a 1-second delay between requests
                    break
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 429:  # Rate limit error
                        print("Rate limit exceeded, retrying...")
                        time.sleep(5)  # Wait 5 seconds before retrying
                        retry_count += 1
                    else:
                        raise

    def add_proposition(self, proposition):
        if self.print_logging:
            print(f"\nAdding: '{proposition}'")

        # If it's the first chunk, create a new one
        if len(self.chunks) == 0:
            if self.print_logging:
                print("No chunks, creating a new one")
            self._create_new_chunk(proposition)
            return

        chunk_id = self._find_relevant_chunk(proposition)

        # If there's an appropriate chunk, add to it
        if chunk_id and chunk_id in self.chunks:
            if self.print_logging:
                print(f"Chunk Found ({self.chunks[chunk_id]['chunk_id']}), adding to: \"{self.chunks[chunk_id]['title']}\"")
            self.add_proposition_to_chunk(chunk_id, proposition)
            return
        else:
            if self.print_logging:
                print("No valid chunk found, creating a new one")
            self._create_new_chunk(proposition)
            

    def add_proposition_to_chunk(self, chunk_id, proposition):
        # Calculate the number of tokens in the current chunk
        current_tokens = self._calculate_chunk_tokens(self.chunks[chunk_id]['propositions'])

        # Calculate the number of tokens in the new proposition
        proposition_tokens = len(self.tokenizer.encode(proposition))

        # Calculate the total number of tokens after adding the new proposition
        total_tokens = current_tokens + proposition_tokens

        # Check if the total number of tokens exceeds the maximum
        if total_tokens > self.max_tokens:
            if self.print_logging:
                print(f"Chunk {chunk_id} will exceed max tokens. Creating a new chunk.")
            # Create a new chunk
            self._create_new_chunk(proposition)
        else:
            # Add the proposition
            self.chunks[chunk_id]['propositions'].append(proposition)

            # Update summary and title
            if self.generate_new_metadata_ind:
                self.chunks[chunk_id]['summary'] = self._update_chunk_summary(self.chunks[chunk_id])
                self.chunks[chunk_id]['title'] = self._update_chunk_title(self.chunks[chunk_id])
                

    def _calculate_chunk_tokens(self, propositions):
        text = ' '.join(propositions)
        tokens = self.tokenizer.encode(text)
        return len(tokens)

    def _create_new_chunk(self, proposition):
        # Calculate the number of tokens in the new proposition
        proposition_tokens = len(self.tokenizer.encode(proposition))

        # Check if the proposition itself exceeds max_tokens
        if proposition_tokens > self.max_tokens:
            if self.print_logging:
                print(f"Proposition exceeds max tokens. Splitting the proposition.")
            # Add logic to split or handle the proposition if necessary

        new_chunk_id = str(uuid.uuid4())[:self.id_truncate_limit]
        new_chunk_summary = self._get_new_chunk_summary(proposition)
        new_chunk_title = self._get_new_chunk_title(new_chunk_summary)

        self.chunks[new_chunk_id] = {
            'chunk_id': new_chunk_id,
            'propositions': [proposition],
            'title': new_chunk_title,
            'summary': new_chunk_summary,
            'chunk_index': len(self.chunks)
        }
        if self.print_logging:
            print(f"Created new chunk ({new_chunk_id}): {new_chunk_title}")

            
           
        
        
        
            
            

    def _update_chunk_summary(self, chunk):
        PROMPT = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic.
                    A new proposition was just added to one of your chunks. You should generate a very brief 1-sentence summary which will inform viewers what the chunk group is about.

                    A good summary will say what the chunk is about and give any clarifying instructions on what to add to the chunk.

                    You will be given a group of propositions which are in the chunk and the chunk's current summary.

                    Your summaries should anticipate generalization. If you get a proposition about apples, generalize it to food. Or month, generalize it to "dates and times".

                    Example:
                    Input: Proposition: Greg likes to eat pizza
                    Output: This chunk contains information about the types of food Greg likes to eat.

                    Only respond with the chunk's new summary, nothing else.
                    """,
                ),
                ("user", "Chunk's propositions:\n{proposition}\n\nCurrent chunk summary:\n{current_summary}"),
            ]
        )

        runnable = PROMPT | self.llm

        new_chunk_summary = runnable.invoke({
            "proposition": "\n".join(chunk['propositions']),
            "current_summary": chunk['summary']
        }).content

        return new_chunk_summary

    def _update_chunk_title(self, chunk):
        PROMPT = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic.
                    A new proposition was just added to one of your chunks. You should generate a very brief updated chunk title which will inform viewers what the chunk group is about.

                    A good title will say what the chunk is about.

                    You will be given a group of propositions which are in the chunk, the chunk summary, and the chunk title.

                    Your title should anticipate generalization. If you get a proposition about apples, generalize it to food. Or month, generalize it to "dates and times".

                    Example:
                    Input: Summary: This chunk is about dates and times that the author talks about
                    Output: Dates & Times

                    Only respond with the new chunk title, nothing else.
                    """,
                ),
                ("user", "Chunk's propositions:\n{proposition}\n\nChunk summary:\n{current_summary}\n\nCurrent chunk title:\n{current_title}"),
            ]
        )

        runnable = PROMPT | self.llm

        updated_chunk_title = runnable.invoke({
            "proposition": "\n".join(chunk['propositions']),
            "current_summary": chunk['summary'],
            "current_title": chunk['title']
        }).content

        return updated_chunk_title

    def _get_new_chunk_summary(self, proposition):
        PROMPT = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic.
                    You should generate a very brief 1-sentence summary which will inform viewers what the chunk group is about.

                    A good summary will say what the chunk is about and give any clarifying instructions on what to add to the chunk.

                    You will be given a proposition which will go into a new chunk. This new chunk needs a summary.

                    Your summaries should anticipate generalization. If you get a proposition about apples, generalize it to food. Or month, generalize it to "dates and times".

                    Example:
                    Input: Proposition: Greg likes to eat pizza
                    Output: This chunk contains information about the types of food Greg likes to eat.

                    Only respond with the new chunk summary, nothing else.
                    """,
                ),
                ("user", "Determine the summary of the new chunk that this proposition will go into:\n{proposition}"),
            ]
        )

        runnable = PROMPT | self.llm

        new_chunk_summary = runnable.invoke({
            "proposition": proposition
        }).content

        return new_chunk_summary

    def _get_new_chunk_title(self, summary):
        PROMPT = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic.
                    You should generate a very brief few-word chunk title which will inform viewers what the chunk group is about.

                    A good chunk title is brief but encompasses what the chunk is about.

                    You will be given a summary of a chunk which needs a title.

                    Your titles should anticipate generalization. If you get a proposition about apples, generalize it to food. Or month, generalize it to "dates and times".

                    Example:
                    Input: Summary: This chunk is about dates and times that the author talks about
                    Output: Dates & Times

                    Only respond with the new chunk title, nothing else.
                    """,
                ),
                ("user", "Determine the title of the chunk that this summary belongs to:\n{summary}"),
            ]
        )

        runnable = PROMPT | self.llm

        new_chunk_title = runnable.invoke({
            "summary": summary
        }).content

        return new_chunk_title

    def get_chunk_outline(self):
        chunk_outline = ""

        for chunk_id, chunk in self.chunks.items():
            single_chunk_string = f"""Chunk ID: {chunk['chunk_id']}\nChunk Name: {chunk['title']}\nChunk Summary: {chunk['summary']}\n\n"""
            chunk_outline += single_chunk_string

        return chunk_outline

    def _find_relevant_chunk(self, proposition):
        current_chunk_outline = self.get_chunk_outline()

        PROMPT = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    Determine whether or not the "Proposition" should belong to any of the existing chunks.

                    A proposition should belong to a chunk if their meaning, direction, or intention are similar.
                    The goal is to group similar propositions and chunks.

                    If you think a proposition should be joined with a chunk, return the chunk id.
                    If you do not think an item should be joined with an existing chunk, just return "No chunks"

                    Example:
                    Input:
                        - Proposition: "Greg really likes hamburgers"
                        - Current Chunks:
                            - Chunk ID: 2n4l3d
                            - Chunk Name: Places in San Francisco
                            - Chunk Summary: Overview of the things to do with San Francisco Places

                            - Chunk ID: 93833k
                            - Chunk Name: Food Greg likes
                            - Chunk Summary: Lists of the food and dishes that Greg likes
                    Output: 93833k
                    """,
                ),
                ("user", "Current Chunks:\n--Start of current chunks--\n{current_chunk_outline}\n--End of current chunks--"),
                ("user", "Determine if the following statement should belong to one of the chunks outlined:\n{proposition}"),
            ]
        )
        runnable = PROMPT | self.llm

        chunk_found_content = runnable.invoke({
            "proposition": proposition,
            "current_chunk_outline": current_chunk_outline
        }).content.strip()

        # Extract chunk ID from the response
        if chunk_found_content.lower() == "no chunks":
            return None
        else:
            # Extract and validate chunk ID from the response
            import re
            # Expected chunk ID pattern (e.g., 5 alphanumeric characters)
            match = re.search(r'\b([a-f0-9]{{{0}}})\b'.format(self.id_truncate_limit), chunk_found_content)
            if match:
                chunk_id = match.group(1)
                if chunk_id in self.chunks:
                    return chunk_id
            return None


    def get_chunks(self, get_type='dict'):
        if get_type == 'dict':
            return self.chunks
        if get_type == 'list_of_strings':
            chunks = []
            for chunk_id, chunk in self.chunks.items():
                chunks.append(" ".join([x for x in chunk['propositions']]))
            return chunks

    def pretty_print_chunks(self):
        print(f"\nYou have {len(self.chunks)} chunks\n")
        for chunk_id, chunk in self.chunks.items():
            print(f"Chunk #{chunk['chunk_index']}")
            print(f"Chunk ID: {chunk_id}")
            print(f"Summary: {chunk['summary']}")
            print(f"Propositions:")
            for prop in chunk['propositions']:
                print(f"    -{prop}")
            print("\n\n")

    def pretty_print_chunk_outline(self):
        print("Chunk Outline\n")
        print(self.get_chunk_outline())

        
        
        
        
        
        
        
if __name__ == "__main__":
    # Set multiple PDF file paths
    pdf_paths = ["Contents/eda_ceds_guidelines_2023.pdf"]

    # Load PDF files
    documents = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        documents.extend(loader.load())

    # Extract text from documents
    texts = []
    for doc in documents:
        texts.append(doc.page_content)

    # Combine all texts into a single string
    full_text = "\n".join(texts)

    # Split into sentences
    sentences = sent_tokenize(full_text)

    # Initialize AgenticChunker
    ac = AgenticChunker()

    # Add sentences to AgenticChunker
    ac.add_propositions(sentences)

    # Pretty print chunks (optional)
    ac.pretty_print_chunks()

    # Get chunks as a list of strings
    chunks = ac.get_chunks(get_type='list_of_strings')

    # Create Document objects from chunks
    chunk_documents = [Document(page_content=chunk) for chunk in chunks]

    # Initialize embedding model
    embeddings = HuggingFaceEmbeddings()

    # Create and index Chroma vector store
    vectorstore = Chroma.from_documents(chunk_documents, embeddings)

    # Create retriever from vector store
    retriever = vectorstore.as_retriever()

    # Initialize LLM model
    api_key = os.getenv("MISTRAL_API_KEY")
    llm = ChatMistralAI(api_key=api_key)

    # Create RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    # Define query for summary
    query = "Summarize this document in about 5000 characters"

    # Get summary result
    summary = qa_chain.run(query)
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

# 예시 실행
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

# In[10]:


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

# 예시 실행
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

# In[11]:


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

# In[12]:


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

