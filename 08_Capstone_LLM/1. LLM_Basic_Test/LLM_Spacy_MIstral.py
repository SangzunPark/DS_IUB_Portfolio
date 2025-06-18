#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import spacy
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# Load PDF files
pdf_paths = ["Contents/eda_ceds_guidelines_2023.pdf"]
documents = []
for path in pdf_paths:
    loader = PyPDFLoader(path)
    documents.extend(loader.load())

# Split text
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# Initialize embedding model and create vector store
embeddings = HuggingFaceEmbeddings()
vectorstore = Chroma.from_documents(texts, embeddings)
retriever = vectorstore.as_retriever()

# Initialize LLM model
load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")
llm = ChatMistralAI(api_key=api_key)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Initialize knowledge graph (in dictionary format)
knowledge_graph = {}

# Entity and relation extraction function
def extract_entities_and_relations(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    
    # Relation extraction example - store simple relations (relation inference possible with SpaCy dependency parsing)
    relations = []
    for token in doc:
        if token.dep_ in {"nsubj", "dobj"} and token.head.dep_ == "ROOT":
            subject = token
            verb = token.head
            for child in token.children:
                if child.dep_ == "prep":
                    relations.append((subject.text, verb.lemma_, child.text))
    return entities, relations

# Function to add entities and relations to the knowledge graph
def add_to_knowledge_graph(entities, relations):
    for entity, entity_type in entities:
        if entity not in knowledge_graph:
            knowledge_graph[entity] = {"type": entity_type, "relations": []}

    for head_entity, relation, tail_entity in relations:
        if head_entity in knowledge_graph:
            knowledge_graph[head_entity]["relations"].append((relation, tail_entity))

# Extract entities and relations from documents and add them to the knowledge graph
for text in texts:
    entities, relations = extract_entities_and_relations(text.page_content)
    add_to_knowledge_graph(entities, relations)

# Knowledge graph query function
def query_knowledge_graph(query_text):
    results = []
    for entity, data in knowledge_graph.items():
        if query_text.lower() in entity.lower():
            results.append((entity, data["type"], data["relations"]))
    return results

# Enhance QA responses by referencing information from the knowledge graph
def enhanced_qa(query):
    graph_data = query_knowledge_graph(query)
    graph_context = " ".join([f"{entity} ({entity_type}) related to {[rel[1] for rel in relations]}" 
                              for entity, entity_type, relations in graph_data])
    combined_query = f"{query} Context from knowledge graph: {graph_context}"
    
    # Execute the original qa_chain
    return qa_chain.run(combined_query)

# Example query execution
query = "Summarize this document in about 5000 characters"
summary = enhanced_qa(query)
print(summary)


# In[ ]:


print(knowledge_graph)

