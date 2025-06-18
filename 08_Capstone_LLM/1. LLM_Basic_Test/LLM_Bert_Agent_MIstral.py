#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import langchain as lc
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.agents import create_react_agent, Tool, AgentExecutor
from dotenv import load_dotenv
from httpx import HTTPStatusError

# Set PDF file paths and load them
pdf_paths = ["Contents/eda_ceds_guidelines_2023.pdf"]
documents = []
for path in pdf_paths:
    loader = PyPDFLoader(path)
    documents.extend(loader.load())

# Split text
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# Initialize embedding model
embeddings = HuggingFaceEmbeddings()
vectorstore = Chroma.from_documents(texts, embeddings)
retriever = vectorstore.as_retriever()

# Initialize LLM model
load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")
llm = ChatMistralAI(api_key=api_key)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Set up relationship extraction model
tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Initialize knowledge graph (in dictionary format)
knowledge_graph = {}

# Entity and relationship extraction function
def extract_entities_and_relations(text):
    doc = nlp(text)
    entities = [(ent['word'], ent['entity_group']) for ent in doc]
    
    # Relationship extraction example - arbitrarily creating relationships (can be replaced with an actual relationship extraction model)
    relations = []
    for i in range(len(entities) - 1):
        relations.append((entities[i], "related_to", entities[i + 1]))

    return entities, relations

# Function to add entities and relationships to the knowledge graph
def add_to_knowledge_graph(entities, relations):
    for entity, entity_type in entities:
        if entity not in knowledge_graph:
            knowledge_graph[entity] = {"type": entity_type, "relations": []}

    for head, relation, tail in relations:
        head_entity, head_type = head
        tail_entity, tail_type = tail
        if head_entity in knowledge_graph:
            knowledge_graph[head_entity]["relations"].append((relation, tail_entity))

# Extract entities and relationships from documents and add them to the knowledge graph
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

# Define Tool for Agent configuration
def retrieve_document(query):
    return enhanced_qa(query)

document_retrieval_tool = Tool(
    name="document_retriever",
    func=retrieve_document,
    description="Retrieve information from documents using QA with knowledge graph context."
)

# Define Agent Prompt
agent_prompt = PromptTemplate(
    input_variables=["query", "tools", "agent_scratchpad"],
    template="""
    You are an assistant with access to both document retrieval and knowledge graph tools.

    Use the following tools as needed: {tools}

    Format your response in this way:
    Thought: Describe your reasoning for using certain tools.
    Action: Choose one of [{tool_names}] based on the query and available context.
    Action Input: Provide the input for the action tool.
    Observation: Record the result of the action.

    Complete and concise response:
    Thought: {agent_scratchpad}
    Query: {query}
    """
)

# Initialize Agent
tools = [document_retrieval_tool]

agent = create_react_agent(
    tools=tools,
    llm=llm,
    prompt=agent_prompt
)

# Set up AgentExecutor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=5)

# Apply delay and retry for invocation
def invoke_with_delay(executor, input_data, delay=2):
    time.sleep(delay)
    return executor.invoke(input_data)

# Query function using Agent (with exception handling)
def agentic_query(query):
    input_data = {
        "query": query,
        "agent_scratchpad": "",
    }
    try:
        # Handle delay and retries
        return invoke_with_delay(agent_executor, input_data)
    except HTTPStatusError as e:
        if e.response.status_code == 429:
            print("Rate limit exceeded. Retrying after delay...")
            time.sleep(5)  # Set retry delay
            return invoke_with_delay(agent_executor, input_data)
        else:
            raise e

# Example query execution
query = "Summarize this document in about 5000 characters"
result = agentic_query(query)
print(result)

