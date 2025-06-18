#!/usr/bin/env python
# coding: utf-8

# # C h u n k i n g 
# Nov 2024
# #### SJ.Park

# ## 1. Chonkie 
# ### Token Chunker
# https://github.com/bhavnicksm/chonkie

# In[1]:


from chonkie import TokenChunker
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_pretrained("gpt2")
chunker = TokenChunker(tokenizer)

text = "Woah! Chonkie, the chunking library is so cool! I love the tiny hippo hehe."
chunks = chunker(text)

for chunk in chunks:
    print(f"Chunk: {chunk.text}")
    print(f"Tokens: {chunk.token_count}")


# ### Result Description:
# 
# Copy Code
# "Woah! Chonkie, the chunking library is so cool! I love the tiny hippo hehe."
# This text undergoes the chunking process by the chunker.
# Role of the Tokenizer:
# 
# The pre-trained GPT-2 tokenizer splits the text into smaller units (tokens).
# For example, the text "Woah! Chonkie" can be split into the following tokens:
# plaintext
# Copy Code
# ['Woah', '!', 'ĠChonkie', ',', 'Ġthe', 'Ġchunking', ...]
# (Here, Ġ represents a space.)
# 
# Role of the TokenChunker:
# 
# The TokenChunker groups the tokens split by the tokenizer into chunks.
# The above result shows that the entire text is grouped into a single chunk.
# 
# Result Interpretation:
# 
# Chunk: Outputs the text chunk itself.
# The result shows that the input text is processed as a single chunk.
# Tokens: Outputs the number of tokens within the chunk.
# It indicates that the input text is split into 24 tokens.
# 
# Why is it processed as a single chunk?
# By default, the TokenChunker sets criteria for dividing the text into chunks. These criteria can be based on a fixed number of tokens, words, sentences, etc.
# Since there was no explicit setting for chunk size in the code, the entire text appears to have been processed as a single chunk.
# For instance, if the chunk size had been set, the text might have been divided into smaller chunks.

# In[2]:


chunker = TokenChunker(tokenizer, chunk_size=10, chunk_overlap=3) 
chunks = chunker(text)

for chunk in chunks:
    print(f"Chunk: {chunk.text}")
    print(f"Tokens: {chunk.token_count}")


# "Chunk: Woah! Chonkie, the chunking" is considered to have 10 tokens because of the way the GPT-2 tokenizer splits text. This is not simply a word count; instead, it uses the Byte-Pair Encoding (BPE) method to process the text, breaking it down into smaller units (tokens).
# 
# What is a Token?
# A token is a smaller unit that makes up a text.
# The GPT-2 tokenizer treats not only words but also subwords, characters, and even spaces as tokens.
# Therefore, text like "Woah! Chonkie, the chunking," which appears to consist of 5 words, can actually be split into 10 tokens.

# In[3]:


from transformers import GPT2Tokenizer

# Initialize the GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Text
text = "Woah! Chonkie, the chunking"

# Tokenize using the tokenizer
tokens = tokenizer.tokenize(text)
print("Tokens:", tokens)
print("Token Count:", len(tokens))


# #### Why Did This Result Occur?
# Subword Segmentation:
# The GPT-2 tokenizer divides long or complex words into smaller units called subwords.
# 
# Example: "Woah" → ['Wo', 'ah']
# Separation of Special Characters:
# Special characters such as ! and , are treated as independent tokens.
# 
# Example: "!" → ['!'], "," → [',']
# Prefix Ġ Representing Spaces:
# The GPT-2 tokenizer uses the prefix Ġ to represent spaces preceding words.
# 
# Example: " the" → ['Ġthe'], " chunk" → ['Ġchunk']
# Subword Splitting Patterns:
# "Chonkie" → ['ĠCh', 'on', 'kie']
# The word "Chonkie" is split into Ch, on, and kie at the subword level.
# "chunking" → ['Ġchunk', 'ing']
# Similarly, "chunking" is divided into chunk and ing.
# This segmentation occurs because the Byte-Pair Encoding (BPE) method used by GPT-2's tokenizer optimizes for a balance between frequent word parts and flexibility to handle rare words.

# In[6]:


# Convert tokens to IDs
token_ids = tokenizer.convert_tokens_to_ids(tokens)
print("Token IDs:", token_ids)


# #### Role of Token IDs: Numeric Representation of Text
# 
# Computers and deep learning models cannot directly understand text data, so it must be converted into numbers (IDs).
# For example, "Woah" is converted to 15432, and "!" is converted to 50256.
# 
# Vocabulary-Based Mapping
# 
# The tokenizer uses a **vocabulary**, a dictionary that defines a unique numeric ID for every possible token.
# Example: "Woah" might be the 15432nd entry in the vocabulary.
# 
# Preparing Data for Model Input
# 
# To input text data into a model, it must be converted into an array of token IDs.
# Example: The text "Woah! Chonkie" → [15432, 50256, 31289]
# 
# The GPT-2 vocabulary size is approximately 50,257, meaning each token ID is a value between 0 and 50,256.
# 
# Vocabulary Sizes of Major Models
# 
# #### GPT Series (OpenAI)
# - **GPT-2**: ~50,257 tokens  
#   Uses the Byte Pair Encoding (BPE) method.
# - **GPT-3**: ~50,257 tokens  
#   Uses the same tokenizer as GPT-2.
# #### BERT Series (Google)
# - **BERT-base/large**: ~30,522 tokens  
#   Uses the WordPiece tokenizer.
# - **ALBERT**: ~30,000 tokens  
#   A simplified version of BERT's WordPiece tokenizer.
# #### T5 (Google)
# - **T5** (Text-to-Text Transfer Transformer): ~32,000 tokens  
#   Uses the SentencePiece tokenizer.
# #### RoBERTa (Facebook)
# - **RoBERTa**: ~50,265 tokens  
#   Uses the BPE method similar to GPT.
# #### DistilBERT (Hugging Face)
# - **DistilBERT**: ~30,522 tokens  
#   Uses BERT's WordPiece tokenizer.
# #### XLNet (Google & CMU)
# - **XLNet-base/large**: ~32,000 tokens  
#   Uses the SentencePiece tokenizer.
# #### Bloom (BigScience)
# - **Bloom**: ~250,680 tokens  
#   A multilingual model covering a wide range of languages.
# #### LLaMA (Meta)
# - **LLaMA**: ~32,000 tokens  
#   Based on SentencePiece with a BPE-style approach.
# #### CLIP (OpenAI)
# - **CLIP**: ~49,408 tokens  
#   A model for combining text and image inputs, uses the BPE method.

# #### Relationship Between Vocabulary Size and Model Performance
# 
# #### When the Vocabulary Size is Large:
# 
# **Advantages:**
# - Better capability to handle diverse languages, syntax, and words.
# - Increased expressiveness, including rare words and special characters.
# 
# **Disadvantages:**
# - Requires more memory and computation during model training and inference.
# - If the vocabulary is excessively large, sparsity issues may arise.
# 
# #### When the Vocabulary Size is Small:
# 
# **Advantages:**
# - Higher computational efficiency and reduced memory usage.
# - Generally leads to faster training speeds.
# 
# **Disadvantages:**
# - May fail to represent rare words effectively.
# - Some words might be overly fragmented into subwords.
# 

# # 2. Chunking by Each Tool
# 
# **TokenChunker:** Splits text into chunks of a fixed token size.
# 
# **WordChunker:** Splits text into chunks based on words.
# 
# **SentenceChunker:** Splits text into chunks based on sentences.
# 
# **SemanticChunker:** Splits text into chunks based on semantic similarity.
# 
# **SDPMChunker:** Splits text into chunks using the Semantic Double-Pass Merge method.
# 

# In[9]:


for chunk in chunks:
    print(dir(chunk))


# ## 1. WordChunker

# In[10]:


from chonkie import WordChunker

# Initialize WordChunker
chunker = WordChunker(chunk_size=5, chunk_overlap=2)

# Input text
text = "Woah! Chonkie, the chunking library is so cool! I love the tiny hippo hehe."

# Generate chunks
chunks = chunker(text)

# Print results
for chunk in chunks:
    print(f"Chunk: {chunk.text}")
    print(f"Token Count: {chunk.token_count}")  # Using the token_count attribute


# ### Behavior Based on Chunking Settings
# 
# **chunk_size=5:**
# - Each chunk can contain up to 5 words.
# - If fewer than 5 words remain at the end of the text, a chunk with fewer words is created.
# 
# **chunk_overlap=2:**
# - There is an overlap of 2 words between adjacent chunks.
# - Example:
#   - Chunk 2: "Chonkie, the"
#   - Chunk 3: "the chunking library is"
# 
# **Difference Between Token Count and Word Count:**
# - The number of tokens in the text can differ from the number of words due to special characters and spaces.
# - For example, `"Woah!"` is split into **[Wo, ah, !]**, resulting in 3 tokens.
# 

# ## 2. SentenceChunker

# In[11]:


import json
from chonkie import SentenceChunker
from autotiktokenizer import AutoTikTokenizer

# Initialize the tokenizer
AutoTikTokenizer._read_json = lambda self, path: json.load(open(path, 'r', encoding='utf-8'))

# Initialize GPT-2 tokenizer
tokenizer = AutoTikTokenizer.from_pretrained("gpt2")

# Initialize SentenceChunker
chunker = SentenceChunker(
    tokenizer=tokenizer,
    chunk_size=512,
    chunk_overlap=128,
    min_sentences_per_chunk=1
)

# Input text
text ="""
    Artificial intelligence is transforming industries. 
    It is widely used in healthcare, finance, and technology. 
    However, ethical concerns remain a challenge. 
    Proper regulation and transparency are necessary for its growth.
    """.replace("\n", " ")

# Generate chunks
chunks = chunker.chunk(text)

# Print results
for idx, chunk in enumerate(chunks):
    print(f"Chunk {idx + 1}: {chunk.text}")
    print(f"Number of sentences: {len(chunk.sentences)}")


# In[12]:


chunker = SentenceChunker(
    tokenizer=tokenizer,
    chunk_size=20,  # smaller chunk size
    chunk_overlap=5,
    min_sentences_per_chunk=1
)

chunks = chunker.chunk(text)

for idx, chunk in enumerate(chunks):
    print(f"Chunk {idx + 1}: {chunk.text}")
    print(f"Number of sentences: {len(chunk.sentences)}")

text = """
    Artificial intelligence is transforming industries. 
    It is widely used in healthcare, finance, and technology. 
    However, ethical concerns remain a challenge. 
    Proper regulation and transparency are necessary for its growth. 
    AI models like GPT are helping researchers worldwide. 
    Still, challenges in scalability and resource allocation persist.
""".replace("\n", " ")

chunks = chunker.chunk(text)

for idx, chunk in enumerate(chunks):
    print(f"Chunk {idx + 1}: {chunk.text}")
    print(f"Number of sentences: {len(chunk.sentences)}")


# ## 3. SemanticChunker
# 
# **SemanticChunker** is a tool that divides text based on syntactic and semantic criteria. While traditional chunking tools split text based on simple metrics such as word count, sentence count, or token count, SemanticChunker analyzes the meaning and context of the text to segment it into appropriate units. This tool is particularly useful for dividing long documents into sections based on topics.
# 

# In[13]:


from chonkie import SemanticChunker

# Initialize SemanticChunker
chunker = SemanticChunker(
    embedding_model="all-mpnet-base-v2",  # Embedding model
    max_chunk_size=512,                  # Maximum tokens per chunk
    similarity_threshold=0.4,            # Sentence similarity threshold
    initial_sentences=2                  # Minimum 2 sentences per chunk
)

# Input text
text = """
Artificial intelligence (AI) is transforming industries and reshaping the way we work.
In healthcare, AI is being used for diagnostic purposes and to develop personalized treatment plans.
In finance, it has revolutionized fraud detection and automated trading systems.
However, despite its many advantages, AI also poses significant ethical concerns.
Issues such as data privacy, algorithmic bias, and job displacement are at the forefront of discussions.
Proper regulation and transparency are essential to address these challenges.
On the other hand, AI models like GPT are enabling researchers and developers to build innovative solutions.
These models can generate human-like text, translate languages, and even create art.
Still, scalability and resource allocation remain persistent challenges.
Addressing these issues will require collaboration between governments, industries, and academic institutions.
""".replace("\n", " ")  

# Generate chunks
chunks = chunker.chunk(text)

# Print results
for idx, chunk in enumerate(chunks):
    print(f"Chunk {idx + 1}: {chunk.text}")
    print(f"Number of semantic sentences: {len(chunk.sentences)}")


# | Model                           | Speed         | Accuracy     | Use Cases                         |
# |---------------------------------|---------------|--------------|-----------------------------------|
# | `all-minilm-l6-v2`              | Very Fast     | Moderate     | General document processing, fast similarity calculations |
# | `all-mpnet-base-v2`             | Moderate      | High         | High-quality document chunking and analysis |
# | `multi-qa-mpnet-base-dot-v1`    | Moderate      | High         | Search, QA systems                |
# | `paraphrase-mpnet-base-v2`      | Moderate      | Very High    | Sentence rephrasing, semantic similarity analysis |
# | `openai/text-embedding-ada-002` | Slow (API-based)| Very High   | Precision analysis, large-scale tasks |
# | `all-distilroberta-v1`          | Very Fast     | High         | Multipurpose embedding generation |
# 

# ## 4. SDPMChunker: Semantic Double-Pass Merge

# In[14]:


from chonkie import SDPMChunker

# Initialize SDPMChunker
chunker = SDPMChunker(
    embedding_model="all-mpnet-base-v2",  # Embedding model
    max_chunk_size=512,                  # Maximum tokens per chunk
    similarity_threshold=0.4,            # Similarity threshold
    skip_window=2                         # Allow non-contiguous chunk merging
)

# Input text
text = """
Artificial intelligence (AI) is transforming industries and reshaping the way we work.
In healthcare, AI is being used for diagnostic purposes and to develop personalized treatment plans.
In finance, it has revolutionized fraud detection and automated trading systems.
However, despite its many advantages, AI also poses significant ethical concerns.
Proper regulation and transparency are essential to address these challenges.
On the other hand, AI models like GPT are enabling researchers and developers to build innovative solutions.
These models can generate human-like text, translate languages, and even create art.
Still, scalability and resource allocation remain persistent challenges.
Collaboration between governments, industries, and academic institutions is essential to address these issues.
Proper regulation will be key in ensuring that the benefits of AI outweigh its risks in the long term.
"""

# Generate chunks
chunks = chunker.chunk(text)

# Print results
for idx, chunk in enumerate(chunks):
    print(f"Chunk {idx + 1}: {chunk.text}")
    print(f"Number of semantic sentences: {len(chunk.sentences)}")


# ### Use of `skip_window`
# 
# When `skip_window=n` is set, the similarity is evaluated by skipping up to `n` chunks from the current chunk.  
# For example, with `skip_window=2`:
# - Sentence 1 is compared for similarity with Sentence 2 and Sentence 3.
# - Sentence 2 is compared for similarity with Sentence 3 and Sentence 4.
# 
# Even if there are non-contiguous relationships, chunks can be merged if they are semantically related.
# 

# ### SDPMChunker vs SemanticChunker
# 
# | **Feature**          | **SDPMChunker**                                     | **SemanticChunker**                           |
# |-----------------------|----------------------------------------------------|-----------------------------------------------|
# | **Operation**         | Semantic Double-Pass Merge: Proceeds in two stages, merges even non-contiguous paragraphs | Single-pass, evaluates similarity only between adjacent sentences or paragraphs |
# | **Merging Scope**     | Can merge non-contiguous chunks if similarity is detected | Focuses only on similarity between adjacent chunks |
# | **`skip_window` Usage** | Allows specifying the range of non-contiguous chunks to search during merging | Not supported                                 |
# | **Application**       | Suitable for long documents with scattered topics  | Best for short or structured text with logical connections between sentences |
# | **Merging Method**    | Re-evaluates semantically related paragraphs before merging | Does not modify results after initial evaluation |
# | **Complexity**        | More complex due to two rounds of similarity evaluation | Suitable for simpler chunking tasks           |
# 
