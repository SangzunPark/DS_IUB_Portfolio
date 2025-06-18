#!/usr/bin/env python
# coding: utf-8

# ## Overview:
# My mini-project involves crawling and extracting data from the BBC News website for analysis. I selected the BBC News website because it is a webpage that can be adequately crawled using Beautiful Soup and regularly updates a diverse range of text data. Lastly, the site lacks its own statistical analysis UI, allowing me to explore various analyses without imitation.
# 
# Articles on the BBC webpage are freely available for the latest articles. Since these recent articles are limited to about 12 pages per continent, I selected news from Europe, Africa, Asia, and North America. This results in approximately 48 pages of data, with each page containing about 9-10 articles.
# 
# Using this data, I conducted sentiment analysis, named entity recognition (NER), and noun word analysis on the articles. Each analysis utilized pie charts, bar charts, and word cloud visualizations.
# 
# Specific Libraries Used:
# textblob (not used due to performance issues): A simple library for text data sentiment analysis and language processing.
# vaderSentiment: A sentiment analysis tool specialized for social media text.
# seaborn: A Python library for statistical data visualization.
# spacy: An industrial-strength Python library for fast natural language processing.
# WordCloud: A visualization tool for generating word clouds from text data.
# 
# ## Code Explanation:
# 
# #### Data Extraction:
# 
# Step 1:
# Use Beautiful Soup to perform the first extraction from the BBC homepage. This initial extraction aims to collect the title, link, and publication date of all articles in the news section. After extracting all pages, I used the find method within the code to extract specific elements. These elements were stored in the news_data list. The process was repeated using pagination code, and the final data was converted into a DataFrame and saved as a CSV file.
# 
# Step 2:
# Using the CSV file saved in Step 1, this code aims to collect the actual article content for each article. Using the saved links from Step 1, I accessed each link and used the get_article_details function to extract detailed text from the articles. The extracted content was then converted into a DataFrame and saved as a CSV file.
# 
# #### Data Analysis:
# Step 1:
# The first step in data analysis is sentiment analysis and visualization. I initially tried using textblob, but due to its low performance, I opted for vaderSentiment and used threshold values determined through experimentation. The analyze_sentiment function uses vaderSentiment, and I used the file saved from Step 2 of data extraction along with this function to produce the results. The resulting values were calculated as ratios and visualized using an sns pie chart.
# 
# Step 2:
# First, I loaded the spacy library, which can perform tokenization and part-of-speech tagging for English models, and stored it in the nlp variable. I created the extract_countries function, where 'GPE' is a label representing countries, cities, and locations. Finally, using a for loop, I applied this function to the data in the loaded CSV file, produced the results, and visualized the top 7 countries using an sns bar chart.
# 
# Step 3:
# Similar to Step 2, I loaded the spacy object and set the token_pos value to "NOUN" to extract nouns from the text, with the noun length restricted to 3 characters or more. Using the Python standard library Counter, I calculated the frequency. To aggregate the data, I calculated the ratio of each word, extracted the top 50 words, and stored them as a dictionary. Finally, I visualized the results using the WordCloud library.
# 
# ## Results:
# The visualizations created effectively explain the current topics of interest for each continent. For example, France is currently in a state of confusion due to the elections. On the day the data was extracted from the BBC, the left and centrist parties overturned the right-wing party in the second election in France. This was a significant event not only in France but throughout Europe, and it was covered extensively by the media. The named entity recognition (NER) visualization also reflected this well, with France and Paris being prominently featured. Sentiment analysis of the articles showed that Europe had a high proportion of positive sentiments. This can be interpreted as the BBC articles covering the French issue positively. The most dominant nouns were "people," "election," and "alliance," which also reflect the current situation in Europe. Other continents also had notable data reflecting the current issues, and the word cloud analysis for all continents showed that "People" was the most significant issue globally. I believe the most important thing is always the people themselves.

# # Europe

# In[41]:


import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime

# BBC URL
base_url = "https://www.bbc.com/news/world/europe"

def get_articles_from_page(url):
    response = requests.get(url)
    response.raise_for_status() 
    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract articles
    articles = soup.find_all('div', {'data-testid': 'liverpool-card'})
    news_data = []

    for article in articles:
        title_tag = article.find('h2', {'data-testid': 'card-headline'})
        link_tag = article.find('a', {'data-testid': 'internal-link'})
        date_tag = article.find('span', {'data-testid': 'card-metadata-lastupdated'})

        if title_tag and link_tag: # extract articles(url)
            title = title_tag.text.strip()
            link = "https://www.bbc.com" + link_tag['href']
            date = date_tag.text.strip() if date_tag else None
            
            news_data.append({'title': title, 'link': link, 'date': date})

    return news_data

# pagination
news_data = []
for page in range(1, 13):  # the maximum page number is 12 in the web  
    page_url = base_url + f"?page={page}"
    news_data.extend(get_articles_from_page(page_url))

# to the dataframe
df = pd.DataFrame(news_data)
df.to_csv('bbc_europe.csv', index=False)

print("First phase completed")


# In[5]:


import requests
from bs4 import BeautifulSoup
import pandas as pd

# load data from the previous CSV file
df = pd.read_csv('bbc_europe.csv')

# article body extracting function
def get_article_details(url):
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, 'html.parser')

    paragraphs = soup.find_all('div', {'data-component': 'text-block'})
    body = ' '.join(p.get_text(separator=' ', strip=True) for p in paragraphs)
    
    return body

# save the body
detailed_news_data = []

for index, row in df.iterrows():
    try:
        body = get_article_details(row['link'])
        detailed_news_data.append({
            'title': row['title'],
            'link': row['link'],
            'date': row['date'],
            'body': body
        })
    except Exception as e:
        print(f"Error processing article at {row['link']}: {e}")

# dataframe
detailed_df = pd.DataFrame(detailed_news_data)
detailed_df.to_csv('bbc_detailed_europe.csv', index=False)

print("Second phase completed")


# In[16]:


# Data analysis & Viz 1

from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd

# load the detailed CSV file
df = pd.read_csv('bbc_detailed_europe.csv')

# VADER Analyzer
analyzer = SentimentIntensityAnalyzer()

# Sentiment analysis function
def analyze_sentiment(text):
    scores = analyzer.polarity_scores(text)
    if scores['compound'] > 0.2:
        return 'Positive'
    elif scores['compound'] < -0.2:
        return 'Negative'
    else:
        return 'Neutral'

df['body'] = df['body'].fillna('')    
    
# Perform sentiment analysis on each article body
df['sentiment'] = df['body'].apply(analyze_sentiment)

print("Articles and sentiment analysis results:")
print(df[['title', 'date', 'sentiment']])

# Calculate sentiment analysis result ratio
sentiment_counts = df['sentiment'].value_counts(normalize=True)

# color change
pastel_colors = sns.color_palette("pastel")

# vizualiztion pie chart
plt.figure(figsize=(8, 8))
plt.pie(sentiment_counts, labels=sentiment_counts.index, colors=pastel_colors, autopct='%1.1f%%', startangle=140)
plt.title('Sentiment Analysis of BBC Europe News Articles - July 8 2024')
plt.axis('equal')  # keep circle 
plt.show()


# In[17]:


# Data analysis & Viz 2

import spacy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# spaCy English model
nlp = spacy.load('en_core_web_sm')

# load the detailed CSV file
df = pd.read_csv('bbc_detailed_europe.csv')
df['body'] = df['body'].fillna('')

# Extract country name 
def extract_countries(text):
    doc = nlp(text)
    countries = [ent.text for ent in doc.ents if ent.label_ == 'GPE']  
    return countries

all_countries = []

for index, row in df.iterrows():
    try:
        countries = extract_countries(row['body'])
        all_countries.extend(countries)
    except Exception as e:
        print(f"Error processing article at {row['link']}: {e}")

# Ratio
country_counts = pd.Series(all_countries).value_counts()

print("Countries most mentioned in articles:")
print(country_counts.head(7))

top_countries = country_counts.head(7)

plt.figure(figsize=(12, 6))
sns.barplot(x=top_countries.values, y=top_countries.index, palette='pastel')
plt.title('Top 7 Countries or cities Mentioned in BBC Europe News Articles - July 8 2024')
plt.xlabel('Mention Count')
plt.ylabel('Country')
plt.show()


# In[5]:


# Data analysis & Viz 3

from collections import Counter
import spacy
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd

# spaCy English model
nlp = spacy.load('en_core_web_sm')

# load the detailed CSV file
df = pd.read_csv('bbc_detailed_europe.csv')

# concatenate all articles
all_text = ' '.join(df['body'].dropna())

# Convert all text to spaCy object
doc = nlp(all_text)

# Extracts only nouns and includes only words with 3 or more letters
nouns = [token.text.lower() for token in doc if token.pos_ == 'NOUN' and len(token.text) >= 3]

# Word frequency calculation
word_counts = Counter(nouns)

# Ratio
total_nouns = sum(word_counts.values())
word_ratios = {word: count / total_nouns for word, count in word_counts.items()}

# Extract the top 50 most used words
most_common_words = word_counts.most_common(50)

# Rario on top 50
most_common_ratios = [(word, count, word_ratios[word]) for word, count in most_common_words]
word_freq = {word: count for word, count in most_common_words}

# Print Result
print("<Top 50 Most Dominant Nouns in Europe>")
print()
for word, count, ratio in most_common_ratios:
    print(f"{word}: {count}, Ratio: {ratio:.2%}")

# Word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Top 50 Most Dominant Nouns in Europe - July 8 2024')
plt.show()


# # Asia

# In[17]:


# BBC URL
base_url = "https://www.bbc.com/news/world/asia"

def get_articles_from_page(url):
    response = requests.get(url)
    response.raise_for_status()  # Check if the request was successful
    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract articles
    articles = soup.find_all('div', {'data-testid': 'liverpool-card'})
    news_data = []

    for article in articles:
        title_tag = article.find('h2', {'data-testid': 'card-headline'})
        link_tag = article.find('a', {'data-testid': 'internal-link'})
        date_tag = article.find('span', {'data-testid': 'card-metadata-lastupdated'})

        if title_tag and link_tag: # extract articles(url)
            title = title_tag.text.strip()
            link = "https://www.bbc.com" + link_tag['href']
            date = date_tag.text.strip() if date_tag else None
            
            news_data.append({'title': title, 'link': link, 'date': date})

    return news_data

# pagination
news_data = []
for page in range(1, 13):  # the maximum page number is 12 in the web  
    page_url = base_url + f"?page={page}"
    news_data.extend(get_articles_from_page(page_url))

# to the dataframe
df = pd.DataFrame(news_data)
df.to_csv('bbc_asia.csv', index=False)

print("First phase completed")


# In[7]:


# load data from the previous CSV file
df = pd.read_csv('bbc_asia.csv')

# article body extracting function
def get_article_details(url):
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, 'html.parser')

    paragraphs = soup.find_all('div', {'data-component': 'text-block'})
    body = ' '.join(p.get_text(separator=' ', strip=True) for p in paragraphs)
    
    return body

# save the body
detailed_news_data = []

for index, row in df.iterrows():
    try:
        body = get_article_details(row['link'])
        detailed_news_data.append({
            'title': row['title'],
            'link': row['link'],
            'date': row['date'],
            'body': body
        })
    except Exception as e:
        print(f"Error processing article at {row['link']}: {e}")

# dataframe
detailed_df = pd.DataFrame(detailed_news_data)
detailed_df.to_csv('bbc_detailed_asia.csv', index=False)

print("Second phase completed")


# In[6]:


from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd

# load the detailed CSV file
df = pd.read_csv('bbc_detailed_asia.csv')

# VADER Analyzer
analyzer = SentimentIntensityAnalyzer()

# Sentiment analysis function
def analyze_sentiment(text):
    scores = analyzer.polarity_scores(text)
    if scores['compound'] > 0.2:
        return 'Positive'
    elif scores['compound'] < -0.2:
        return 'Negative'
    else:
        return 'Neutral'

df['body'] = df['body'].fillna('')    
    
# Perform sentiment analysis on each article body
df['sentiment'] = df['body'].apply(analyze_sentiment)

print("Articles and sentiment analysis results:")
print(df[['title', 'date', 'sentiment']])

# Calculate sentiment analysis result ratio
sentiment_counts = df['sentiment'].value_counts(normalize=True)

# color change
pastel_colors = sns.color_palette("pastel")

# vizualiztion pie chart
plt.figure(figsize=(8, 8))
plt.pie(sentiment_counts, labels=sentiment_counts.index, colors=pastel_colors, autopct='%1.1f%%', startangle=140)
plt.title('Sentiment Analysis of BBC Asia News Articles - July 8 2024')
plt.axis('equal')  # keep circle 
plt.show()


# In[7]:


import spacy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# spaCy English model
nlp = spacy.load('en_core_web_sm')

# load the detailed CSV file
df = pd.read_csv('bbc_detailed_asia.csv')
df['body'] = df['body'].fillna('')

# Extract country name 
def extract_countries(text):
    doc = nlp(text)
    countries = [ent.text for ent in doc.ents if ent.label_ == 'GPE']  
    return countries

all_countries = []

for index, row in df.iterrows():
    try:
        countries = extract_countries(row['body'])
        all_countries.extend(countries)
    except Exception as e:
        print(f"Error processing article at {row['link']}: {e}")

# Ratio
country_counts = pd.Series(all_countries).value_counts()

print("Countries most mentioned in articles:")
print(country_counts.head(7))

top_countries = country_counts.head(7)

plt.figure(figsize=(12, 6))
sns.barplot(x=top_countries.values, y=top_countries.index, palette='pastel')
plt.title('Top 7 Countries or cities Mentioned in BBC Asia News Articles - July 8 2024')
plt.xlabel('Mention Count')
plt.ylabel('Country')
plt.show()


# In[8]:


from collections import Counter
import spacy
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd

# spaCy English model
nlp = spacy.load('en_core_web_sm')

# load the detailed CSV file
df = pd.read_csv('bbc_detailed_asia.csv')

# concatenate all articles
all_text = ' '.join(df['body'].dropna())

# Convert all text to spaCy object
doc = nlp(all_text)

# Extracts only nouns and includes only words with 3 or more letters
nouns = [token.text.lower() for token in doc if token.pos_ == 'NOUN' and len(token.text) >= 3]

# Word frequency calculation
word_counts = Counter(nouns)

# Ratio
total_nouns = sum(word_counts.values())
word_ratios = {word: count / total_nouns for word, count in word_counts.items()}

# Extract the top 50 most used words
most_common_words = word_counts.most_common(50)

# Rario on top 50
most_common_ratios = [(word, count, word_ratios[word]) for word, count in most_common_words]
word_freq = {word: count for word, count in most_common_words}

# Print Result
print("<Top 50 Most Dominant Nouns in Asia>")
print()
for word, count, ratio in most_common_ratios:
    print(f"{word}: {count}, Ratio: {ratio:.2%}")

# Word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Top 50 Most Dominant Nouns in Asia - July 8 2024')
plt.show()


# # Africa

# In[72]:


import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime

# BBC URL
base_url = "https://www.bbc.com/news/world/africa"

def get_articles_from_page(url):
    response = requests.get(url)
    response.raise_for_status() 
    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract articles
    articles = soup.find_all('div', {'data-testid': 'liverpool-card'})
    news_data = []

    for article in articles:
        title_tag = article.find('h2', {'data-testid': 'card-headline'})
        link_tag = article.find('a', {'data-testid': 'internal-link'})
        date_tag = article.find('span', {'data-testid': 'card-metadata-lastupdated'})

        if title_tag and link_tag: # extract articles(url)
            title = title_tag.text.strip()
            link = "https://www.bbc.com" + link_tag['href']
            date = date_tag.text.strip() if date_tag else None
            
            news_data.append({'title': title, 'link': link, 'date': date})

    return news_data

# pagination
news_data = []
for page in range(1, 12):  # the maximum page number is 11 in the web  
    page_url = base_url + f"?page={page}"
    news_data.extend(get_articles_from_page(page_url))

# to the dataframe
df = pd.DataFrame(news_data)
df.to_csv('bbc_africa.csv', index=False)

print("First phase completed")


# In[9]:


# load data from the previous CSV file
df = pd.read_csv('bbc_africa.csv')

# article body extracting function
def get_article_details(url):
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, 'html.parser')

    paragraphs = soup.find_all('div', {'data-component': 'text-block'})
    body = ' '.join(p.get_text(separator=' ', strip=True) for p in paragraphs)
    
    return body

# save the body
detailed_news_data = []

for index, row in df.iterrows():
    try:
        body = get_article_details(row['link'])
        detailed_news_data.append({
            'title': row['title'],
            'link': row['link'],
            'date': row['date'],
            'body': body
        })
    except Exception as e:
        print(f"Error processing article at {row['link']}: {e}")

# dataframe
detailed_df = pd.DataFrame(detailed_news_data)
detailed_df.to_csv('bbc_detailed_africa.csv', index=False)

print("Second phase completed")


# In[9]:


from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd

# load the detailed CSV file
df = pd.read_csv('bbc_detailed_africa.csv')

# VADER Analyzer
analyzer = SentimentIntensityAnalyzer()

# Sentiment analysis function
def analyze_sentiment(text):
    scores = analyzer.polarity_scores(text)
    if scores['compound'] > 0.2:
        return 'Positive'
    elif scores['compound'] < -0.2:
        return 'Negative'
    else:
        return 'Neutral'

df['body'] = df['body'].fillna('')    
    
# Perform sentiment analysis on each article body
df['sentiment'] = df['body'].apply(analyze_sentiment)

print("Articles and sentiment analysis results:")
print(df[['title', 'date', 'sentiment']])

# Calculate sentiment analysis result ratio
sentiment_counts = df['sentiment'].value_counts(normalize=True)

# color change
pastel_colors = sns.color_palette("pastel")

# vizualiztion pie chart
plt.figure(figsize=(8, 8))
plt.pie(sentiment_counts, labels=sentiment_counts.index, colors=pastel_colors, autopct='%1.1f%%', startangle=140)
plt.title('Sentiment Analysis of BBC Africa News Articles - July 8 2024')
plt.axis('equal')  # keep circle 
plt.show()


# In[10]:


import spacy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# spaCy English model
nlp = spacy.load('en_core_web_sm')

# load the detailed CSV file
df = pd.read_csv('bbc_detailed_africa.csv')
df['body'] = df['body'].fillna('')

# Extract country name 
def extract_countries(text):
    doc = nlp(text)
    countries = [ent.text for ent in doc.ents if ent.label_ == 'GPE']  
    return countries

all_countries = []

for index, row in df.iterrows():
    try:
        countries = extract_countries(row['body'])
        all_countries.extend(countries)
    except Exception as e:
        print(f"Error processing article at {row['link']}: {e}")

# Ratio
country_counts = pd.Series(all_countries).value_counts()

print("Countries most mentioned in articles:")
print(country_counts.head(7))

top_countries = country_counts.head(7)

plt.figure(figsize=(12, 6))
sns.barplot(x=top_countries.values, y=top_countries.index, palette='pastel')
plt.title('Top 7 Countries or cities Mentioned in BBC Africa News Articles - July 8 2024')
plt.xlabel('Mention Count')
plt.ylabel('Country')
plt.show()


# In[11]:


from collections import Counter
import spacy
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd

# spaCy English model
nlp = spacy.load('en_core_web_sm')

# load the detailed CSV file
df = pd.read_csv('bbc_detailed_africa.csv')

# concatenate all articles
all_text = ' '.join(df['body'].dropna())

# Convert all text to spaCy object
doc = nlp(all_text)

# Extracts only nouns and includes only words with 3 or more letters
nouns = [token.text.lower() for token in doc if token.pos_ == 'NOUN' and len(token.text) >= 3]

# Word frequency calculation
word_counts = Counter(nouns)

# Ratio
total_nouns = sum(word_counts.values())
word_ratios = {word: count / total_nouns for word, count in word_counts.items()}

# Extract the top 50 most used words
most_common_words = word_counts.most_common(50)

# Rario on top 50
most_common_ratios = [(word, count, word_ratios[word]) for word, count in most_common_words]
word_freq = {word: count for word, count in most_common_words}

# Print Result
print("<Top 50 Most Dominant Nouns in Africa>")
print()
for word, count, ratio in most_common_ratios:
    print(f"{word}: {count}, Ratio: {ratio:.2%}")

# Word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Top 50 Most Dominant Nouns in Africa- July 8 2024')
plt.show()


# # USA & Canada

# In[23]:


# BBC URL
base_url = "https://www.bbc.com/news/us-canada"

def get_articles_from_page(url):
    response = requests.get(url)
    response.raise_for_status()  # Check if the request was successful
    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract articles
    articles = soup.find_all('div', {'data-testid': 'liverpool-card'})
    news_data = []

    for article in articles:
        title_tag = article.find('h2', {'data-testid': 'card-headline'})
        link_tag = article.find('a', {'data-testid': 'internal-link'})
        date_tag = article.find('span', {'data-testid': 'card-metadata-lastupdated'})

        if title_tag and link_tag: # extract articles(url)
            title = title_tag.text.strip()
            link = "https://www.bbc.com" + link_tag['href']
            date = date_tag.text.strip() if date_tag else None
            
            news_data.append({'title': title, 'link': link, 'date': date})

    return news_data

# pagination
news_data = []
for page in range(1, 13):  # the maximum page number is 12 in the web  
    page_url = base_url + f"?page={page}"
    news_data.extend(get_articles_from_page(page_url))

# to the dataframe
df = pd.DataFrame(news_data)
df.to_csv('bbc_us-canada.csv', index=False)

print("First phase completed")


# In[11]:


# load data from the previous CSV file
df = pd.read_csv('bbc_us-canada.csv')

# article body extracting function
def get_article_details(url):
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, 'html.parser')

    paragraphs = soup.find_all('div', {'data-component': 'text-block'})
    body = ' '.join(p.get_text(separator=' ', strip=True) for p in paragraphs)
    
    return body

# save the body
detailed_news_data = []

for index, row in df.iterrows():
    try:
        body = get_article_details(row['link'])
        detailed_news_data.append({
            'title': row['title'],
            'link': row['link'],
            'date': row['date'],
            'body': body
        })
    except Exception as e:
        print(f"Error processing article at {row['link']}: {e}")

# dataframe
detailed_df = pd.DataFrame(detailed_news_data)
detailed_df.to_csv('bbc_detailed_us-canada.csv', index=False)

print("Second phase completed")


# In[13]:


from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd

# load the detailed CSV file
df = pd.read_csv('bbc_detailed_us-canada.csv')

# VADER Analyzer
analyzer = SentimentIntensityAnalyzer()

# Sentiment analysis function
def analyze_sentiment(text):
    scores = analyzer.polarity_scores(text)
    if scores['compound'] > 0.2:
        return 'Positive'
    elif scores['compound'] < -0.2:
        return 'Negative'
    else:
        return 'Neutral'

df['body'] = df['body'].fillna('')    
    
# Perform sentiment analysis on each article body
df['sentiment'] = df['body'].apply(analyze_sentiment)

print("Articles and sentiment analysis results:")
print(df[['title', 'date', 'sentiment']])

# Calculate sentiment analysis result ratio
sentiment_counts = df['sentiment'].value_counts(normalize=True)

# color change
pastel_colors = sns.color_palette("pastel")

# vizualiztion pie chart
plt.figure(figsize=(8, 8))
plt.pie(sentiment_counts, labels=sentiment_counts.index, colors=pastel_colors, autopct='%1.1f%%', startangle=140)
plt.title('Sentiment Analysis of BBC USA_Canada News Articles - July 8 2024')
plt.axis('equal')  # keep circle 
plt.show()


# In[14]:


import spacy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# spaCy English model
nlp = spacy.load('en_core_web_sm')

# load the detailed CSV file
df = pd.read_csv('bbc_detailed_us-canada.csv')
df['body'] = df['body'].fillna('')

# Extract country name 
def extract_countries(text):
    doc = nlp(text)
    countries = [ent.text for ent in doc.ents if ent.label_ == 'GPE']  
    return countries

all_countries = []

for index, row in df.iterrows():
    try:
        countries = extract_countries(row['body'])
        all_countries.extend(countries)
    except Exception as e:
        print(f"Error processing article at {row['link']}: {e}")

# Ratio
country_counts = pd.Series(all_countries).value_counts()

print("Countries most mentioned in articles:")
print(country_counts.head(7))

top_countries = country_counts.head(7)

plt.figure(figsize=(12, 6))
sns.barplot(x=top_countries.values, y=top_countries.index, palette='pastel')
plt.title('Top 7 Countries or cities Mentioned in BBC USA_Canada News Articles - July 8 2024')
plt.xlabel('Mention Count')
plt.ylabel('Country')
plt.show()


# In[15]:


from collections import Counter
import spacy
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd

# spaCy English model
nlp = spacy.load('en_core_web_sm')

# load the detailed CSV file
df = pd.read_csv('bbc_detailed_us-canada.csv')

# concatenate all articles
all_text = ' '.join(df['body'].dropna())

# Convert all text to spaCy object
doc = nlp(all_text)

# Extracts only nouns and includes only words with 3 or more letters
nouns = [token.text.lower() for token in doc if token.pos_ == 'NOUN' and len(token.text) >= 3]

# Word frequency calculation
word_counts = Counter(nouns)

# Ratio
total_nouns = sum(word_counts.values())
word_ratios = {word: count / total_nouns for word, count in word_counts.items()}

# Extract the top 50 most used words
most_common_words = word_counts.most_common(50)

# Rario on top 50
most_common_ratios = [(word, count, word_ratios[word]) for word, count in most_common_words]
word_freq = {word: count for word, count in most_common_words}

# Print Result
print("<Top 50 Most Dominant Nouns in USA_Canada ")
print()
for word, count, ratio in most_common_ratios:
    print(f"{word}: {count}, Ratio: {ratio:.2%}")

# Word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Top 50 Most Dominant Nouns in USA_Canada - July 8 2024')
plt.show()


# # Top 50 Most Dominant Words in The World - July 8 2024

# In[15]:


from wordcloud import WordCloud

# the files
files = [
    'word_analysis_bbc_europe.csv',
    'word_analysis_bbc_asia.csv',
    'word_analysis_bbc_africa.csv',
    'word_analysis_bbc_us-canada.csv'
]

# concatenate all files
dataframes = [pd.read_csv(file) for file in files]
combined_df = pd.concat(dataframes, ignore_index=True)

word_counts = Counter()
for _, row in combined_df.iterrows():
    word_counts[row['word']] += row['count']

total_nouns = sum(word_counts.values())
word_ratios = {word: count / total_nouns for word, count in word_counts.items()}
most_common_words = word_counts.most_common(50)
most_common_ratios = [(word, count, word_ratios[word]) for word, count in most_common_words]

print("<Top 50 Most Dominant Nouns in The World Today>")
print()
for word, count, ratio in most_common_ratios:
    print(f"{word}: {count}, Ratio: {ratio:.2%}")

most_common_df = pd.DataFrame(most_common_ratios, columns=['word', 'count', 'ratio'])
most_common_df.to_csv('most_common_nouns_combined.csv', index=False)

# Rario on top 50
most_common_ratios = [(word, count, word_ratios[word]) for word, count in most_common_words]
word_freq_world = {word: count for word, count in most_common_words}

# Word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq_world)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Top 50 Most Dominant Words in The World - July 8 2024')
plt.show()

