#  BBC News Web Scraping & Text Analysis Project

## Project Overview

This mini project involves web scraping articles from the **BBC News** website (Europe, Asia, Africa, and North America) and performing **sentiment analysis**, **named entity recognition (NER)**, and **noun frequency analysis** on the collected data.

---

##  Key Features

- **Data Collection**  
  Scrape article titles, links, dates, and full texts using BeautifulSoup.

- **Sentiment Analysis**  
  Classify articles into *Positive*, *Negative*, or *Neutral* using VADER and visualize the proportions.

- **Named Entity Recognition (NER)**  
  Extract and count country and city mentions using spaCy.

- **Noun Frequency Analysis**  
  Identify the top 50 most frequently used nouns and visualize with WordCloud.

- **Global Aggregation**  
  Merge data across regions to highlight globally dominant issues and keywords.

---

##  Libraries Used

- `requests`, `BeautifulSoup`, `pandas`, `matplotlib`, `seaborn`  
- `vaderSentiment`, `spacy`, `wordcloud`, `collections.Counter`

---

##  Key Findings

- **France** and **Paris** were frequently mentioned due to major elections.
- Common key terms included: **people**, **election**, and **alliance**.
- European articles showed a high level of positive sentiment.
- Across all continents, **"people"** was the most mentioned noun, reflecting a global human-centered focus.

---

##  Author

**Sangzun Park**
Graduate Student, MS in Data Science  
Indiana University Bloomington  