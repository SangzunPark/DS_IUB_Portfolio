# Olympic Performance and Economic Indicators

This project explores the relationship between Olympic medal rankings and national economic/development indicators, using data from the 2016 Summer Olympics and prior games. It includes exploratory data analysis, classification modeling (logistic regression and random forest), feature importance analysis, and clustering with KMeans and PCA.

---

##  Files

- `Usableai_Project_Final.ipynb`: Main analysis notebook containing the entire workflow.
- `requirements.txt`: Python dependencies (you can generate it with `pip freeze > requirements.txt`).

---

##  Analysis Overview

1. **Exploratory Data Analysis (EDA)**: Medal standings and socioeconomic indicators (GDP, GNI per capita, HDI, population).
2. **Classification Models**: Predicting Top 10 / Top 20 medal-winning countries using:
   - Logistic Regression (with GridSearch tuning)
   - Random Forest (optimized model)
3. **Feature Importance**: Ranking contribution of each variable.
4. **Clustering Analysis**: Grouping countries using `medals_per_million` + development indicators (via KMeans and PCA).

---

##  How to Run

1. Clone the repository or download the `.ipynb` file.
2. Install the required packages:

```bash
pip install -r requirements.txt