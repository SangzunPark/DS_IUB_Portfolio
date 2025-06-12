
# Home Credit Default Risk - Machine Learning Project

##  Project Objective

This project tackles the **Home Credit Default Risk** prediction task using structured tabular data. The objective is to:
- Analyze customer behavior based on financial features.
- Predict the likelihood of loan repayment.
- Build a robust predictive pipeline using both **traditional ML** and **deep learning models**.

---

##  Dataset & Features

- Dataset: Home Credit Default Risk (Kaggle competition)
- Size: Multiple CSVs (e.g., `application_train.csv`, `bureau.csv`, etc.)
- Target Variable: `TARGET` (0 = repaid, 1 = defaulted)

### Feature Engineering Highlights
- Merged 6 datasets via customer ID (`SK_ID_CURR`)
- Performed exploratory analysis and feature selection using SHAP and feature importances
- Selected important numerical and categorical features
- Finalized feature list: ~150 features including engineered ones

---

##  Preprocessing Pipeline

- Separate pipelines for numerical and categorical features:
  - **Numerical**: imputation (mean), standard scaling
  - **Categorical**: imputation (most frequent), one-hot encoding
- Combined using `FeatureUnion`
- Saved features & importances using `pickle`

---

##  Traditional Machine Learning Models

Implemented and evaluated several classic ML models:
- **Logistic Regression**
- **Random Forest**
- **XGBoost**
- **LightGBM**

Evaluation Metric: ROC-AUC on validation set

Best model (LightGBM) used for feature selection and submission.

---

##  Deep Learning with PyTorch

Constructed a **Neural Network** using PyTorch:
- Fully connected feedforward architecture
- ReLU activation
- Binary Cross Entropy Loss
- Optimizer: Adam
- Early stopping based on validation loss
- Achieved performance comparable to ensemble tree models

---

##  Kaggle Submission

- Predictions generated from the best model
- Submission formatted as:
  - `SK_ID_CURR`
  - `TARGET (probability of default)`
- CSV file created and exported for Kaggle submission

---

##  Libraries Used

```bash
numpy
pandas
scikit-learn
matplotlib
seaborn
xgboost
lightgbm
torch
pickle
```

---

##  Insights

- Feature selection significantly improves model performance
- LightGBM and PyTorch performed best in cross-validation
- Pipeline modularity helps in model comparison and Kaggle testing

---

##  Files

- `ML_Project_HCDR.py` – full pipeline code
- `ML_Project_HCDR.ipynb` – Jupyter development version


---

##  Author

**Sangzun Park and IUB AML team members**
Graduate Student, MS in Data Science  
Indiana University Bloomington  
