#!/usr/bin/env python
# coding: utf-8

# 
# 

# ## Project Implementation

# In[1]:


from scipy import stats
import latexify
import time
import numpy as np
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
import os
import zipfile
import pickle
import json
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.metrics import log_loss, classification_report, roc_auc_score, make_scorer
from scipy import stats
from sklearn.svm import SVC
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')


# In[2]:


application_train = pd.read_csv('application_train.csv')        
application_test = pd.read_csv('application_test.csv')        
bureau = pd.read_csv('bureau.csv')                  
bureau_balance = pd.read_csv('bureau_balance.csv')            
credit_card_balance = pd.read_csv('credit_card_balance.csv')      
previous_application = pd.read_csv('previous_application.csv')     
installments_payments = pd.read_csv('installments_payments.csv')     
POS_CASH_balance = pd.read_csv('POS_CASH_balance.csv')


# ## EDA

# In[4]:


application_train.info()


# In[5]:


application_train.describe()


# In[6]:


application_test.info()


# In[7]:


application_test.describe()


# In[8]:


application_train.describe(include='all')


# In[9]:


application_test.describe(include='all')


# In[10]:


application_train.dtypes


# In[11]:


application_train.isnull()


# In[12]:


application_train.isna()


# In[13]:


application_train.corr()


# In[14]:


application_train.columns


# ## Plots based on Gender

# In[15]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(5, 5))
with sns.color_palette("rocket"):
    sns.countplot(data=application_train, x='CODE_GENDER')
plt.xlabel('Gender')
plt.ylabel('Count of Borrowers')
plt.title('Count of Borrowers by Gender')
plt.show()


# ## Plots based on Family Status

# In[16]:


plt.style.use('ggplot')
plt.figure(figsize=(8, 8))
with sns.color_palette("rocket"):
    sns.countplot(data=application_train, y='NAME_FAMILY_STATUS')
plt.xlabel('Count')
plt.ylabel('Family Status')
plt.title('Distribution of Family Status')
plt.show()


# ## Plots based on Age

# In[17]:


ages = [int(-x/365) for x in application_train.DAYS_BIRTH]

age_counts = pd.Series(ages).value_counts().sort_index()
labels = [f'{age}-{age+9}' for age in range(0, 100, 10)]

plt.figure(figsize=(7, 7))
with sns.color_palette("rocket_r"):
    age_counts.plot(kind='bar')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()


# ## Plot based on Count of Children

# In[18]:


plt.figure(figsize=(20, 8), dpi=200)
plt.hist(application_train['CNT_CHILDREN'], bins=range(12), edgecolor='black')
plt.title("Histogram of Frequency of Children")
plt.xlabel('Number of Children')
plt.ylabel('Frequency')
plt.show()


# ## Unique Values in Columns

# In[19]:


unique_counts = application_train.nunique()

for col_name, unique_count in unique_counts.iteritems():
    print(f"{col_name}: {unique_count}")


# In[20]:


plt.figure(figsize=(10, 5))
filtered_counts = unique_counts[unique_counts > 10]
plt.bar(filtered_counts.index, filtered_counts.values)
plt.xticks(rotation=90)
plt.xlabel('Column Name')
plt.ylabel('Count of Unique Values')
plt.title('Count of Unique Values per Column (with more than 10 unique values)')
plt.show()


# ## Bureau Balance Dataset
# 
#  This the Data Description Section of the BB Dataset

# In[21]:


bureau_balance.info()


# In[22]:


bureau_balance.describe()


# ## Credit Card Balance
#  This the Data Description Section of the CC Dataset

# In[23]:


credit_card_balance.info()


# In[24]:


credit_card_balance.describe()


# In[25]:


sns.set(rc={'figure.figsize':(18,5)})
with sns.color_palette("viridis"):
    ax = sns.violinplot(x= credit_card_balance['AMT_BALANCE'])
ax.set_xticks([x/10*1e6 for x in range(-5,15,1)])
ax.set_xlabel('AMT_BALANCE')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of AMT_BALANCE')
plt.show()


# In[26]:


sns.set(rc={'figure.figsize':(12,5)})
with sns.color_palette("pastel"):
    ax = sns.countplot(data= credit_card_balance[credit_card_balance.CNT_DRAWINGS_CURRENT < 10], x="CNT_DRAWINGS_CURRENT")
    ax.set_xticks(range(0,10,1))
    ax.set_xlabel("Number of Drawings")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Number of Drawings (Less than 10)")
    plt.show()


# ## Installment Payments Dataset

# In[27]:


installments_payments.info()


# In[28]:


installments_payments.describe()


# In[29]:


plt.hist( installments_payments["DAYS_ENTRY_PAYMENT"] / -365 , bins=25, color = "red")
plt.xlabel("Years Since Payment")
plt.ylabel("Frequency")
plt.title("Distribution of Payment Entry Days")
plt.show()


# ## POS Cash Balance

# In[30]:


POS_CASH_balance.info()


# In[31]:


POS_CASH_balance.describe()


# ## Previous Applications

# In[32]:


previous_application.info()


# In[33]:


previous_application.describe()


# ## Missing data

# In[34]:


missing_data = pd.DataFrame({'column_name': application_train.columns, 
                             'count_missing': application_train.isnull().sum(), 
                             'pct_missing': application_train.isnull().sum() / len(application_train) * 100})
missing_data


# In[35]:


missing_data_filtered = missing_data.query('count_missing > 100')
missing_data_filtered.plot(x='column_name', y='count_missing', kind='bar', figsize=(12,5))
plt.xlabel('Column Name')
plt.ylabel('Count of Missing Values')
plt.title('Count of Missing Values by Column (Missing > 100)')
plt.show()


# In[36]:


missing_data_test = pd.DataFrame({'column_name': application_test.columns, 
                             'count_missing': application_test.isnull().sum(), 
                             'pct_missing': application_test.isnull().sum() / len(application_test) * 100})
missing_data_test


# In[37]:


missing_data_test_filtered = missing_data_test.query('count_missing > 100')
missing_data_test_filtered.plot(x='column_name', y='count_missing', kind='bar', figsize=(12,5))
plt.xlabel('Column Name')
plt.ylabel('Count of Missing Values')
plt.title('Count of Missing Values by Column (Missing > 100)')
plt.show()


# In[38]:


sns.catplot(data = application_train, x='TARGET', kind = 'count')
plt.xlabel('Target');
plt.ylabel('Numbers of Borrowers');
plt.title('Target values against the number of borrowers');
plt.show()


# In[39]:


# Count the number of unique values in a column
unique_count = application_train['SK_ID_CURR'].nunique()

# Print the result
print('The number of unique values in the column is:', unique_count)
print(unique_count == application_train.shape[0])


# In[40]:


application_train.shape


# In[41]:


application_test.shape


# In[42]:


df_apps = previous_application
df_apps.head()


# In[43]:


plt.figure(figsize = (10, 5), dpi = 200)
prevAppCounts = df_apps['SK_ID_CURR'].value_counts(dropna=False)
prevAppCounts[prevAppCounts >50].plot(kind='bar')
plt.xticks(rotation=90)
plt.show()


# ## Correlation Analysis

# In[44]:


corr = application_train.corr()['TARGET'].sort_values()
print('Top Positive Correlations:\n', corr.tail(10))
print('\nTop Negative Correlations:\n', corr.head(10))


# In[45]:


corr_train = application_train.corr()['TARGET'].sort_values()
corr_train = corr_train.reset_index().rename(columns={'index':'Attributes','TARGET':'Correlation'})
corr_train


# In[46]:


plt.figure(figsize = (10, 5))
sns.barplot(x='Attributes',y='Correlation',data= corr_train[corr_train.Correlation>0], palette = ['yellow'])
plt.xlabel('Attributes');
plt.ylabel('Positive Correlation');
plt.suptitle('Positive Correlated attributes with target');
plt.xticks(rotation=90);
plt.show()


# In[47]:


plt.figure(figsize = (30, 5))
sns.barplot(x='Attributes',y='Correlation',data= corr_train[corr_train.Correlation<=0], palette = ['blue'])
plt.xlabel('Attributes');
plt.ylabel('Negative Correlation');
plt.suptitle('Negative Correlated attributes with target');
plt.xticks(rotation=90);
plt.show()


# In[48]:


application_train['DAYS_BIRTH']=abs(application_train['DAYS_BIRTH'])
corr1= application_train.corr()['TARGET']
corr1['DAYS_BIRTH']


# In[49]:


plt.hist(application_train['DAYS_BIRTH']/365, edgecolor='k',bins=50,color ="green")
plt.title('Client Age'); 
plt.xlabel('Age (in years)'); 
plt.ylabel('Count');


# #### Average failure statistics to repay loans by age 

# In[50]:


age=application_train[['TARGET','DAYS_BIRTH']]
age['YEARS']=age['DAYS_BIRTH']/365


# In[51]:


age['BIN']=pd.cut(age['YEARS'],bins=np.linspace(20,70,num=10))
age.head(20)


# In[52]:


# Grouping the data by the bin and calculate averages
age_grouped=age.groupby('BIN').mean()
age_grouped


# In[53]:


plt.figure(figsize = (10,8))
plt.bar(age_grouped.index.astype(str),100 * age_grouped['TARGET'],color ="blue")
plt.xticks(rotation = 90);
plt.xlabel('Risk (%)'); 
plt.ylabel('Age (inyears)');
plt.title('Age vs Risk');


# ### Missing values in prevApps

# In[54]:


df_apps.isna().sum()


# ### Feature engineering for prevApp table

# The groupby output will have an index or multi-index on rows corresponding to your chosen grouping variables. To avoid setting this index, pass “as_index=False” to the groupby operation.
# 
# ```python
# import pandas as pd
# import dateutil
# 
# # Load data from csv file
# data = pd.DataFrame.from_csv('phone_data.csv')
# # Convert date from string to date times
# data['date'] = data['date'].apply(dateutil.parser.parse, dayfirst=True)
# 
# data.groupby('month', as_index=False).agg({"duration": "sum"})
# ```
# Pandas `reset_index()` to convert Multi-Index to Columns
# We can simplify the multi-index dataframe using reset_index() function in Pandas. By default, Pandas reset_index() converts the indices to columns.
# 
# ### Fixing Column names after Pandas agg() function to summarize grouped data
# Since we have both the variable name and the operation performed in two rows in the Multi-Index dataframe, we can use that and name our new columns correctly.
# 
# For more details unstacking groupby results and examples please see [here](https://cmdlinetips.com/2020/05/fun-with-pandas-groupby-aggregate-multi-index-and-unstack/)
# 
# 
# 
# 
# For more details and examples please see [here](https://www.shanelynn.ie/summarising-aggregation-and-grouping-data-in-python-pandas/)

# In[3]:


# Create aggregate features (via pipeline)
class FeaturesAggregation(BaseEstimator, TransformerMixin):
    def __init__(self, features=None, agg_needed=["mean"]): # no *args or **kargs
        self.features = features
        self.agg_needed = agg_needed
        self.agg_op_features = {}
        for f in features:
            self.agg_op_features[f] = self.agg_needed[:]

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        result = X.groupby(["SK_ID_CURR"]).agg(self.agg_op_features)
        df_result = pd.DataFrame()
        for x1, x2 in result.columns:
            new_col = x1 + "_" + x2
            df_result[new_col] = result[x1][x2]
        df_result = df_result.reset_index(level=["SK_ID_CURR"])
        return df_result


# In[4]:


previous_feature = ["AMT_APPLICATION", "AMT_CREDIT", "AMT_ANNUITY", "approved_credit_ratio", 
                    "AMT_ANNUITY_credit_ratio", "Interest_ratio", "LTV_ratio", "SK_ID_PREV", "approved"]

agg_needed = ["min", "max", "mean", "count", "sum"]

def previous_feature_aggregation(df, feature, agg_needed):
    df['approved_credit_ratio'] = (df['AMT_APPLICATION']/df['AMT_CREDIT']).replace(np.inf, 0)
    # installment over credit approved ratio
    df['AMT_ANNUITY_credit_ratio'] = (df['AMT_ANNUITY']/df['AMT_CREDIT']).replace(np.inf, 0)
    # total interest payment over credit ratio
    df['Interest_ratio'] = (df['AMT_ANNUITY']/df['AMT_CREDIT']).replace(np.inf, 0)
    #  loan cover ratio
    df['LTV_ratio'] = (df['AMT_CREDIT']/df['AMT_GOODS_PRICE']).replace(np.inf, 0)
    df['approved'] = np.where(df.AMT_CREDIT >0 ,1, 0)
    
    test_pipeline = make_pipeline(FeaturesAggregation(feature, agg_needed))
    return(test_pipeline.fit_transform(df))
    
previous_application_agg = previous_feature_aggregation(previous_application, previous_feature, agg_needed)


# In[5]:


previous_application_agg.isna().sum()


# # Joining secondary tables with the primary table
# 
# In the case of the HCDR competition (and many other machine learning problems that involve multiple tables in 3NF or not)  we need to join these datasets (denormalize) when using a machine learning pipeline. Joining the secondary tables with the primary table will lead to lots of new features about each loan application; these features will tend to be aggregate type features or meta data about the loan or its application. How can we do this when using Machine Learning Pipelines?
# 
# ## Joining `previous_application` with `application_x`
# We refer to the `application_train` data (and also `application_test` data also) as the **primary table** and the other files as the **secondary tables** (e.g., `previous_application` dataset). All tables can be joined using the primary key `SK_ID_PREV`.
# 
# Let's assume we wish to generate a feature based on previous application attempts. In this case, possible features here could be:
# 
# * A simple feature could be the number of previous applications.
# * Other summary features of original features such as `AMT_APPLICATION`, `AMT_CREDIT` could be based on average, min, max, median, etc.
#  
# To build such features, we need to join the `application_train` data (and also `application_test` data also) with the 'previous_application' dataset (and the other available datasets).
# 
# When joining this data in the context of pipelines, different strategies come to mind with various tradeoffs:
# 
# 1. Preprocess each of the non-application data sets, thereby generating many new (derived) features, and then joining (aka merge) the results with the `application_train` data (the labeled dataset) and with the `application_test` data (the unlabeled submission dataset) prior to processing the data (in a train, valid, test partition) via your machine learning pipeline. [This approach is recommended for this HCDR competition. WHY?]
# 
# * Do the joins as part of the transformation steps. [Not recommended here. WHY?]. How can this be done? Will it work?
#   * This would be necessary if we had dataset wide features such as IDF (inverse document frequency) which depend on the entire subset of data as opposed to a single loan application (e.g., a feature about the relative amount applied for such as the percentile of the loan amount being applied for).
#  
# I want you to think about this section and build on this.

# ## Nextstep is secondary tables processing and merging with the primary table
# 
# 1. Transform all the secondary tables to features that can be joined into the main table the application table (labeled and unlabeled)
#    * 'bureau', 'bureau_balance', 'credit_card_balance', 'installments_payments', 
#    * 'previous_application', 'POS_CASH_balance'
# * Merge the transformed secondary tables with the primary tables (i.e., the `application_train` data (the labeled dataset) and with the `application_test` data (the unlabeled submission dataset)), thereby leading to X_train, y_train, X_valid, etc.
# * Proceed with the learning pipeline using X_train, y_train, X_valid, etc.
# * Generate a submission file using the learnt model
# 

# ### Missing values handling in Installment payments

# In[6]:


installments_payments.isna().sum()


# In[7]:


payments_features = ["DAYS_INSTALMENT_DIFF", "AMT_PATMENT_PCT"]

agg_needed = ["mean"]

def payments_feature_aggregation(df, feature, agg_needed):
    df['DAYS_INSTALMENT_DIFF'] = df['DAYS_INSTALMENT'] - df['DAYS_ENTRY_PAYMENT']
    df['AMT_PATMENT_PCT'] = [x/y if (y != 0) & pd.notnull(y) else np.nan for x,y in zip(df.AMT_PAYMENT,df.AMT_INSTALMENT)]
    
    test_pipeline = make_pipeline(FeaturesAggregation(feature, agg_needed))
    return(test_pipeline.fit_transform(df))
    
installments_payments_agg = payments_feature_aggregation(installments_payments, payments_features, agg_needed)


# In[8]:


installments_payments_agg.isna().sum()


# ### Missing values handling in Credit Card Balance

# In[9]:


credit_card_balance.isna().sum()


# In[10]:


credit_features = ["AMT_BALANCE", "AMT_DRAWINGS_PCT", "AMT_DRAWINGS_ATM_PCT", "AMT_DRAWINGS_OTHER_PCT", "AMT_DRAWINGS_POS_PCT", 
                   "AMT_PRINCIPAL_RECEIVABLE_PCT", "CNT_DRAWINGS_ATM_CURRENT", "CNT_DRAWINGS_CURRENT", "CNT_DRAWINGS_OTHER_CURRENT",
                   "CNT_DRAWINGS_POS_CURRENT", "SK_DPD", "SK_DPD_DEF"]

agg_needed = ["mean"]

def credit_feature_aggregation(df, feature, agg_needed):
    df['AMT_DRAWINGS_PCT'] = [x/y if (y != 0) & pd.notnull(y) else np.nan for x,y in zip(df.AMT_DRAWINGS_CURRENT,df.AMT_CREDIT_LIMIT_ACTUAL)]
    df['AMT_DRAWINGS_ATM_PCT'] = [x/y if (y != 0) & pd.notnull(y) else np.nan for x,y in zip(df.AMT_DRAWINGS_ATM_CURRENT,df.AMT_CREDIT_LIMIT_ACTUAL)]
    df['AMT_DRAWINGS_OTHER_PCT'] = [x/y if (y != 0) & pd.notnull(y) else np.nan for x,y in zip(df.AMT_DRAWINGS_OTHER_CURRENT,df.AMT_CREDIT_LIMIT_ACTUAL)]
    df['AMT_DRAWINGS_POS_PCT'] = [x/y if (y != 0) & pd.notnull(y) else np.nan for x,y in zip(df.AMT_DRAWINGS_POS_CURRENT,df.AMT_CREDIT_LIMIT_ACTUAL)]
    df['AMT_PRINCIPAL_RECEIVABLE_PCT'] = [x/y if (y != 0) & pd.notnull(y) else np.nan for x,y in zip(df.AMT_RECEIVABLE_PRINCIPAL,df.AMT_RECIVABLE)]
    
    test_pipeline = make_pipeline(FeaturesAggregation(feature, agg_needed))
    return(test_pipeline.fit_transform(df))


credit_card_balance_agg = credit_feature_aggregation(credit_card_balance, credit_features, agg_needed)
  


# In[11]:


credit_card_balance_agg.isna().sum()


# In[12]:


application_train.keys()
installments_payments_agg.keys()
previous_application_agg.keys()
credit_card_balance_agg.keys()


# ## Merge primary table with secondary tables using features based on meta data and aggregage stats performed above

# In[46]:


train_dataset= application_train  #primary dataset
    
merged_data = True

if merged_data:
    # 1. Join/Merge in prevApps Data
    train_dataset = train_dataset.merge(previous_application_agg, how='left', on='SK_ID_CURR')

    # 2. Join/Merge in Installments Payments  Data
    train_dataset = train_dataset.merge(installments_payments_agg, how='left', on="SK_ID_CURR")

    # 3. Join/Merge in Credit Card Balance Data
    train_dataset = train_dataset.merge(credit_card_balance_agg, how='left', on="SK_ID_CURR")


# # Creating a test data set with similar fields

# In[47]:


X_kaggle_test= application_test


if merged_data:
    # 1. Join/Merge in prevApps Data
    X_kaggle_test = X_kaggle_test.merge(previous_application_agg, how='left', on='SK_ID_CURR')

    # 2. Join/Merge in Installments Payments  Data
    X_kaggle_test = X_kaggle_test.merge(installments_payments_agg, how='left', on="SK_ID_CURR")

    # 3. Join/Merge in Credit Card Balance Data
    X_kaggle_test = X_kaggle_test.merge(credit_card_balance_agg, how='left', on="SK_ID_CURR")


# ## HCDR preprocessing

# ## Evaluation metrics
# Submissions are evaluated on [area under the ROC curve](http://en.wikipedia.org/wiki/Receiver_operating_characteristic) between the predicted probability and the observed target.
# 
# The SkLearn `roc_auc_score` function computes the area under the receiver operating characteristic (ROC) curve, which is also denoted by AUC or AUROC. By computing the area under the roc curve, the curve information is summarized in one number. 
# 
# ```python
# from sklearn.metrics import roc_auc_score
# >>> y_true = np.array([0, 0, 1, 1])
# >>> y_scores = np.array([0.1, 0.4, 0.35, 0.8])
# >>> roc_auc_score(y_true, y_scores)
# 0.75
# ```

# #### ACCURACY:

# In[67]:


import latexify
@latexify.function
def Accuracy():
    return (TN + TP) / (TN + FP + TP + FN)
Accuracy


# #### F1 SCORE:
# 

# In[68]:


@latexify.function
def F1Score():
    return (2) * (Precision * Recall) / (Precision + Recall)
F1Score


# #### RECALL:
# 

# In[69]:


@latexify.function
def Recall():
    return (TP) / (TP + FN)
Recall


# #### LOG LOSS:
# 

# In[70]:


@latexify.function(use_math_symbols=True)
def logLoss():
    return (-1 / N) * Sigma(_i**N) (y_i * log(p(y_i)) + (1 - y_i) * log(1 - p(y_i)))
logLoss


# In[71]:


#Block Diagram


# In[17]:


from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values


# In[73]:


def pct(x):
    return round(100*x,3)


# # Results as Experiment Logs

# In[74]:


try:
    experimentLog
except NameError:
    experimentLog = pd.DataFrame(columns=["exp_name", 
                                   "Train Acc", 
                                   "Valid Acc",
                                   "Test  Acc",
                                   "Train AUC", 
                                   "Valid AUC",
                                   "Test  AUC", 
                                   "Train F1 Score",
                                   "Valid F1 Score",
                                   "Test F1 Score"
                                  ])


# In[75]:


def get_results(expLog, exp_name, model, X_train, y_train, X_valid, y_valid, X_test, y_test):
    expLog.loc[len(expLog)] = [f"{exp_name}"] + list(np.round(
               [accuracy_score(y_train, model.predict(X_train)), 
                accuracy_score(y_valid, model.predict(X_valid)),
                accuracy_score(y_test, model.predict(X_test)),
                roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]),
                roc_auc_score(y_valid, model.predict_proba(X_valid)[:, 1]),
                roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]),
                f1_score(y_train, model.predict(X_train)), 
                f1_score(y_valid, model.predict(X_valid)), 
                f1_score(y_test, model.predict(X_test))],
    4))
    return expLog


# # Creating the Pipeline for the features transformation

# In[48]:


num_features = []
cat_features = []
for x in train_dataset:
    if(train_dataset[x].dtype == np.float64 or train_dataset[x].dtype == np.int64):
        num_features.append(x)
    else:
        cat_features.append(x)
num_features.remove('TARGET')
num_features.remove('SK_ID_CURR')

# Pipeline
cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_features)),
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(sparse=False, handle_unknown="ignore"))
    ])

num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_features)),
        ('imputer', SimpleImputer(strategy='mean')),
        ('std_scaler', StandardScaler()),
    ])

data_prep_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])  

selected_features = num_features + cat_features + ["SK_ID_CURR"]
tot_features = f"{len(selected_features)}:   Num:{len(num_features)},    Cat:{len(cat_features)}"

print('Total Features:', tot_features)


# # Splitting the data into Train and Test datasets

# In[49]:


y_train = train_dataset['TARGET']
X_train = train_dataset[selected_features]
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.17, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.17, random_state=42)
X_kaggle_test= X_kaggle_test[selected_features]

print(f"X train           shape: {X_train.shape}")
print(f"X validation      shape: {X_valid.shape}")
print(f"X test            shape: {X_test.shape}")
print(f"X X_kaggle_test   shape: {X_kaggle_test.shape}")


# In[50]:


(data_prep_pipeline)


# # Baseline Model evalutaion on the Full scale data

# In[79]:


np.random.seed(42)
basepipeline_regressor = Pipeline([
        ("preparation", data_prep_pipeline),
        ("linear", LogisticRegression())
    ])
model = basepipeline_regressor.fit(X_train, y_train)


# In[80]:


np.round(accuracy_score(y_train, model.predict(X_train)), 3)


# ## Performing Crossvalidation on the Dataset

# In[81]:


cv_Splits = ShuffleSplit(n_splits=5, test_size=0.45, random_state=42)


# In[82]:


from sklearn.model_selection import cross_val_score

model = basepipeline_regressor.fit(X_train, y_train)
np.random.seed(42)

logit_scores = cross_val_score(basepipeline_regressor,X_train , y_train,cv=cv_Splits)               
logit_score_train = pct(logit_scores.mean())

logit_score_test  = basepipeline_regressor.score(X_test, y_test)

print("Test Acc: ", logit_score_test)


# In[83]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss, classification_report, roc_auc_score, make_scorer

roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])


# In[84]:


pred = model.predict(X_test)
plt.hist(pred)
f1_train = f1_score(y_train, model.predict(X_train))
f1_valid = f1_score(y_valid, model.predict(X_valid))
f1_test = f1_score(y_test, model.predict(X_test))
print("F1 Score for Test set: ", f1_test)


# In[85]:


experiment_Labels = f"Baseline_{len(selected_features)}_features"
experimentLog.loc[len(experimentLog)] = [f"{experiment_Labels}"] + list(np.round(
               [accuracy_score(y_train, model.predict(X_train)), 
                accuracy_score(y_valid, model.predict(X_valid)),
                accuracy_score(y_test, model.predict(X_test)),
                roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]),
                roc_auc_score(y_valid, model.predict_proba(X_valid)[:, 1]),
                roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]),
                f1_train, f1_valid, f1_test],
    4)) 
experimentLog


# Inorder to improve the F1 Score, we have used the Adaptive Synthetic Sampling on the data using the library function in Python

# # Inorder to improve the F1 Score, we have to use the either the Synthetic Sample generator or the undersampling to make it a balanced dataset, The Synthetic sample generator takes only numerical input and hence the categorical inputs lose the singnificance before feeding them to model, So we are opting for undersampling

# In[51]:


train_dataset_sampled=train_dataset[train_dataset['TARGET']==1]
train_dataset_sampled = train_dataset_sampled.append(train_dataset[train_dataset['TARGET']==0].reset_index(drop=True).sample(n = 60000))


# In[52]:


train_dataset_sampled['TARGET'].value_counts()


# In[53]:


train_dataset_sampled_2=train_dataset[train_dataset.TARGET==1].copy()
train_dataset_sampled_2['weight']=1
default_cashloans=train_dataset_sampled_2[(train_dataset_sampled_2.TARGET==1) & (train_dataset_sampled_2.NAME_CONTRACT_TYPE=='Cash loans')].shape[0]
default_revolvingloans=train_dataset_sampled_2[(train_dataset_sampled_2.TARGET==1)&(train_dataset_sampled_2.NAME_CONTRACT_TYPE=='Revolving loans')].shape[0]


# In[54]:


# Undersampling Cash loans 
df_sample=train_dataset[(train_dataset.NAME_CONTRACT_TYPE =='Cash loans')&(train_dataset.TARGET==0)].sample(n=int(1.45*default_cashloans),random_state=1)
df_sample['weight']=train_dataset[(train_dataset.NAME_CONTRACT_TYPE=='Cash loans')&(train_dataset.TARGET==0)].shape[0]/int(1.45*default_cashloans)
train_dataset_sampled_2=pd.concat([train_dataset_sampled_2, df_sample])


# In[55]:


# Undersampling Revovling Loans
df_sample=train_dataset[(train_dataset.NAME_CONTRACT_TYPE=='Revolving loans')&(train_dataset.TARGET==0)].sample(n=int(1.45*default_revolvingloans),random_state=1)
df_sample['weight']=train_dataset[(train_dataset.NAME_CONTRACT_TYPE=='Revolving loans')&(train_dataset.TARGET==0)].shape[0]/int(1.45*default_revolvingloans)
train_dataset_sampled_2=pd.concat([train_dataset_sampled_2, df_sample])


# In[56]:


train_dataset_sampled_2['TARGET'].value_counts()


# In[57]:


y_train_sampled = train_dataset_sampled_2['TARGET']
X_train_sampled = train_dataset_sampled_2[selected_features]
X_train_sampled, X_valid_sampled, y_train_sampled, y_valid_sampled = train_test_split(X_train_sampled, y_train_sampled, test_size=0.15, random_state=42)
X_train_sampled, X_test_sampled, y_train_sampled, y_test_sampled = train_test_split(X_train_sampled, y_train_sampled, test_size=0.15, random_state=42)
X_kaggle_test= X_kaggle_test[selected_features]

print(f"X train sampled           shape: {X_train_sampled.shape}")
print(f"X validation sampled      shape: {X_valid_sampled.shape}")
print(f"X test sampled          shape: {X_test_sampled.shape}")
print(f"X X_kaggle_test   shape: {X_kaggle_test.shape}")


# In[94]:


model = basepipeline_regressor.fit(X_train_sampled, y_train_sampled)


# # Results of the experiment

# In[95]:


exp_name = f"Baseline_resampled2_{len(selected_features)}_features"
experimentLog = get_results(experimentLog, exp_name, model, X_train_sampled, y_train_sampled, X_valid_sampled, y_valid_sampled, X_test_sampled, y_test_sampled)
experimentLog


# # Support Vector Machine

# The SVM has a longer runtime on the Entire feature dataset 
# 

# In[96]:


from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
np.random.seed(42)

svm_sampled_pipeline = Pipeline([
    ("preparation", data_prep_pipeline),
    ("svm", LinearSVC(random_state=42,dual=False))
])


LSVC_model = CalibratedClassifierCV(svm_sampled_pipeline, cv=5)


# In[97]:


LSVC_model.fit(X_train_sampled, y_train_sampled)


# In[98]:


score_test  = LSVC_model.score(X_test_sampled, y_test_sampled)
score_test


# In[99]:


exp_name = f"SVM_resampled2_{len(selected_features)}_features"
experimentLog = get_results(experimentLog, exp_name, LSVC_model, X_train_sampled, y_train_sampled, X_valid_sampled, y_valid_sampled, X_test_sampled, y_test_sampled)
experimentLog


# # Random Forest

# In[100]:


from sklearn.ensemble import RandomForestClassifier
random_forest_pipeline= Pipeline([
    ("preparation", data_prep_pipeline),
    ("random forest", RandomForestClassifier(random_state=42, bootstrap=True, max_depth=21, 
                      max_features=6, min_samples_leaf=10, min_samples_split=13, n_estimators=500))
])


model = random_forest_pipeline.fit(X_train_sampled, y_train_sampled)

exp_name = f"Random_Forest_sampled_{len(selected_features)}_features"
experimentLog = get_results(experimentLog, exp_name, model, X_train_sampled, y_train_sampled, X_valid_sampled, y_valid_sampled, X_test_sampled, y_test_sampled)
experimentLog


# # XGBoost

# In[101]:


# Attaching XGBoost model to the above pipeline
xgboost_pipeline = Pipeline([
    ("preparation", data_prep_pipeline),
    ("xgboost", XGBClassifier(random_state=42, 
                objective='binary:logistic', max_depth=5, eta=0.001, 
                learning_rate=0.01, colsample_bytree=0.7, n_estimators=1000))
])

model = xgboost_pipeline.fit(X_train_sampled, y_train_sampled)

# Results
exp_name = f"XGBoost_Sampled_{len(selected_features)}_features"
experimentLog = get_results(experimentLog, exp_name, model, X_train_sampled, y_train_sampled, X_valid_sampled, y_valid_sampled, X_test_sampled, y_test_sampled)
experimentLog


# 
# ### Hyper-parameter tuning for Random Forest model

# In[102]:


features_dict = dict()
def GridSearch_Validation(X_train, y_train, X_test, y_test):
    classifiers = [
        ('XGBoost', XGBClassifier(random_state=42)),
        ('RandomForest', RandomForestClassifier(random_state=42))
        #('LogisticRegression', LogisticRegression())
    ]
    
    # grid search parameters for the classifier
    
    param_grid = {
         'XGBoost': {
        'max_depth': [3, 7],
        'learning_rate': [0.1, 0.01],
        'n_estimators': [100, 500],
        'subsample': [0.4, 1.0],
        'colsample_bytree': [0.4, 0.7]
    },
        'RandomForest': {
        'n_estimators': [100, 500],
        'max_depth': [5, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
        }
    }
    
    
    for (name, classifier) in classifiers:
        
        print('****** STARTING', name,' Model *****')
        parameters = param_grid[name]
        print("Parameters:")
        for p in sorted(parameters.keys()):
            print("\t"+str(p)+": "+ str(parameters[p]))
        
        # generate the pipeline
        full_pipeline = Pipeline([
            ("preparation", FeatureUnion(transformer_list=[("num_pipeline", num_pipeline)])),
            ("predictor", classifier)
        ])
        
        
        # Execute the grid search
        params = {}
        for p in parameters.keys():
            pipe_key = 'predictor__'+str(p)
            params[pipe_key] = parameters[p] 
        
        grid_search = GridSearchCV(full_pipeline, params, scoring='accuracy', cv=2, 
                                   n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)

        grid_search.best_estimator_.fit(X_train, y_train)

        
        # Training accuracy
        cvSplits = ShuffleSplit(n_splits=3, test_size=0.7, random_state=42)
        best_train_scores = cross_val_score(full_pipeline,X_train , y_train, cv=cvSplits)
        best_train_accuracy = pct(best_train_scores.mean())
        
        best_test_accuracy = pct(grid_search.best_estimator_.score(X_test, y_test))

        # Importance of features
        features = num_features[:]
        print('\nTotal number of features:', len(features))
        importances = grid_search.best_estimator_.named_steps["predictor"].feature_importances_
        
        # selecting features based on importance values
        new_indices = [idx for idx, x in enumerate(importances) if x>0.01]
        new_importances = [x for idx, x in enumerate(importances) if x>0.01]
        new_features = [features[i] for i in new_indices]
        
        print('Total number of selected features:', len(new_features))
        
        # Best parameters found using grid search
        print(f"Best Parameters for {name}:")
        best_parameters = grid_search.best_estimator_.get_params()
        best_params = []
        for param_name in sorted(params.keys()):
            best_params.append((param_name, best_parameters[param_name]))
            print("\t"+str(param_name)+": " + str(best_parameters[param_name]))
        print("****** FINISH",name," *****")
        
        # Results
        #experimentLog = get_results(experimentLog, exp_name, model, X_train_sampled, y_train_sampled, X_valid_sampled, y_valid_sampled, X_test_sampled, y_test_sampled)
        features_dict['features'] = features
        features_dict['importances'] = importances


# In[103]:


GridSearch_Validation(X_train_sampled, y_train_sampled,
                      X_test_sampled, y_test_sampled)


# # Feature Selection
# 

# ## Performance with features with importance > 0 for both XGBoost and Random Forest Models

# In[117]:


features = features_dict['features']
importances = features_dict['importances']

new_indices = [idx for idx, x in enumerate(importances) if x>0]
new_importances = [x for idx, x in enumerate(importances) if x>0]

new_features = [features[i] for i in new_indices]
print(len(new_features))

   
num_pipeline1 = Pipeline([
            ('selector', DataFrameSelector(new_features)),
            ('imputer', SimpleImputer(strategy='mean')),
            ('std_scaler', StandardScaler()),
        ])

data_prep_pipeline1 = FeatureUnion(transformer_list=[
            ("num_pipeline", num_pipeline1),
            ("cat_pipeline", cat_pipeline),
        ])
selected_features = new_features + cat_features + ["SK_ID_CURR"]


rf_full_pipeline_1 = Pipeline([
    ("preparation", data_prep_pipeline1),
    ("xgboost", XGBClassifier(random_state=42, 
                objective='binary:logistic', max_depth=3, eta=0.1, 
                learning_rate=0.1, colsample_bytree=0.7, n_estimators=500))])


# Training the model
model = rf_full_pipeline_1.fit( X_train_sampled, y_train_sampled)

# Results
exp_name = f"XGboost_Best{len(selected_features)}_features > 0"
experimentLog = get_results(experimentLog, exp_name, model, X_train_sampled, y_train_sampled, X_valid_sampled, y_valid_sampled, X_test_sampled, y_test_sampled)
experimentLog


# In[118]:


features = features_dict['features']
importances = features_dict['importances']

new_indices = [idx for idx, x in enumerate(importances) if x>0]
new_importances = [x for idx, x in enumerate(importances) if x>0]

new_features = [features[i] for i in new_indices]
print(len(new_features))

num_pipeline1 = Pipeline([
            ('selector', DataFrameSelector(new_features)),
            ('imputer', SimpleImputer(strategy='mean')),
            ('std_scaler', StandardScaler()),
        ])

data_prep_pipeline1 = FeatureUnion(transformer_list=[
            ("num_pipeline", num_pipeline1),
            ("cat_pipeline", cat_pipeline),
        ])
selected_features = new_features + cat_features + ["SK_ID_CURR"]


rf_full_pipeline_1 = Pipeline([
    ("preparation", data_prep_pipeline1),
       ("random forest", RandomForestClassifier(random_state=42, bootstrap=False, max_depth=10, 
                                                min_samples_leaf=1, min_samples_split=10, n_estimators=100))])

# Training the model
model = rf_full_pipeline_1.fit( X_train_sampled, y_train_sampled)

# Results
exp_name = f"RandomForest{len(selected_features)}_features > 0"
experimentLog = get_results(experimentLog, exp_name, model, X_train_sampled, y_train_sampled, X_valid_sampled, y_valid_sampled, X_test_sampled, y_test_sampled)
experimentLog


# ## Performance with features with importance > 0.01

# In[119]:


features = features_dict['features']
importances = features_dict['importances']

new_indices = [idx for idx, x in enumerate(importances) if x>0.01]
new_importances = [x for idx, x in enumerate(importances) if x>0.01]

new_features = [features[i] for i in new_indices]
print(len(new_features))

num_pipeline1 = Pipeline([
            ('selector', DataFrameSelector(new_features)),
            ('imputer', SimpleImputer(strategy='mean')),
            ('std_scaler', StandardScaler()),
        ])

data_prep_pipeline1 = FeatureUnion(transformer_list=[
            ("num_pipeline", num_pipeline1),
            ("cat_pipeline", cat_pipeline),
        ])

selected_features = new_features + cat_features + ["SK_ID_CURR"]

rf_full_pipeline_1 = Pipeline([
    ("preparation", data_prep_pipeline1),
    ("xgboost", XGBClassifier(random_state=42, 
                objective='binary:logistic', max_depth=3, eta=0.1, 
                learning_rate=0.1, colsample_bytree=0.7, n_estimators=500))])


# Training the model
model = rf_full_pipeline_1.fit( X_train_sampled, y_train_sampled)

# Results
exp_name = f"XGboost_Best{len(selected_features)}_features > 0.01"
experimentLog = get_results(experimentLog, exp_name, model, X_train_sampled, y_train_sampled, X_valid_sampled, y_valid_sampled, X_test_sampled, y_test_sampled)
experimentLog


# In[120]:


features = features_dict['features']
importances = features_dict['importances']

new_indices = [idx for idx, x in enumerate(importances) if x>0.01]
new_importances = [x for idx, x in enumerate(importances) if x>0.01]

new_features = [features[i] for i in new_indices]
print(len(new_features))

num_pipeline1 = Pipeline([
            ('selector', DataFrameSelector(new_features)),
            ('imputer', SimpleImputer(strategy='mean')),
            ('std_scaler', StandardScaler()),
        ])

data_prep_pipeline1 = FeatureUnion(transformer_list=[
            ("num_pipeline", num_pipeline1),
            ("cat_pipeline", cat_pipeline),
        ])
selected_features = new_features + cat_features + ["SK_ID_CURR"]

rf_full_pipeline_1 = Pipeline([
    ("preparation", data_prep_pipeline1),
       ("random forest", RandomForestClassifier(random_state=42, bootstrap=False, max_depth=10, 
                                                min_samples_leaf=1, min_samples_split=10, n_estimators=100))])

# Training the model
model = rf_full_pipeline_1.fit( X_train_sampled, y_train_sampled)

# Results
exp_name = f"RandomForest{len(selected_features)}_features > 0.01"
experimentLog = get_results(experimentLog, exp_name, model, X_train_sampled, y_train_sampled, X_valid_sampled, y_valid_sampled, X_test_sampled, y_test_sampled)
experimentLog


# In[121]:


features = features_dict['features']
importances = features_dict['importances']

new_indices = [idx for idx, x in enumerate(importances) if x>0.1]
new_importances = [x for idx, x in enumerate(importances) if x>0.1]

new_features = [features[i] for i in new_indices]
print(len(new_features))

num_pipeline1 = Pipeline([
            ('selector', DataFrameSelector(new_features)),
            ('imputer', SimpleImputer(strategy='mean')),
            ('std_scaler', StandardScaler()),
        ])

data_prep_pipeline1 = FeatureUnion(transformer_list=[
            ("num_pipeline", num_pipeline1),
            ("cat_pipeline", cat_pipeline),
        ])
selected_features = new_features + cat_features + ["SK_ID_CURR"]

rf_full_pipeline_1 = Pipeline([
    ("preparation", data_prep_pipeline1),
       ("random forest", RandomForestClassifier(random_state=42, bootstrap=False, max_depth=10, 
                                                min_samples_leaf=1, min_samples_split=10, n_estimators=100))])

# Training the model
model = rf_full_pipeline_1.fit( X_train_sampled, y_train_sampled)

# Results
exp_name = f"RandomForest{len(selected_features)}_features > 0.1"
experimentLog = get_results(experimentLog, exp_name, model, X_train_sampled, y_train_sampled, X_valid_sampled, y_valid_sampled, X_test_sampled, y_test_sampled)
experimentLog


# ## Neural Networks 
# 

# In[58]:


import torch.nn as nn
import torch
import tensorflow as tf
import torch.nn as nn
import torch.nn.functional as func
from torch.nn.functional import binary_cross_entropy
import torch.optim as optim
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss
from torch.optim import lr_scheduler


# ## Re-classifying the dataset to fit and train the NN models

# In[59]:


with open("features.pickle", 'rb') as handle:
    features_dict = pickle.load(handle)

features = features_dict['features']
importances = features_dict['importances']

# selecting features with importance values > 0.005
new_indices = [idx for idx, x in enumerate(importances) if x>0]
new_importances = [x for idx, x in enumerate(importances) if x>0]

new_features = [features[i] for i in new_indices]
# print(len(new_features))

# Pipeline
cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_features)),
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(sparse=False, handle_unknown="ignore"))
    ])

num_pipeline = Pipeline([
        ('selector', DataFrameSelector(new_features)),
        ('imputer', SimpleImputer(strategy='mean')),
        ('std_scaler', StandardScaler()),
    ])

data_prep_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline)]
    )  

selected_features = new_features + cat_features + ["SK_ID_CURR"]
tot_features = f"{len(selected_features)}:   Num:{len(new_features)},    Cat:{len(cat_features)}"
np.random.seed(42)



# splitting the dataset into train and test datasets with selected features
y_train_sampled = train_dataset_sampled_2['TARGET']
X_train_sampled = train_dataset_sampled_2[selected_features]
X_train_sampled, X_valid_sampled, y_train_sampled, y_valid_sampled = train_test_split(X_train_sampled, y_train_sampled, test_size=0.17, random_state=42)
X_train_sampled, X_test_sampled, y_train_sampled, y_test_sampled = train_test_split(X_train_sampled, y_train_sampled, test_size=0.17, random_state=42)
X_kaggle_test= X_kaggle_test[selected_features]

print(f"X train sampled           shape: {X_train_sampled.shape}")
print(f"X validation sampled      shape: {X_valid_sampled.shape}")
print(f"X test sampled          shape: {X_test_sampled.shape}")
print(f"X X_kaggle_test   shape: {X_kaggle_test.shape}")

# Handling missing values and standardizing the data
X_train_std = data_prep_pipeline.fit_transform(X_train_sampled)
X_valid_std = data_prep_pipeline.transform(X_valid_sampled)
X_test_std = data_prep_pipeline.transform(X_test_sampled)
X_kaggle_test_std = data_prep_pipeline.transform(X_kaggle_test)


X_train_tensor = torch.from_numpy(X_train_std)
X_valid_tensor = torch.from_numpy(X_valid_std)
X_test_tensor = torch.from_numpy(X_test_std)
X_kaggle_test_tensor = torch.from_numpy(X_kaggle_test_std)

y_train_tensor =  torch.from_numpy(y_train_sampled.to_numpy())
y_valid_tensor =  torch.from_numpy(y_valid_sampled.to_numpy())
y_test_tensor = torch.from_numpy(y_test_sampled.to_numpy())


# In[60]:


# create TensorDataset in PyTorch
hcdr_train = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
hcdr_valid = torch.utils.data.TensorDataset(X_valid_tensor, y_valid_tensor)
hcdr_test = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)

# print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
# create dataloader
# DataLoader is implemented in PyTorch, which will return an iterator to iterate training data by batch.
train_batch_size = 96
valid_test_batch_size = 64
trainloader_hcdr = torch.utils.data.DataLoader(hcdr_train, batch_size=train_batch_size, shuffle=True, num_workers=2)
validloader_hcdr = torch.utils.data.DataLoader(hcdr_valid, batch_size=valid_test_batch_size, shuffle=True, num_workers=2)
testloader_hcdr = torch.utils.data.DataLoader(hcdr_test, batch_size=valid_test_batch_size, shuffle=True, num_workers=2)


# In[32]:


# Method to create, define and run a deep neural network model
#
def run_hcdr_model(
    hidden_layer_neurons=[32, 16, 2],
    opt=optim.SGD,
    epochs=5,
    learning_rate=1e-3
):
    
    D_in = X_test_std.shape[1]  # Input layer neurons depend on the input dataset shape
    D_out = 2  # Output layer neurons - depend on what you're trying to predict, here, 2 classes: 0 and 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    str_neurons = [str(h) for h in hidden_layer_neurons]
    arch_string = f"{D_in}-{'-'.join(str_neurons)}-{D_out}"
    
    layers = [
        torch.nn.Linear(D_in, hidden_layer_neurons[0]),  # X.matmul(W1)
        nn.ReLU()  # ReLU( X.matmul(W1))
    ]
    
    # Add hidden layers
    for i in range(1, len(hidden_layer_neurons)):
        prev, curr = hidden_layer_neurons[i - 1], hidden_layer_neurons[i]
        layers.append(torch.nn.Linear(prev, curr))
        layers.append(nn.ReLU())
  
        
    
    # Add final layer
    layers.append(nn.Linear(hidden_layer_neurons[len(hidden_layer_neurons)-1], D_out)) # Relu( X.matmul(W1)).matmul(W2))
    
    # Use the nn package to define our model and loss function.
    # use the sequential API makes things simple
    model = torch.nn.Sequential(*layers)
    device
    model.to(device)

   
    loss_fn = nn.CrossEntropyLoss()  #for classfication 
    optimizer = opt(model.parameters(), lr=learning_rate)

    #summary(model, (4, 20))
    print('-'*50)
    print('Model:')
    print(model)
    print('-'*50)
    
    '''
    Training Process:
        Load a batch of data.
        Zero the grad.
        Predict the batch of the data through net i.e forward pass.
        Calculate the loss value by predict value and true value.
        Backprop i.e get the gradient with respect to parameters
        Update optimizer i.e gradient update
    '''

    loss_history = []
    acc_history = []
    def train_epoch(epoch, model, loss_fn, opt, train_loader):
        running_loss = 0.0
        count = 0
        y_pred = []
        epoch_target = []
        
        for batch_id, data in enumerate(train_loader):
            inputs, target = data[0].to(device), data[1].to(device)        
            # 1:zero the grad, 2:forward pass, 3:calculate loss,  and 4:backprop!
            opt.zero_grad()
            preds = model(inputs.float()) #prediction over the input data

            # compute loss and gradients
            loss = loss_fn(preds, target)    #mean loss for this batch

            loss.backward() #calculate nabla_w
            loss_history.append(loss.item())
            opt.step()  #update W
            y_pred.extend(torch.argmax(preds, dim=1).tolist())
            epoch_target.extend(target.tolist())
            #from IPython.core.debugger import Pdb as pdb;    pdb().set_trace() #breakpoint; dont forget to quit

            running_loss += loss.item()
            count += 1

        loss = np.round(running_loss/count, 3)
        
        #accuracy
        correct = (np.array(y_pred) == np.array(epoch_target))
        accuracy = correct.sum() / correct.size
        accuracy = np.round(accuracy, 3)
        return loss, accuracy



    #from IPython.core.debugger import Pdb as pdb;    pdb().set_trace() #breakpoint; dont forget to quit
    def evaluate_model(epoch, model, loss_fn, opt, data_loader, tag = "Test"):
        overall_loss = 0.0
        count = 0
        y_pred = []
        epoch_target = []
        for i,data in enumerate(data_loader):
            inputs, target = data[0].to(device), data[1].to(device)                
            preds = model(inputs.float())      

            loss = loss_fn(preds, target)           # compute loss value

            overall_loss += (loss.item())  # compute total loss to save to logs
            y_pred.extend(torch.argmax(preds, dim=1).tolist())
            epoch_target.extend(target.tolist())
            count += 1

        # compute mean loss
        loss = np.round(overall_loss/count, 3)
        #accuracy
        correct = (np.array(y_pred) == np.array(epoch_target))
        accuracy = correct.sum() / correct.size
        accuracy = np.round(accuracy, 3)
        return loss, accuracy
        


    for epoch in range(epochs):
        # print(f"Epoch {epoch+1}")
        train_loss, train_accuracy = train_epoch(epoch, model, loss_fn, optimizer, trainloader_hcdr)
        valid_loss, valid_accuracy = evaluate_model(epoch, model, loss_fn, optimizer, validloader_hcdr, tag = "Validation")
        print(f"Epoch {epoch+1}: Train Accuracy: {train_accuracy}\t Validation Accuracy: {valid_accuracy}")
    print("-"*50)
    test_loss, test_accuracy = evaluate_model(epoch, model, loss_fn, opt, testloader_hcdr, tag="Test")
    
    return arch_string, train_accuracy, valid_accuracy, test_accuracy


# In[61]:


#
# NOTE: Run this cell however number of times you want to achieve larger train/test accuracy
# Experiment with different arguments to the function
#

import pandas as pd
torch.manual_seed(0)
#==================================================#
#    Modify START   #
#==================================================#
'''
(hidden_layers_neurons) - A list of the number of neurons in the hidden layers in order. DEFAULT: [32, 16, 8] => 1st hidden layer: 32 neurons, 2nd: 16, 3rd: 8
(opt) - The optimizer function to use: SGD, Adam, etc.,  DEFAULT: optim.SGD
(epochs) - The total number of epochs to train your model for,  DEFAULT: 5
(learning_rate) - The learning rate to take the gradient descent step with
'''

hidden_layer_neurons = [32,12,4]
opt = optim.Adam  # optim.SGD, Optim.Adam, etc.
epochs = 7
learning_rate = 1e-2

#==================================================#
#    Modify END #
#==================================================#

arch_string, train_accuracy, valid_accuracy, test_accuracy = run_hcdr_model(
    hidden_layer_neurons,
    opt,
    epochs,
    learning_rate
)

try: hcdrLog 
except : hcdrLog = pd.DataFrame(
    columns=[
        "Architecture string", 
        "Optimizer", 
        "Epochs", 
        "Train accuracy",
        "Valid accuracy",
        "Test accuracy",
    ]
)

hcdrLog.loc[len(hcdrLog)] = [
    arch_string, 
    f"{opt}", 
    f"{epochs}", 
    f"{train_accuracy * 100}%",
    f"{valid_accuracy * 100}%",
    f"{test_accuracy * 100}%",
]

hcdrLog   


# ## MLP model Evaluation

# In[62]:


get_ipython().system('pip install -q pytorch-lightning')


# In[63]:


import pytorch_lightning as pl
import torch 
import torch.nn as nn 

from torchmetrics import Accuracy


# In[180]:


class MultiLayerPerceptron(pl.LightningModule):
    def __init__(self, hidden_units=(32, 16)):
        super().__init__()
        
        # new PL attributes:
        self.train_acc = Accuracy("binary", num_classes=2)  #change to binary class and 2 for HCDR
        self.valid_acc = Accuracy("binary", num_classes=2)
        self.test_acc = Accuracy("binary",num_classes=2)
        
     
        input_size = 293
        all_layers = [nn.Flatten()]
        for hidden_unit in hidden_units: 
            layer = nn.Linear(input_size, hidden_unit) 
            all_layers.append(layer) 
            all_layers.append(nn.ReLU()) 
            input_size = hidden_unit 
 
        all_layers.append(nn.Linear(hidden_units[-1], 2)) 
        self.model = nn.Sequential(*all_layers)

    def forward(self, x):
        x = self.model(x.float())
        return x

    def training_step(self, batch, batch_idx):
        # add in a loss layer
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.train_acc.update(preds, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def training_epoch_end(self, outs):
        self.log("train_acc", self.train_acc.compute())
        self.train_acc.reset()
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.valid_acc.update(preds, y)
        self.log("valid_loss", loss, prog_bar=True)
        return loss
    
    def validation_epoch_end(self, outs):
        self.log("valid_acc", self.valid_acc.compute(), prog_bar=True)
        self.valid_acc.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.test_acc.update(preds, y)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_acc.compute(), prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        return optimizer


# In[175]:


class MLP(pl.LightningModule):
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        super().__init__()
        
        self.learning_rate = learning_rate
        
        self.hidden_size = hidden_size
        
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        
        self.loss = nn.CrossEntropyLoss()
        
        self.train_accuracy =Accuracy("binary", num_classes=2)  #change to binary class and 2 for HCDR
        self.valid_accuracy = Accuracy("binary", num_classes=2)
        self.test_accuracy = Accuracy("binary",num_classes=2)
        
    def forward(self, x):
        x = self.model(x.float())
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        acc = self.train_accuracy(y_hat.argmax(dim=1), y)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_accuracy', acc, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        acc = self.valid_accuracy(y_hat.argmax(dim=1), y)
        self.log('valid_loss', loss, prog_bar=True)
        self.log('valid_accuracy', acc, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        acc = self.test_accuracy(y_hat.argmax(dim=1), y)
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_accuracy', acc, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


# In[165]:


from torch.utils.data import DataLoader
from torch.utils.data import random_split
 
from torchvision.datasets import MNIST
from torchvision import transforms


# In[ ]:





# In[177]:


#hcdr_train = torch.utils.data.TensorDataset(X_train_tensor.float(), y_train_tensor.float())
#hcdr_valid = torch.utils.data.TensorDataset(X_valid_tensor.float(), y_valid_tensor.float())
#hcdr_test = torch.utils.data.TensorDataset(X_test_tensor.float(), y_test_tensor.float())

hcdr_train = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
hcdr_valid = torch.utils.data.TensorDataset(X_valid_tensor, y_valid_tensor)
hcdr_test = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)


#hcdr_train = torch.utils.data.TensorDataset(X_train_tensor.long(), y_train_tensor.long())
#hcdr_valid = torch.utils.data.TensorDataset(X_valid_tensor.long(), y_valid_tensor.long())
#hcdr_test = torch.utils.data.TensorDataset(X_test_tensor.long(), y_test_tensor.long())

train_batch_size = 96
valid_test_batch_size = 96
trainloader_hcdr = torch.utils.data.DataLoader(hcdr_train, batch_size=train_batch_size, shuffle=True, num_workers=2)
validloader_hcdr = torch.utils.data.DataLoader(hcdr_valid, batch_size=valid_test_batch_size, shuffle=True, num_workers=2)
testloader_hcdr = torch.utils.data.DataLoader(hcdr_test, batch_size=valid_test_batch_size, shuffle=True, num_workers=2)


# In[181]:


import pandas as pd

class HCDR_DataModule(pl.LightningDataModule):
    """Sample Data module for HCDR
    Load data from CSV and convert into a numpy array, 
    then into Pytorch Tensor 
    and then into a Tensor Dataset
    """
    def __init__(self,  batchsize = 96):
        super().__init__()
        self.batchsize = batchsize
        self.transform = transforms.Compose([transforms.ToTensor()])
        
    def prepare_data(self):
        pass 

    def setup(self, stage=None):
        # stage is either 'fit', 'validate', 'test', or 'predict'
        # here note relevant
        self.train = hcdr_train
        self.val = hcdr_valid
        self.test = hcdr_test
        # df = pd.read_pickle(self.data_path + "XY_valid_small.pkl")
        # self.val = torch.utils.data.TensorDataset(df.iloc[:,:-1].values, df.iloc[:, -1].values.astype('float'))
        # df = pd.read_pickle(self.data_path + "XY_test_small.pkl")
        # self.test = torch.utils.data.TensorDataset(df.iloc[:,:-1].values, df.iloc[:, -1].values.astype('float'))


    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batchsize, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batchsize, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batchsize, num_workers=4)
    
    
torch.manual_seed(1) 


# In[179]:


input_value=293
model = MLP(input_size=input_value, hidden_size=32, output_size=2,learning_rate=0.01)
trainer = pl.Trainer(max_epochs=7)

# Fit model
trainer.fit(model,train_dataloaders=trainloader_hcdr,val_dataloaders=validloader_hcdr)

# Evaluate on test set
trainer.test(model, testloader_hcdr)


# In[182]:


from pytorch_lightning.callbacks import ModelCheckpoint


mnistclassifier = MultiLayerPerceptron()

callbacks = [ModelCheckpoint(save_top_k=1, mode='max', monitor="valid_acc")] # save top 1 model

if torch.cuda.is_available(): # if you have GPUs
    print("GPU is available")
    trainer = pl.Trainer(max_epochs=10, callbacks=callbacks, gpus=1)
else:
    trainer = pl.Trainer(max_epochs=10, callbacks=callbacks)


trainer.fit(model=mnistclassifier, datamodule=HCDR_DataModule())


# In[183]:


trainer.test(model=mnistclassifier, datamodule=HCDR_DataModule(), ckpt_path='best')


# In[ ]:





# In[ ]:





# ## Submission File Prep
# 
# For each SK_ID_CURR in the test set, you must predict a probability for the TARGET variable. The file should contain a header and have the following format:
# 
# ```python 
# SK_ID_CURR,TARGET
# 100001,0.1
# 100005,0.9
# 100013,0.2
# etc.
# ```

# In[ ]:


X_kaggle_test


# In[ ]:


test_scores = model.predict_proba(X_kaggle_test)[:, 1]


# In[ ]:


test_scores[0:10]


# In[ ]:


# Submission dataframe
submit_df = application_test[['SK_ID_CURR']]
submit_df['TARGET'] = test_scores
submit_df.head()


# In[ ]:


submit_df.to_csv("submission.csv",index=False)


# In[47]:


fig = Image(filename='./Kaggle Submission Code.png')
fig


# In[48]:


fig = Image(filename='./Kaggle Final Submission.png')
fig

