# CODSOFT
Data Science Internship
# Objectives
1- Build a machine learning model to identify fraudulent credit card transactions.

2- Preprocess and normalize the transaction data, handle class imbalance issues, and split the dataset into training and testing sets.

3- Train a classification algorithm, such as logistic regression or random forests, to classify transactions as fraudulent or genuine.

4- Evaluate the model's performance using metrics like precision, recall, and F1-score, and consider techniques like oversampling or undersampling for improving results.

# Data Preprocessing 
## Import Libraries
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```
## Load Dataset
```python
Credit_dataset = pd.read_csv('/content/creditcard.csv', on_bad_lines='skip')
```
## Get some info and description about the dataset
```python
Credit_dataset.head()
Credit_dataset.describe()
Credit_dataset.info()
```
## Check for Null Values
```python
print(Credit_dataset.isnull().sum())
```
