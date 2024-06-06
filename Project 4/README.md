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
![Null Values](null.png)

## Check for Missing Values
```python
Credit_dataset.isna().sum()
```
![Missing Data](missing.png)

# Visualization
## Plotting a Heatmap
```python
numeric_columns = Credit_dataset.select_dtypes(include=["int64", "float64"])
import seaborn as sns
sns.heatmap(numeric_columns.corr(), cmap = "YlGnBu")
plt.show()
```
![heatmap](heatmap.png)
## Checking the 'Class' Count
```python
Credit_dataset['Class'].value_counts()
```
![Count](ClassCount.png)

## Plotting a bar chart to Visualize the 'Class' Count
```python
Class_counts = Credit_dataset['Class'].value_counts()

#plot the bar chart
plt.figure(figsize=(10, 8))
sns.barplot(x=Class_counts.index, y=Class_counts.values, color='red')
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Count of Each Class in Iris Dataset')
plt.show()
```
![Class Count](imbalance.png)

## Plotting the distribution of Features
```python
Credit_dataset.hist(bins=30, figsize=(30, 30))
```
![Distribution](distribution.png)
- From the Above figures and data we can deduce the following:
- - An imbalance occurs in the 'Class' column
  - We have to handle missing values
  - We have to normalize the features
## 
