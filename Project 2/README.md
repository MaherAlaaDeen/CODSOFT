# CODSOFT
Data Science Internship
# Project 2 : Sales Prediction Using Python
# Objectives:
1- Sales prediction involves forecasting the amount of a product that customers will purchase, taking into account various factors such as advertising expenditure, target audience segmentation, and advertising platform selection

2- In businesses that offer products or services, the role of a Data Scientist is crucial for predicting future sales. They utilize machine learning techniques in Python to analyze and interpret data, allowing them to make informed decisions regarding advertising costs. By leveraging these predictions, businesses can optimize their advertising strategies and maximize sales potential. Let's embark on
the journey of sales prediction using machine learning in Python.

# Data Preprocessing
## Import libraries
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```
## Load the Dataset
```python
Sales_dataset = pd.read_csv('/content/advertising.csv')
```
## Get some info and description about the Dataset
```python
Sales_dataset.head()
```
![Data Head](head.png)

```python
Sales_dataset.describe()
```
![Data Described](described.png)

```python
Sales_dataset.info()
```
![data info](info.png)
- From the given observation we can deduce that there is no missing data to handle.
- Moreover there is no need to categorical data to be encoded
# Plot a Heatmap: Understand the features and identify the relation between features and Sales
```python
import seaborn as sns
sns.heatmap(Sales_dataset.corr(), cmap = 'YlGnBu')
```
![Heatmap](Heatmap.png)
- From the Correlation Heatmap, we can deduce that 'TV' advertisement resulted in the most sales.
# Scatter plot for TV ads vs Sales
```python
plt.figure(figsize=(10, 6))  # Adjust size as needed
plt.scatter(Sales_dataset['TV'], Sales_dataset['Sales'], color='blue', alpha=0.5) 
plt.title('TV Advertising vs. Sales')
plt.xlabel('TV Advertising ')
plt.ylabel('Sales')
plt.grid(True)
plt.show()
```
![TV vs Sales](TVSales.png)

# Scatter plot for Radio ads vs Sales
```python
plt.figure(figsize=(10, 6))  # Adjust size as needed
plt.scatter(Sales_dataset['Radio'], Sales_dataset['Sales'], color='blue', alpha=0.5) 
plt.title('Radio Advertising vs. Sales')
plt.xlabel('Radio Advertising ')
plt.ylabel('Sales')
plt.grid(True)
plt.show()
```
![radio vs sales](RadioSales.png)
# Scatter plot for Newspaper ads vs Sales
```python
plt.figure(figsize=(10, 6))  # Adjust size as needed
plt.scatter(Sales_dataset['Newspaper'], Sales_dataset['Sales'], color='blue', alpha=0.5) 
plt.title('Newspaper Advertising vs. Sales')
plt.xlabel('Newspaper Advertising ')
plt.ylabel('Sales')
plt.grid(True)
plt.show()
```
![Newspaper vs Sales](NewspaperSales.png)

