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
