# CODSOFT
Data Science Internship
# Project 1 : Titanic Survival Prediction
# Objectives: 
1-Use the Titanic dataset to build a model that predicts whether a
passenger on the Titanic survived or not. This is a classic beginner
project with readily available data.

2-The dataset typically used for this project contains information
about individual passengers, such as their age, gender, ticket
class, fare, cabin, and whether or not they survived.

# Data Preprocessing
## Import libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```
## Load the Dataset
```python
titanic_dataset = pd.read_csv('/content/Titanic-Dataset.csv')
```
## Get some Info and description about the dataset
```python
print(titanic_dataset.describe())
print(titanic_dataset.info())
print(titanic_dataset.head())
print(titanic_dataset.tail())
```
