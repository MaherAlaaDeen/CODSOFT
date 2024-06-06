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
## Filling the missing values with 0 in the Class column
```python
Credit_dataset['Class'] = Credit_dataset['Class'].fillna(0)
```
## Handling Missing Values
```python
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values= np.nan, strategy = 'mean')
Credit_dataset = imputer.fit_transform(Credit_dataset)
```
## Creating the matrix of features and the dependent variable
```python
X = Credit_dataset.iloc[:, :-1].values
y = Credit_dataset.iloc[:, -1].values
print('Matrix of Features: ', X)
print('Dependent Variable: ', y)
```
## Feature Engineering
### Normalization
```python
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
X = sc.fit_transform(X)
X
```
- Since the Features is normally distrubted along 0 ==> Normalization
## Splitting the dataset into train and test
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1, stratify = y)
print('X_train: ', X_train)
print('X_test: ', X_test)
print('y_train: ', y_train)
print('y_test: ', y_test)
```
## Oversampling
- Oversampling is increasing the number of instances in the minority class.
- Balance the class distribution by generating synthetic data.
Oversampling is done after splitting the dataset in order to:
- Prevent Information Leakage
- Improves Model Generalization

```python
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler()
# Perform oversampling on your training data
ros_X_train, ros_y_train = ros.fit_resample(X_train, y_train)
```
### Class distribution before and after Oversampling
```python
from collections import Counter
print('before sampling class distribution: ', Counter(y_train))
print('after sampling class distribution: ', Counter(ros_y_train))
```
before sampling class distribution:  Counter({0.0: 6948, 1.0: 27})

after sampling class distribution:  Counter({0.0: 6948, 1.0: 6948})

# Machine Learning
## Choose between the following models
### import libraries
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```
### Define classifiers
```python
classifiers = {
    "Random Forest Classifier" : RandomForestClassifier(n_estimators=100, random_state=42),
    "K-Nearest Neighbors Classifier" : KNeighborsClassifier(n_neighbors=5),
    "Support Vector Machine Classifier" : SVC(kernel='rbf', random_state = 42),
    "Logistic Regression": LogisticRegression(random_state=42)
}

accuracies = []
```
### Loop through classifiers: Get the best model with the best accuracy
```python
for name, classifier in classifiers.items():
  classifier.fit(ros_X_train, ros_y_train)
  y_pred = classifier.predict(X_test)
  accuracy = accuracy_score(y_test, y_pred)
  accuracies.append(accuracy)
  print(f"{name} Accuracy: {accuracy}")
```
- Random Forest Classifier Accuracy: 0.9989966555183947
- K-Nearest Neighbors Classifier Accuracy: 0.9986622073578595
- Support Vector Machine Classifier Accuracy: 0.9986622073578595
- Logistic Regression Accuracy: 0.9979933110367893

### Plotting the Accuracies
```python
fig, ax = plt.subplots()
models = classifiers.keys()
y_pos = np.arange(len(models))
ax.barh(y_pos, accuracies, align = 'center')
ax.set_yticks(y_pos)
ax.set_yticklabels(models)
ax.invert_yaxis()
ax.set_xlabel('Accuracy')
ax.set_title('Classifier Accuracies')
plt.show()
```
![Accuracy plot](accuracies.png)

### Find the best model
```python
best_accuracy = max(accuracies)
best_model = list(classifiers.keys())[accuracies.index(best_accuracy)]
print(f"\nBest Model: {best_model} with Accuracy: {best_accuracy}")
```
- Best Model: Random Forest Classifier with Accuracy: 0.9989966555183947

### Train the Model on the dataset
```python
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, random_state = 42)
classifier.fit(ros_X_train, ros_y_train)
```
### Predict on the test set
y_pred = classifier.predict(X_test)

### Evaluate the Model
```python
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report
#computer accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)

#computer confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix: ", cm)

#computer precision
precision = precision_score(y_test, y_pred, average = 'weighted')
print("precision: ", precision)


#computer recall
recall = recall_score(y_test, y_pred, average = 'weighted')
print("recall: ", recall)

#computer F1 Score
f1 = f1_score(y_test, y_pred, average = 'weighted')
print("F1 Score: ", f1)

# generate Classfication Report
class_report = classification_report(y_test, y_pred)
print("Classification Report:\n", class_report)
```
- Accuracy:  0.9989966555183947
- Confusion Matrix:  [[2979    0]
 [   3    8]]
- precision:  0.9989976649192817
- recall:  0.9989966555183947
- F1 Score:  0.9989176965891743






