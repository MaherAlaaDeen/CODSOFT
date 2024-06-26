# -*- coding: utf-8 -*-
"""Sales_Prediction.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1nekwtUqLlw9i1U3T0oV3Z2zDucxlM4mg
"""

#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import dataset
Sales_dataset = pd.read_csv('/content/advertising.csv')

Sales_dataset.head()

Sales_dataset.describe()

Sales_dataset.info()

import seaborn as sns
sns.heatmap(Sales_dataset.corr(), cmap = 'YlGnBu')
##plt.show()

#TV vs Sales

plt.figure(figsize=(10, 6))  # Adjust size as needed
plt.scatter(Sales_dataset['TV'], Sales_dataset['Sales'], color='blue', alpha=0.5)
plt.title('TV Advertising vs. Sales')
plt.xlabel('TV Advertising ')
plt.ylabel('Sales')
plt.grid(True)
plt.show()

#Radio vs Sales
plt.figure(figsize=(10, 6))  # Adjust size as needed
plt.scatter(Sales_dataset['Radio'], Sales_dataset['Sales'], color='blue', alpha=0.5)
plt.title('Radio Advertising vs. Sales')
plt.xlabel('Radio Advertising ')
plt.ylabel('Sales')
plt.grid(True)
plt.show()

#Newspaper vs Sales
plt.figure(figsize=(10, 6))  # Adjust size as needed
plt.scatter(Sales_dataset['Newspaper'], Sales_dataset['Sales'], color='blue', alpha=0.5)
plt.title('Newspaper Advertising vs. Sales')
plt.xlabel('Newspaper Advertising ')
plt.ylabel('Sales')
plt.grid(True)
plt.show()

media_channels = ['TV', 'Radio', 'Newspaper']
total_sales = [Sales_dataset['TV'].sum(), Sales_dataset['Radio'].sum(), Sales_dataset['Newspaper'].sum()]

plt.bar(media_channels, total_sales, color='skyblue')
plt.xlabel('Media Channel')
plt.ylabel('Total Sales')
plt.title('Total Sales by Media Channel')
plt.show()

X = Sales_dataset.iloc[:, :-1].values
y = Sales_dataset.iloc[:, -1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state = 1)
print('x_train: ', X_train)
print('x_test: ', X_test)
print('y_train: ', y_train)
print('y_test: ', y_test)

#feature Scaling
#Standardization
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print('Featured Scaled x-train:\n ',X_train)
print('Featured Scaled x-test:\n ',X_test)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

from sklearn.metrics import r2_score, mean_squared_error
# Calculate R-squared
r2 = r2_score(y_test, y_pred)
# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("R-squared:", r2)
print("RMSE:", rmse)

print('regressor coefficient: ', regressor.coef_)
print('regressor intercept: ',regressor.intercept_)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

# Generate predicted values using the trained model
predicted_values = regressor.coef_[0] * X_train[:, 0] + regressor.intercept_

# Plot the training dataset and the trained model
plt.figure(figsize=(8, 6))
plt.scatter(X_train[:, 0], y_train, color='blue', label='Training Data')
plt.plot(X_train[:, 0], predicted_values, color='red', label='Trained Model')
plt.title('Trained Model and Training Dataset')
plt.xlabel('Independent Variable 1')
plt.ylabel('Target Variable')
plt.legend()
plt.grid(True)
plt.show()

# Generate predicted values using the trained model
predicted_values = regressor.coef_[1] * X_train[:, 1] + regressor.intercept_

# Plot the training dataset and the trained model
plt.figure(figsize=(8, 6))
plt.scatter(X_train[:, 1], y_train, color='blue', label='Training Data')
plt.plot(X_train[:, 1], predicted_values, color='red', label='Trained Model')
plt.title('Trained Model and Training Dataset')
plt.xlabel('Independent Variable 2')
plt.ylabel('Target Variable')
plt.legend()
plt.grid(True)
plt.show()

# Generate predicted values using the trained model
predicted_values = regressor.coef_[2] * X_train[:, 2] + regressor.intercept_

# Plot the training dataset and the trained model
plt.figure(figsize=(8, 6))
plt.scatter(X_train[:, 2], y_train, color='blue', label='Training Data')
plt.plot(X_train[:, 2], predicted_values, color='red', label='Trained Model')
plt.title('Trained Model and Training Dataset')
plt.xlabel('Independent Variable 3')
plt.ylabel('Target Variable')
plt.legend()
plt.grid(True)
plt.show()

# predicting Sales
# Reshape the input data into a 2D array
input_data = np.array([230.1, 37.8, 69.2]).reshape(1, -1)

input_data_scaled = sc.transform(input_data)

# Make predictions using the reshaped data
predicted_value = regressor.predict(input_data_scaled)

# Print the predicted value
print(predicted_value)

from sklearn.metrics import mean_absolute_percentage_error
mape = mean_absolute_percentage_error(y_test, y_pred) * 100
print(f'Mean Absolute Percentage Error (MAPE): {mape:.2f}%')