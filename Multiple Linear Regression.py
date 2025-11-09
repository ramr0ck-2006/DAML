# Prog 6: Multiple Linear Regression (MLR)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
iris = pd.read_csv(r"D:\Demo\IRIS.csv")
print(iris.head())

# Independent variables
x = iris[['SepalLengthCm', 'SepalWidthCm']]

# Dependent variable
y = iris['PetalLengthCm']

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Model training
LR = LinearRegression()
LR.fit(x_train, y_train)

# Prediction
y_pred = LR.predict(x_test)

print("Predicted values:")
print(y_pred)
