import numpy as np
import matplotlib
matplotlib.use('Agg') # This is only required to use matplotlib on Cloud9
import matplotlib.pyplot as plt
import pandas as pd

# import model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
# import module to calculate model perfomance metrics
from sklearn import metrics

data = pd.read_csv("height-length.csv")

x = data["hand_length"]
y = data["height"]

# Splitting X and y into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)

# We need to reshape our x array since it has only on feature 
# (sci-kit learn expects more than one feature)
x_train = x_train.values.reshape(-1, 1)
x_test = x_test.values.reshape(-1, 1)

# Linear Regression Model
linreg = LinearRegression()

# Fit the model to the training data (learn the coefficients)
linreg.fit(x_train, y_train)

# make predictions on the testing set
y_pred = linreg.predict(x_test)
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# compute the RMSE of our predictions
your_hand_length = input("What is your hand length in inches?\n")
your_height = linreg.predict(np.array(float(your_hand_length)).reshape(1,-1))[0]
print("Your predicted height is: {:.2f} in, {:.2f} ft".format(your_height, your_height/12))