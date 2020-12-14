# Making the imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (12.0, 9.0)

# Preprocessing Input data
data = pd.read_csv('data.csv')
X = data.iloc[:, 0]
Y = data.iloc[:, 1]
print(X)
print(Y)
# plt.scatter(X, Y)
# plt.show()

#X = np.array([2,4,5])
#Y = np.array([1.2, 2.8, 5.3])
#plt.scatter(X,Y)
#plt.show()

b0 = 0 #intercept
b1 = 1 #slope
lr = 0.001 #learning rate
iterations = 100 #number of iterations
error = [] #error array to calc cost for each iterations

for itr in range(iterations):
    error_cost = 0
    cost_b0 = 0
    cost_b1 = 0

    for i in range(len(X)):
        y_pred = (b0 + b1 * X[i]) #Predict the value for given X

        error_cost = error_cost + (Y[i] - y_pred) ** 2 #calculate the error in prediction for all points

        for j in range(len(X)):
            partial_wrt_b0 = -2 * (Y[j] - (b0 + b1 * X[j])) #partial derivative 1
            partial_wrt_b1 = (-2 * X[j]) * (Y[j] - (b0 + b1 * X[j])) #partial derivative 2

            cost_b0 = cost_b0 + partial_wrt_b0 #calc cost for each number
            cost_b1 = cost_b1 + partial_wrt_b1 #calc cost for each number
        
        b0 = b0 - lr * cost_b0 #update values
        b1 = b1 - lr * cost_b1 #update values
    
    error.append(error_cost)

print(b0, b1);
y_pred = b0 + b1 * X

plt.scatter(X,Y)
plt.plot(X, y_pred)
plt.show()

y_new_pred = b0 + b1 * 3
print(y_new_pred)

# # Building the model
# m = 0
# c = 0

# L = 0.0001  # The learning Rate
# epochs = 1000  # The number of iterations to perform gradient descent

# n = float(len(X)) # Number of elements in X

# # Performing Gradient Descent 
# for i in range(epochs): 
#     Y_pred = m*X + c  # The current predicted value of Y
#     D_m = (-2/n) * sum(X * (Y - Y_pred))  # Derivative wrt m
#     D_c = (-2/n) * sum(Y - Y_pred)  # Derivative wrt c
#     m = m - L * D_m  # Update m
#     c = c - L * D_c  # Update c
    
# print (m, c)

# # Making predictions
# Y_pred = m*X + c

# plt.scatter(X, Y) 
# plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color='red')  # regression line
# plt.show()