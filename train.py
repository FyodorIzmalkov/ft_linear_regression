import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

errorCost = []
thetaValues = []

def makePrediction(theta0, theta1, kilometers):
    return theta0 + theta1 * kilometers

# Sum of Squared Errors (SSE)
def getErrorCost(pricePrediction, len, price):
    return sum(np.square(pricePrediction - price)) / (2 * len)

def getAverage(kms):
    return sum(kms) / len(kms)

def meanNormalization(kms):
    return (kms - getAverage(kms)) / (max(kms) - min(kms))

def meanNormReverse(theta0, theta1, kms):
    diffBetweenMaxAndMin = max(kms) - min(kms)
    theta0 -= theta1 * getAverage(kms) / diffBetweenMaxAndMin
    theta1 /= diffBetweenMaxAndMin
    return theta0, theta1

def performGradientDescent(kilometers, price):
    theta0 = 0
    theta1 = 0
    learningRate = 0.1
    iterations = 2000
    m = len(kilometers)
    oneDividedByM = 1 / m

    for i in range(iterations):
        pricePrediction = makePrediction(theta0, theta1, kilometers)
        tmpTheta0 = learningRate * oneDividedByM * sum(pricePrediction - price)
        tmpTheta1 = learningRate * oneDividedByM * sum((pricePrediction - price) * kilometers)
        theta0 -= tmpTheta0
        theta1 -= tmpTheta1
        errorCost.append(getErrorCost(pricePrediction, m, price))
        if i % 50 == 0:
            thetaValues.append([theta0, theta1])

    return theta0, theta1

def printAverageErrorPercantage(kilometers, price, theta0, theta1):
    prediction = makePrediction(theta0, theta1, kilometers)
    length = len(kilometers)
    averageError = 0.0
    for i in range(length):
        averageError += abs(prediction[i] - price[i]) / price[i]
    print("Average percantage error is: " , averageError / length * 100)

def showErrorCosts(figure):
    numOfIterations = list(range(len(errorCost)))

    if "-line" in sys.argv or "-history" in sys.argv or "-scatter" in sys.argv:
        axes = figure.add_subplot(2, 1, 1)
    else:
        axes = figure.add_subplot(1, 1, 1)

    axes.set_xlabel('Number of iterations')
    axes.set_ylabel('Mean squared error')
    axes.set_title('Cost vs Iteration analysis')
    axes.plot(numOfIterations, errorCost, color='green')


def showOther(figure,x, y, theta0, theta1):
    if "-cost" in sys.argv:
        axes = figure.add_subplot(2, 1, 2)
    else:
        axes = figure.add_subplot(1, 1, 1)

    axes.set_xlabel("kilometers")
    axes.set_ylabel("price")
    axes.set_title('Car price to kilometers prediction')
    axes.scatter(x, y, label='Scatter Plot', color='blue')

    if "-line" in sys.argv:
        axes.plot(x, 
                    makePrediction(theta0, theta1, x),
                    label='Regression Line',
                    color='black')

    if "-history" in sys.argv:
        for theta in thetaValues:
            theta0, theta1 = meanNormReverse(theta[0], theta[1], x)
            axes.plot(x,
                        makePrediction(theta0, theta1, x),
                        color='red')
    
    axes.legend()

def getFigure():
    figure = plt.figure('ft_linear_regression', figsize=(10, 10))
    plt.gcf().subplots_adjust(left=0.08,
                            bottom=0.05,
                            right=0.95,
                            top=0.95,
                            wspace=0,
                            hspace=0.25)
    return figure


def showGrpahs(data, kilometers, price, theta0, theta1):
    figure = getFigure()

    if "-cost" in sys.argv:
        showErrorCosts(figure)
    if "-line" in sys.argv or "-history" in sys.argv or "-scatter" in sys.argv:
        showOther(figure,kilometers, price, theta0, theta1)
    
    plt.show()

def saveWeights(theta0, theta1):
    weights = {
        "theta0": theta0,
        "theta1": theta1
    }
    with open('weights.txt', 'w') as outfile:
        json.dump(weights, outfile)
    print("Weights saved!")

def trainOnData(data):
    dataColumns = data.columns.values[:2]
    kilometers = np.array(data[dataColumns[0]].values)
    price = np.array(data[dataColumns[1]].values)

    theta0, theta1 = performGradientDescent(meanNormalization(kilometers), price)
    theta0, theta1 = meanNormReverse(theta0, theta1, kilometers)
    return kilometers, price, theta0, theta1

def main():
    try:
        data = pd.read_csv("data.csv")
    except:
        print("Failed to parse data file")
        return -1
    
    kilometers, price, theta0, theta1 = trainOnData(data)
    saveWeights(theta0, theta1)

    if "-cost" in sys.argv or "-line" in sys.argv or "-history" in sys.argv or "-scatter" in sys.argv:
        showGrpahs(data, kilometers, price, theta0, theta1)

    if "-error" in sys.argv:
        printAverageErrorPercantage(kilometers, price, theta0, theta1)


if __name__ == "__main__":
    main()