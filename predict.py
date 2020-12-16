import json
from train import makePrediction 

def getWeightsFromFile():
    with open('weights.txt') as file:
        weights = json.load(file)
    return weights["theta0"], weights["theta1"]

def getInputFromUser():
    typedMileage = input("Type mileage of the car: ")
    return float(typedMileage)

def main():
    try:
        typedMileage = getInputFromUser()
    except:
        print("Type a valid value")
        return -1
    theta0, theta1 = 0, 0
    try:
        theta0, theta1 = getWeightsFromFile()
        if type(theta0) is not float or type(theta1) is not float:
            print("Recovered data is bad")
            theta0, theta1 = 0, 0
    except:
        pass
    print("Prediction value is: ", makePrediction(theta0, theta1, typedMileage))

if __name__ == "__main__":
    main()