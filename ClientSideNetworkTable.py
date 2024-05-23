import time
import json
from networktables import NetworkTables
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import math
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import LinearSVR
from numpy import absolute
from numpy import mean
from numpy import std
# To see messages from network tables, you must set up logging
import logging

vectorValuedFunction = False  #Checks to see if it has multiple inputs
inputVariableDimensions = 1
hasParseDataRan = False
mvInputVector = []
mvInputVectors = []
mvOutputVector = []
mvOutputVectors = []

# Holds data from file
dataArray = [[], []]
graphSubplotSize = [2, 2, 1]
dataLambda = []
# Holds output of the function
functionOut = []
# For making the graph look smooth
functXList = []
# Holds residuals (actual value minus predicted value)
residList = []
residSquaredList = []
# Holds all the differences between individual values and the mean
stdDevList = []
# Basically "If a coefficient is below this number, set it to zero"
# If you want completely raw values, set deadband to zero
deadband = .000001
# Digits to truncate to
keptDecimalPlaces = 5
poly = PolynomialFeatures(degree=3, include_bias=False)
model = GradientBoostingRegressor()
wrapper = MultiOutputRegressor(model)


# r-squared is the proportion of variability in the data accounted for the model, or how accurate the graph is to
# the model. Higher r squared is better. 1 means the graph is perfectly accurate, 0 means it is not accurate at all.
# Formula for r squared is 1-[(sum of differences between actual minus predicted)^2/(sum of y minus the y mean)^2]
def runData(shouldShow):
    global dataArray
    global mvInputVectors
    global mvOutputVectors
    parseData(1)
    n = len(dataArray[0])
    deg = 1
    while n >= 9:
        n -= 10
        deg += 1
    #print("Based on the rule of 10, the polynomial degree should be:", deg)
    model = computeRSquared(deg, shouldShow)
    return model

def runValue(value):
    row = [4.04]
    row_poly = poly.transform([row])
    log_yhat = wrapper.predict(row_poly)
    print(wrapper.estimators_[0][0])
    yhat = np.exp(log_yhat)

    # summarize the prediction
    print('Predicted: %s' % yhat[0])
def computeRSquared(deg, shouldCheck):
    global mvInputVectors,mvOutputVectors,wrapper
    # Variable declarations (i typically program in java, sue me)
    # define base model
    mvInputVectors_poly = poly.fit_transform(mvInputVectors)
    log_mvOutputVectors = np.log(mvOutputVectors)
    # Define the direct multioutput wrapper model
    # Fit the model on the polynomial features of the dataset
    wrapper.fit(mvInputVectors_poly, log_mvOutputVectors)
    # Get the coefficients and intercepts from each LinearSVR estimator
    #coefficients = [estimator.coef_ for estimator in wrapper.estimators_]
    #intercepts = [estimator.intercept_ for estimator in wrapper.estimators_]

    return model


def parseData(outputs):
    global hasParseDataRan, dataArray, mvInputVectors, mvOutputVectors
    if not hasParseDataRan:
        # This reads all txt files in sample_data, and puts them all in a nx2 matrix (n being amount of rows,
        # 2 being the amount of columns) reads every file in the sample_data folder
        for file in os.listdir("data"):
            # checks if the file is a .csv (rio data log file type)
            if file.endswith(".txt"):
                # opens the reader, runs while ignoring heading
                reader = open("data/" + file, "r", encoding="utf-8")
                lineCount = 0

                for line in reader:

                    try:
                        # so it doesn't read headings of txt files, or the column names
                        if lineCount > 1:

                            # x vals are data array 0 y vals are data array 1
                            readDataLambda = line.split(",")
                            #Multiple input variables
                            mvInputVector = []
                            for i in range(0, len(readDataLambda) - outputs):  #all except the last value in the list
                                #Create a new input vector for the function
                                mvInputVector.append(float(readDataLambda[i]))
                                print(readDataLambda[i])
                            #Log the input vector

                            mvInputVectors.append(mvInputVector)
                            #Log the output vector

                            mvOutputVector = []
                            for i in range(len(readDataLambda) - outputs, len(readDataLambda)):  #all except the last value in the list
                                #Create a new input vector for the function
                                mvOutputVector.append(float(readDataLambda[i]))
                            mvOutputVectors.append(mvOutputVector)
                            #dataArray = np.array(mvInputVectors,mvOutputVectors)
                    except:
                        # if an error occurs, print the line that it failed to read
                        print("Error reading line", lineCount)
                    lineCount += 1
                reader.close()
        mvInputVectors = np.array(mvInputVectors)
        mvOutputVectors = np.array(mvOutputVectors)
        print(mvInputVectors)
        print(mvOutputVectors)

        """print("Data Array X Values:")
        print(dataArray[0])
        print(" ")
        print("Data Array Y Values:")
        print(dataArray[1])
        print(" ")"""
    else:
        print("Data has already been parsed, moving on")
    hasParseDataRan = True


logging.basicConfig(level=logging.ERROR)


def runModel(shouldShow):
    model = runData(shouldShow)
    return "hi"


while not NetworkTables.isConnected():
    NetworkTables.initialize(server="10.1.35.2")
    time.sleep(2)
    NetworkTables.initialize(server="localhost")
    time.sleep(2)

#connected

#TODO: make custom loop running faster
sd = NetworkTables.getTable("SmartDashboard")

data_to_robot = {
    "test": "0"  #comma then next value
}
i = 0
lastSentUpdate = 0
print("Connected.")
while True:
    # Read JSON from robot
    data_to_robot.clear()
    json_response = sd.getString("FromRobot", "default")
    if json_response != "default":
        data_from_robot = json.loads(json_response)

        #print("Robot status:", data_from_robot["status"])
        #CALL THE COMPUTE R SQUARED FUNCTION HERE!
        if "shouldUpdateModel" in data_from_robot:
            if time.time() - lastSentUpdate > .5:
                print("DO")
                lastSentUpdate = time.time()
                m_input = data_from_robot["shouldUpdateModel"]
                if m_input == "modelUpdate" and inputVariableDimensions == 1:
                    modelString = runModel(False)
                    data_to_robot["modelUpdated"] = modelString
                    print("update time" + str(time.time() - lastSentUpdate))
                    currentTime = time.time()
                    runValue(4.5)
                    print("running value time: " + str(time.time() - currentTime))

    i += 1
    data_to_robot["test"] = i
    # Convert dictionary to JSON string
    json_data_to_robot = json.dumps(data_to_robot)
    # Send JSON to robot
    sd.putString("ToRobot", json_data_to_robot)
    time.sleep(0.105)
