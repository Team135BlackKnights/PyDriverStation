import sys
import time
import json
from networktables import NetworkTables
from re import X
import numpy as np
import os
import io
import matplotlib.pyplot as plt
import math
from decimal import Decimal

# To see messages from networktables, you must set up logging
import logging
global vectorValuedFunction
vectorValuedFunction = False #Checks to see if it has multiple inputs
global inputVariableDimensions
inputVariableDimensions= 1
global hasParseDataRan
hasParseDataRan = False
global mvInputVectors
mvInputVector = []
global mvOutputVectors
mvOutputVectors = []

# Holds data from file
global dataArray
dataArray = [[], []]
global graphSubplotSize
graphSubplotSize = [2, 2, 1]
dataLambda = []
# Holds output of the function
functionOut = []
# For making the graph look smooth
functXList = []
# Holds residuals (actual value minus predicted value)
global residList
residList = []
global residSquaredList
residSquaredList = []
# Holds all of the differences between individual values and the mean
global stdDevList
stdDevList = []
# Basically "If a coefficient is below this number, set it to zero"
# If you want completely raw values, set deadband to zero
deadband = .000001
# Digits to truncate to
keptDecimalPlaces = 5


# r-squared is the proportion of variablility in the data accounted for the model, or how accurate the graph is to the model. Higher r squared is better. 1 means the graph is perfectly accurate, 0 means it is not accurate at all.
# Formula for r squared is 1-[(sum of differences between actual minus predicted)^2/(sum of y minus the y mean)^2]
def runData(shouldShow):
    global dataArray
    global mvInputVectors
    global mvOutputVectors
    parseData()
    n = len(dataArray[0])
    deg = 1
    while (n >= 9):
        n -= 10
        deg += 1
    #print("Based on the rule of 10, the polynomial degree should be:", deg)
    model = computeRSquared(deg, shouldShow)
    return model


def computeRSquared(deg, shouldShow):
    # Variable declarations (i typically program in java, sue me)
    plt.subplot(graphSubplotSize[0], graphSubplotSize[1], graphSubplotSize[2])
    plt.plot(dataArray[0], dataArray[1], "ro")
    rSquared = 0
    rSquaredNum = 0
    # sample number
    n = len(dataArray[0])
    # average of output
    meanY = np.mean(dataArray[1])
    model = np.polynomial.polynomial.polyfit(dataArray[0], dataArray[1], deg, full=False)
    graph = np.poly1d(model)
    # Square individual residuals
    for i in range(len(dataArray[0])):
        # Computes residual (Actual minus predicted)
        residVal = np.polynomial.polynomial.polyval(dataArray[0][i], graph) - dataArray[1][i]
        # Deadbands the residual
        if abs(residVal) < deadband:
            residVal = 0
        residList.append(residVal)

    for i in range(n):
        residSquaredList.append(residList[i] * residList[i])
    # Numerator is the sum of squared residuals
    rSquaredNum = np.sum(residSquaredList)
    # Squares the difference between each data point's y coord and the mean, then sums them
    for i in dataArray[1]:
        indDiv = (i - meanY)
        indDiv = indDiv * indDiv
        stdDevList.append(indDiv)
    rSquaredDenom = np.sum(stdDevList)
    # Computes r squared
    rSquared = 1 - (rSquaredNum / rSquaredDenom)
    # Outputs r squared
    print("R Squared:", rSquared, "\n")
    if shouldShow:

        maxMinusMin = max(dataArray[0]) - min(dataArray[0])
        indSegLen = maxMinusMin / 50

        # intended to graph the polynomial from 15 percent of the range below the min to 15 percent of the range above the maximu
        # values that show minimum and maximum values to graph
        graphMin = -.33 * maxMinusMin + min(dataArray[0])
        graphMax = .33 * maxMinusMin + max(dataArray[0])
        indSegLen = (graphMax - graphMin) / 50
        # We use 50 segments to approximate the graph with edges included
        for i in range(50):
            functXVal = graphMin + (i * indSegLen)
            functXList.append(functXVal)
            functionOut.append(np.polynomial.polynomial.polyval(functXVal, graph))

        plt.subplot(graphSubplotSize[0], graphSubplotSize[1], graphSubplotSize[2])
        plt.plot(functXList, functionOut)
        plt.title("X vs Y")
        # Graphs lists of points
        plt.subplot(graphSubplotSize[0], graphSubplotSize[1], (graphSubplotSize[2] + 1))
        plt.plot(dataArray[0], residList, "ro")
        # Creates the x axis (for better referencing)
        xValues = [min(dataArray[0]), max(dataArray[0])]
        yValues = [0, 0]
        plt.plot(xValues, yValues)
        plt.title("Residuals")
        plt.show()
    return model

def parseData():
    global hasParseDataRan
    global dataArray
    global mvInputVectors
    global mvOutputVectors
    if not hasParseDataRan:
        # This reads all txt files in sample_data, and puts them all in a nx2 matrix (n being amount of rows, 2 being the amount of columns)
        # reads every file in the sample_data folder
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
                            dataLambda = line.split(",")

                            dataLambdalen = len(dataLambda)
                            if len(dataLambda) == 2: #if input variable is 1
                                dataArray[0].append(float(dataLambda[0]))
                                dataArray[1].append(float(dataLambda[1]))
                            else: #Multiple input variables
                                mvInputVector = []
                                for i in range(0, len(dataLambda)-2): #all except the last value in the list
                                    #Create a new input vector for the function
                                    mvInputVector.append(dataLambda[i])
                                #Log the input vector
                                mvInputVectors.append(mvInputVector)
                                #Log the output vector
                                mvOutputVectors.append(dataLambda[:-1])

                    except:
                        # if an error occurs, print the line that it failed to read
                        print("Error reading line", lineCount)
                    lineCount += 1
        if (vectorValuedFunction):
            mvInputVectors = np.array(mvInputVector)
            mvOutputVectors = np.array(mvOutputVectors)
        else:
            # converts read values into np array for regression
            dataArray = np.array(dataArray)
            # For Debugging, makes sure that your data looks good
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
    model = runData(True)
    modelString = ""
    for i in range(len(model)):
      xValue = model[i]
      if abs(xValue) < deadband:
        xValue = 0
      #Truncates value to the amount of decimal places specified in the variable (done so that floating-point error is negligible)
      xValue *= math.pow(10, keptDecimalPlaces)
      xValue = math.trunc(xValue)
      xValue /= math.pow(10, keptDecimalPlaces)
      modelString += str(xValue) + ","
    modelString = modelString[:-1]
    return modelString
while not NetworkTables.isConnected():
    NetworkTables.initialize(server="10.1.35.2")
    time.sleep(2)
    NetworkTables.initialize(server="localhost")
    time.sleep(2)

#connected

#get network Table
#TODO: make custom loop running faster
sd = NetworkTables.getTable("SmartDashboard")

data_to_robot = {
    "test": "0"  #comma then next value
}
i = 0
while True:
    # Read JSON from robot
    data_to_robot.clear()
    json_response = sd.getString("FromRobot", "default")
    if json_response != "default":
        data_from_robot = json.loads(json_response)
        if "status" in data_from_robot:
            print("Robot status:", data_from_robot["status"])
            #CALL THE COMPUTE R SQUARED FUNCTION HERE!
        if "shouldUpdateModel" in data_from_robot:
            input = data_from_robot["shouldUpdateModel"]
            if (input == "modelUpdate" & inputVariableDimensions == 1):
                modelString = runModel(True)
                data_to_robot["modelUpdated"] = modelString


    i += 1
    data_to_robot["test"] = i
    # Convert dictionary to JSON string
    json_data_to_robot = json.dumps(data_to_robot)

    # Send JSON to robot
    sd.putString("ToRobot", json_data_to_robot)
    time.sleep(0.105)
