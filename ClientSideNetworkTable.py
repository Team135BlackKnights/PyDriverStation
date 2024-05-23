import time
import json
from networktables import NetworkTables
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import PolynomialFeatures
import random

# To see messages from network tables, you must set up logging
import logging

vectorValuedFunction = False
#Checks to see if it has multiple inputs
inputVariableDimensions = 1
hasParseDataRan = False
mvInputVector = []
mvInputVectors = []
mvOutputVector = []
mvOutputVectors = []
variableNames = []
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
#Stores values for testing
global x_test, y_test
x_test = []
y_test = []


# r-squared is the proportion of variability in the data accounted for the model, or how accurate the graph is to
# the model. Higher r squared is better. 1 means the graph is perfectly accurate, 0 means it is not accurate at all.
# Formula for r squared is 1-[(sum of differences between actual minus predicted)^2/(sum of y minus the y mean)^2]
def runData(shouldShow):
    global dataArray
    global mvInputVectors
    global mvOutputVectors
    parseData(1)  #number of outputs
    n = len(dataArray[0])
    deg = 1
    while n >= 9:
        n -= 10
        deg += 1
    #print("Based on the rule of 10, the polynomial degree should be:", deg)
    computeRSquared(deg, shouldShow)


def runValue(value):
    row = [value]
    row_poly = poly.transform([row])
    #log_yhat = wrapper.predict(row_poly)
    yhat = wrapper.predict(row_poly)
    # summarize the prediction
    print('Predicted: %s' % yhat[0])
    return yhat[0]


def computeRSquared(deg, shouldCheck):
    global mvInputVectors, mvOutputVectors, wrapper, poly
    poly = PolynomialFeatures(degree=deg, include_bias=False)
    # Variable declarations (i typically program in java, sue me)
    # define base model
    mvInputVectors_poly = poly.fit_transform(mvInputVectors)
    #log_mvOutputVectors = np.log(mvOutputVectors)
    # Define the direct multioutput wrapper model
    # Fit the model on the polynomial features of the dataset
    wrapper.fit(mvInputVectors_poly, mvOutputVectors)
    # Get the coefficients and intercepts from each LinearSVR estimator
    #coefficients = [estimator.coef_ for estimator in wrapper.estimators_]
    #intercepts = [estimator.intercept_ for estimator in wrapper.estimators_]


def graphImportance():
    global x_test
    global y_test
    feature_importance = model.feature_importances_
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + 0.5
    fig = plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.barh(pos, feature_importance[sorted_idx], align="center")
    plt.yticks(pos, np.array(variableNames)[sorted_idx])
    plt.title("Feature Importance (MDI)")

    result = permutation_importance(
        model, x_test, y_test, n_repeats=10, random_state=42, n_jobs=2
    )
    sorted_idx = result.importances_mean.argsort()
    plt.subplot(1, 2, 2)
    plt.boxplot(
        result.importances[sorted_idx].T,
        vert=False,
        labels=np.array(variableNames)[sorted_idx],
    )
    plt.title("Permutation Importance (test set)")
    fig.tight_layout()
    plt.show()


def parseData(outputs):
    global hasParseDataRan, dataArray, mvInputVectors, mvOutputVectors, variableNames
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
                    generatedNumber = random.randint(0, 5)
                    try:
                        # so it doesn't read headings of txt files, or the column names
                        if lineCount == 1:
                            variableNames = line.split(",")
                        else:
                            # x vals are data array 0 y vals are data array 1
                            readDataLambda = line.split(",")
                            #Multiple input variables
                            mvInputVector = []
                            for i in range(0, len(readDataLambda) - outputs):  #all except the last value in the list
                                #Create a new input vector for the function
                                mvInputVector.append(float(readDataLambda[i]))
                                print(readDataLambda[i])
                            #We use a Simple Random Sample for test data. If the number equals this value we use it
                            if generatedNumber == 0:
                                x_test.append(mvInputVector)
                            else:
                                #Log the input vector
                                mvInputVectors.append(mvInputVector)

                            mvOutputVector = []
                            for i in range(len(readDataLambda) - outputs,
                                           len(readDataLambda)):  #all except the last value in the list
                                #Create a new input vector for the function
                                mvOutputVector.append(float(readDataLambda[i]))
                            #If the random number equals zero, send the point to test array
                            if generatedNumber == 0:
                                y_test.append(mvOutputVector)
                            else:
                                # Else log the output vector
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
                    modelString = runData(False)
                    data_to_robot["modelUpdated"] = modelString
                    print("update time" + str(time.time() - lastSentUpdate))
                    print(runValue(4.5))
        if "modelDistance" in data_from_robot:
            m_distance = data_from_robot["modelDistance"]
            angle = runValue(m_distance)[0]
            data_to_robot["predictedAngle"] = str(angle)
    i += 1
    data_to_robot["time"] = i
    # Convert dictionary to JSON string
    json_data_to_robot = json.dumps(data_to_robot)
    # Send JSON to robot
    sd.putString("ToRobot", json_data_to_robot)
    time.sleep(0.105)
