import time
import json

import pandas as pd
from networktables import NetworkTables
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import PolynomialFeatures
import random

# To see messages from network tables, you must set up logging
import logging
import joblib

vectorValuedFunction = False
hasParseDataRan = False
mvInputVector = []
mvInputVectors = []
mvOutputVector = []
mvOutputVectors = []
variableNames = []
inputNames = []
outputNames = []
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
testInput = []
testOutput = []


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
    # define base model
    mvInputVectors_poly = poly.fit_transform(mvInputVectors)
    #log_mvOutputVectors = np.log(mvOutputVectors)
    # Define the direct multi output wrapper model
    # Fit the model on the polynomial features of the dataset
    wrapper.fit(mvInputVectors_poly, mvOutputVectors)
    model.fit(mvInputVectors_poly, mvOutputVectors)
    if shouldCheck:
        predictions = wrapper.predict(testInput)
        overall_mse = mean_squared_error(testOutput, predictions)
        overall_mae = mean_absolute_error(testOutput, predictions)
        overall_r2 = r2_score(testOutput, predictions)

        print(f"Overall Mean Squared Error: {overall_mse}")
        print(f"Overall Mean Absolute Error: {overall_mae}")
        print(f"Overall R^2 Score: {overall_r2}")


def graphImportance():
    global model
    feature_importances = []
    for regressor in wrapper.estimators_:
        feature_importances.append(regressor.feature_importances_)
    feature_importances = pd.DataFrame(feature_importances, columns=inputNames)
    num_outputs = feature_importances.shape[0]
    fig, axes = plt.subplots(num_outputs, 1, figsize=(12, 6 * num_outputs))

    if num_outputs == 1:
        axes = [axes]  # Ensure axes is iterable when there's only one output

    for i, ax in enumerate(axes):
        sorted_idx = np.argsort(feature_importances.iloc[i])
        pos = np.arange(sorted_idx.shape[0]) + 0.5
        ax.barh(pos, feature_importances.iloc[i, sorted_idx], align="center")
        ax.set_yticks(pos)
        ax.set_yticklabels(np.array(inputNames)[sorted_idx])
        ax.set_title(f"Feature Importance for Output {i + 1} ({outputNames[i]})")

    plt.tight_layout()
    plt.show()

    result = permutation_importance(
        wrapper, testInput, testOutput, n_repeats=10, random_state=42, n_jobs=-1
    )
    sorted_idx = result.importances_mean.argsort()
    plt.subplot(1, 2, 2)
    plt.boxplot(
        result.importances[sorted_idx].T,
        vert=False,
        tick_labels=np.array(inputNames)[sorted_idx],
    )
    plt.title("Permutation Importance (test set)")
    fig.tight_layout()
    plt.show()


def parseData(outputs):
    global hasParseDataRan, dataArray, mvInputVectors, mvOutputVectors, variableNames, testInput, testOutput
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
                        if lineCount == 0:
                            variableNames = line.split(",")
                            variableNames[0] = variableNames[0][1:]
                            variableNames[len(variableNames) - 1] = variableNames[len(variableNames) - 1][:-1]
                            for i in range(len(variableNames) - outputs):
                                inputNames.append(variableNames[i])
                                variableNames.pop(i)
                            for i in range(len(variableNames) - outputs, len(variableNames)):
                                outputNames.append(variableNames[i])
                                variableNames.pop(i)
                        else:
                            # x vals are data array 0 y vals are data array 1
                            readDataLambda = line.split(",")
                            #Multiple input variables
                            mvInputVector = []
                            for i in range(0, len(readDataLambda) - outputs):  #all except the last value in the list
                                #Create a new input vector for the function
                                mvInputVector.append(float(readDataLambda[i]))
                                #print(readDataLambda[i])

                            #Log the input vector
                            mvInputVectors.append(mvInputVector)

                            mvOutputVector = []
                            for i in range(len(readDataLambda) - outputs,
                                           len(readDataLambda)):  #all except the last value in the list
                                #Create a new input vector for the function
                                mvOutputVector.append(float(readDataLambda[i]))
                            mvOutputVectors.append(mvOutputVector)
                    except ValueError:
                        # if an error occurs, print the line that it failed to read
                        print("ValueError reading line", lineCount, "\nMODEL WILL CRASH SHORTLY.")
                    except TypeError:
                        print("TypeError reading line", lineCount, "\nMODEL WILL CRASH SHORTLY.")
                    lineCount += 1
                reader.close()
        sample = int(len(mvInputVectors) * .4)
        # Generate a list of indices from 0 to len(mvInputVectors) - 1
        indices = list(range(len(mvInputVectors)))
        # Randomly sample indices
        sampled_indices = random.sample(indices, sample)
        # Create testInput and testOutput lists based on the sampled indices
        offset = 0
        for j in sampled_indices:
            testOutput.append(mvOutputVectors[j - offset])
            mvOutputVectors.pop(j - offset)
            testInput.append(mvInputVectors[j - offset])
            mvInputVectors.pop(j - offset)
            offset += 1
        mvInputVectors = np.array(mvInputVectors)
        mvOutputVectors = np.array(mvOutputVectors)
        testInput = np.array(testInput)
        testOutput = np.array(testOutput)

        #print(mvInputVectors)
        #print(mvOutputVectors)

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


def saveModel():
    timestamp = time.strftime("%m%d-%H%M%S")
    directory = "Models/" + str(timestamp)
    os.makedirs(directory)
    joblib.dump(poly, directory + "/" + "PolynomialFeatures")
    joblib.dump(wrapper, directory + "/" + "wrapper")


def load_latest_model(backupShower):
    global poly, wrapper
    # Get list of subdirectories in the Models directory
    model_directories = [f.path for f in os.scandir("Models") if f.is_dir()]

    if not model_directories:
        print("No models found. Running Model.")
        runData(backupShower)
    # Sort directories by creation time (modification time of the directory)
    else:
        latest_directory = max(model_directories, key=os.path.getmtime)

        # Load model from the latest directory
        poly = joblib.load(os.path.join(latest_directory, "PolynomialFeatures"))
        wrapper = joblib.load(os.path.join(latest_directory, "wrapper"))


parseData(1)
load_latest_model(True)
graphImportance()

#saveModel()
while not NetworkTables.isConnected():
    NetworkTables.initialize(server="10.1.35.2")
    print("Connecting to Robot")
    time.sleep(2)
    if NetworkTables.isConnected():
        break
    print("Failed")
    NetworkTables.initialize(server="localhost")
    print("Connecting to Local")
    time.sleep(2)

#connected

#If this isn't fast enough (100ms) then you'll need a custom Entry.
sd = NetworkTables.getTable("SmartDashboard")

data_to_robot = {
    "test": "0"  #comma then next value
}
i = 0
lastSentUpdate = 0
print("Connected.")
last_execution_time = time.time()
while True:
    current_time = time.time()
    if current_time - last_execution_time > .11:
        last_execution_time = current_time
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
                    if m_input == "modelUpdate":
                        runData(False)
                        print("update time" + str(time.time() - lastSentUpdate))
            if "modelDistance" in data_from_robot:
                m_distance = float(data_from_robot["modelDistance"])
                timeOld = time.time()
                outputs = runValue(m_distance)
                print("time to get val:" + str(time.time() - timeOld))
                data_to_robot["outputs"] = str(outputs)
        i += 1
        data_to_robot["time"] = i
        # Convert dictionary to JSON string
        json_data_to_robot = json.dumps(data_to_robot)
        # Send JSON to robot
        sd.putString("ToRobot", json_data_to_robot)
