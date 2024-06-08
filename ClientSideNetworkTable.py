import socket
import time
import json
import threading
import pandas as pd
from networktables import NetworkTables
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor  # , RandomForestRegressor  #Add me if you want RandomForest!
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
import random

# To see messages from network tables, you must set up logging
import logging
import joblib

# Variable Declarations
hasParseDataRan = False
mvInputVector = []
mvInputVectors = []
mvOutputVector = []
mvOutputVectors = []
variableNames = []
inputNames = []
outputNames = []
dataLambda = []





'''
If you care more about overfitting the model less, use
model = RandomForestRegressor(n_jobs=-1,n_estimators=100)'''
model = GradientBoostingRegressor(n_estimators=100)
wrapper = MultiOutputRegressor(model)
# Stores values for testing
testInput = []
testOutput = []
# Number of outputs
outputSize = 1


def runData(shouldShow, reRunning):
    """This function parses all the data sent to the data handler,
    creates an AI model out of the data, and saves the model as a folder named as the created timestamp

    Parameters:
        shouldShow (boolean): Whether an analysis of the model should be printed
        reRunning (boolean):  Whether the data has been run previously"""

    global mvInputVectors
    global mvOutputVectors
    if not reRunning:
        parseData(outputSize)  # number of outputs

    createModel(shouldShow)
    saveModel()


def sendModel():
    """Sends the file to the RoboRIO"""
    if host == "localhost":
        HOST = host
    else:
        HOST = '10.1.35.63'  # Orange Pi 5 Address
    PORT = 5801  # The same port as used by the server

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        #Creates the model, names it the timestamp
        model_directories = latest_model()
        latest_directory = max(model_directories, key=os.path.getmtime)

        with open(os.path.join(latest_directory, "wrapper"), 'rb') as f:
            while True:
                bytes_read = f.read(1024)
                if not bytes_read:
                    break
                s.sendall(bytes_read)
    print('File sent successfully')
    return


def runValue(value):
    """Computes what the model outputs as a specific input value (returns essentially an f(x) if value is x)

    Parameters:
        value (float): The value that you want a model output for

    Returns:
        float: the model output for that value"""
    row = [[value]]
    yhat = wrapper.predict(row)
    # summarize the prediction
    # print('Predicted: %s' % yhat[0])
    return yhat


def createModel(shouldCheck):
    """Create the neural network model based on the data in mvInputVectors and mvOutputVectors

    Parameters:
        shouldCheck (boolean): Whether you want a detailed analysis of the function's fit printed out"""


    global mvInputVectors, mvOutputVectors, wrapper
    # Fit the model on the polynomial features of the dataset
    wrapper.fit(mvInputVectors, mvOutputVectors)

    if shouldCheck:
        predictions = wrapper.predict(testInput)
        overall_mse = mean_squared_error(testOutput, predictions)
        overall_mae = mean_absolute_error(testOutput, predictions)
        overall_r2 = r2_score(testOutput, predictions)

        print(f"Overall Mean Squared Error: {overall_mse}")
        print(f"Overall Mean Absolute Error: {overall_mae}")
        print(f"Overall R^2 Score: {overall_r2}")



def graphImportance():
    """Graph the feature importances.
    Feature importances show how much a particular variable (property of the input that changes) effects the result
     of the data."""
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


def parseData(outputSize):
    """Takes the inputs from the data folder and converts them into a program-usable array.

    :param outputSize: The length of the output vectors.
    """
    global hasParseDataRan, mvInputVectors, mvOutputVectors, variableNames, testInput, testOutput
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
                            for i in range(len(variableNames) - outputSize):
                                inputNames.append(variableNames[i])
                            for i in range(len(variableNames) - outputSize, len(variableNames)):
                                outputNames.append(variableNames[i])
                        else:
                            # x vals are data array 0 y vals are data array 1
                            readDataLambda = line.split(",")
                            # Multiple input variables
                            mvInputVector = []
                            for i in range(0, len(readDataLambda) - outputSize):  # all except the last value in the list
                                # Create a new input vector for the function
                                mvInputVector.append(float(readDataLambda[i]))
                                # print(readDataLambda[i])

                            # Log the input vector
                            mvInputVectors.append(mvInputVector)

                            mvOutputVector = []
                            for i in range(len(readDataLambda) - outputSize,
                                           len(readDataLambda)):  # all except the last value in the list
                                # Create a new input vector for the function
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
        # print(mvInputVectors)
        # print(mvOutputVectors)

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
    """Saves the model as a .pkl file on a local drive"""
    timestamp = time.strftime("%m%d-%H%M%S")
    directory = "Models/" + str(timestamp)
    os.makedirs(directory)
    # joblib.dump(poly, directory + "/" + "PolynomialFeatures")
    joblib.dump(wrapper, directory + "/" + "wrapper")



def latest_model():
    """Returns the path of the most recently timestamped directory. Must use max(return, key = os.path.getmtime)!!!"""
    directory = "Models"

    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Return the list of paths of subdirectories in the directory
    return [f.path for f in os.scandir(directory) if f.is_dir()]



def load_latest_model(backupShower):
    """
    Loads the model from the aforementioned files
    :param backupShower: the backup data in case no models are found
    :return:
    """
    global wrapper
    # Get list of subdirectories in the Models directory
    model_directories = latest_model()

    if not model_directories:
        print("No models found. Running Model.")
        runData(backupShower, False)
    # Sort directories by creation time (modification time of the directory)
    else:
        parseData(outputSize)
        latest_directory = max(model_directories, key=os.path.getmtime)

        # Load model from the latest directory
        # poly = joblib.load(os.path.join(latest_directory, "PolynomialFeatures"))
        wrapper = joblib.load(os.path.join(latest_directory, "wrapper"))


load_latest_model(True)


#graphImportance()
host = ""

def connect():
    """Keeps trying to make a connection with either the robot or the simulation logs"""
    global host
    while not NetworkTables.isConnected():
        NetworkTables.initialize(server="10.1.35.2")
        print("Connecting to Robot")
        time.sleep(2)
        if NetworkTables.isConnected():
            host = "10.1.35.2"
            break
        print("Failed")
        NetworkTables.initialize(server="localhost")
        host = "localhost"
        print("Connecting to Local")
        time.sleep(2)
    print("Connected to " + host)


connect()
# connected
sd = NetworkTables.getTable("SmartDashboard")

data_to_robot = {
    "test": "0"  # comma then next value
}
i = 0
lastSentUpdate = 0
last_execution_time = time.time()
while True:
    if not NetworkTables.isConnected():
        connect()
    current_time = time.time()
    if current_time - last_execution_time > .11:
        last_execution_time = current_time
        # Read JSON from robot
        data_to_robot.clear()
        json_response = sd.getString("FromRobot", "default")
        if json_response != "default":
            data_from_robot = json.loads(json_response)

            # print("Robot status:", data_from_robot["status"])
            # CALL THE COMPUTE R SQUARED FUNCTION HERE!
            if "shouldUpdateModel" in data_from_robot:
                if time.time() - lastSentUpdate > .5:
                    # print("DO")
                    lastSentUpdate = time.time()
                    m_input = data_from_robot["shouldUpdateModel"]
                    m_input = m_input[1:-1].replace(" ", "")
                    dataSave = m_input
                    m_input = m_input.split(",")
                    print(m_input)
                    inputVals = []
                    for i in range(len(m_input) - outputSize):
                        inputVals.append(float(m_input[i]))
                    outputVals = []
                    for i in range(len(m_input) - outputSize, len(m_input)):
                        outputVals.append(m_input[i])
                    new_input_array = np.array([inputVals])
                    new_output_array = np.array([outputVals])
                    mvInputVectors = np.concatenate((mvInputVectors, new_input_array), axis=0)
                    mvOutputVectors = np.concatenate((mvOutputVectors, new_output_array), axis=0)

                    runData(False, True)
                    print("update time" + str(time.time() - lastSentUpdate))
                    data_to_robot["modelUpdated"] = str("True")
                    with open("data/shooterData.txt", 'a') as file:
                        file.write(f'\n{dataSave}')
                    send_thread = threading.Thread(target=sendModel)
                    send_thread.start()
            if "modelDistance" in data_from_robot:
                m_distance = float(data_from_robot["modelDistance"])
                timeOld = time.time()
                outputs = runValue(m_distance)
                # print("time to get val:" + str(time.time() - timeOld))
                data_to_robot["outputs"] = str(outputs)
        i += 1
        data_to_robot["time"] = i
        # Convert dictionary to JSON string
        json_data_to_robot = json.dumps(data_to_robot)
        # Send JSON to robot
        sd.putString("ToRobot", json_data_to_robot)
