import sys
import time
import json
from networktables import NetworkTables

# To see messages from networktables, you must setup logging
import logging

logging.basicConfig(level=logging.DEBUG)

NetworkTables.initialize(server="localhost") # or localhost if sim!!!

sd = NetworkTables.getTable("SmartDashboard")

data_to_robot = {
    "test": "0" #comma then next value
}

i = 0
while True:
    # Read JSON from robot
    json_response = sd.getString("DataHandlerResponse", "default")
    if json_response != "default":
        data_from_robot = json.loads(json_response)
        if "status" in data_from_robot:
            print("Robot status:", data_from_robot["status"])
    i +=1
    data_to_robot["test"] = i
    # Convert dictionary to JSON string
    json_data_to_robot = json.dumps(data_to_robot)

    # Send JSON to robot
    sd.putString("DataHandler", json_data_to_robot)
    time.sleep(0.105)
