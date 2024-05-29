import os
import socket
import threading
import json
import time

import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

model = GradientBoostingRegressor(n_estimators=100)
wrapper = MultiOutputRegressor(model)
def send_to_roborio(data, roborio_ip, roborio_port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((roborio_ip, roborio_port))
        s.sendall((json.dumps(data) + '\n').encode())
        data_from_robot = json.loads(s.recv(1024).decode())
        if "modelDistance" in data_from_robot:
            m_distance = float(data_from_robot["modelDistance"])
            timeOld = time.time()
            outputs = runValue(m_distance)
            print("time to get val:" + str(time.time() - timeOld))
            data["outputs"] = str(outputs)
        else:
            if 'outputs' in data:
                data.pop('outputs')

def latest_model():
    directory = "ModelsPI"

    # Create the directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)

    # Return the list of paths of subdirectories in the directory
    return [f.path for f in os.scandir(directory) if f.is_dir()]
def load_latest_model():
    global wrapper
    # Get list of subdirectories in the Models directory
    model_directories = latest_model()

    if not model_directories:
        print("No models found. Do not use runValue.")
    # Sort directories by creation time (modification time of the directory)
    else:
        latest_directory = max(model_directories, key=os.path.getmtime)
        wrapper = joblib.load(os.path.join(latest_directory, "wrapper.pkl"))

def handle_client(conn, addr):
    global wrapper
    print('Connected by', addr)
    data = b''
    while True:
        packet = conn.recv(1024)
        if not packet:
            break
        data += packet
    timestamp = time.strftime("%m%d-%H%M%S")
    directory = "ModelsPI/" + str(timestamp)
    os.makedirs(directory, exist_ok=True)

    # Write data to a file in the specified directory
    saved_file_path = os.path.join(directory, 'wrapper.pkl')
    with open(saved_file_path, 'wb') as temp_file:
        temp_file.write(data)

    # Load the wrapper.pkl file using pickle
    with open(saved_file_path, 'rb') as file:
        wrapper = joblib.load(file)

def runValue(value):
    row = [[value]]
    yhat = wrapper.predict(row)
    # summarize the prediction
    # print('Predicted: %s' % yhat[0])
    return yhat[0]

def main():
    HOST = '0.0.0.0'  # Listen on all available interfaces
    PORT = 5801  # Port to listen on
    data_to_robot = {'timestamp': '0'}
    roborio_ip = '10.1.35.2'  # Replace with the actual IP address of the roboRIO
    roborio_port = 5802  # Port on which the roboRIO is listening
    timestamp = 0
    load_latest_model()
    time.sleep(60) #wait for RIO to boot.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        print("BINDING...")
        s.bind((HOST, PORT))
        s.listen()
        s.settimeout(.001)
        print(f'Server listening on {HOST}:{PORT}')
        while True:
            timestamp +=1
            try:
                conn, addr = s.accept()
                client_thread = threading.Thread(target=handle_client, args=(conn, addr))
                client_thread.start()
            except TimeoutError:
                pass
            data_to_robot['timestamp'] = str(timestamp)
            send_to_roborio(data_to_robot, roborio_ip, roborio_port)
            #data_to_robot.clear()


            # Send the received data to the roboRIO
# BEFORE YOU CALL MAIN MAKE SURE: (only use localhost if sim)
# 1. HOST (laptop) = 10.1.35.5  OR localhost
# 2. roborio_ip = 10.1.35.2 OR localhost
# 3. You have NOT touched send_to_roborio.
# Moving any lines in this function will CRASH THE PI (robot will keep working), I almost promise you.
main()
