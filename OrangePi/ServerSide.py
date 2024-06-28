import os
import socket
import threading
import json
import time
import re
import joblib
import numpy as np
from frccontrol import DcBrushedMotor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

#Custom Imports
from DoubleJointedArm import DoubleJointedArm
#Gradient Booster is a tree algorithm that uses "trees" (estimators) like neural network layers.
model = GradientBoostingRegressor(n_estimators=100)
#We are always going to be using at least one input and one output, so we require this wrapper
wrapper = MultiOutputRegressor(model)

#Do not edit the directories. It will crash.
model_naming = re.compile(r"model_v(\d+)\.pkl")
directory = "../OrangePi/ModelsPI/"
MOTOR_KRAKEN_X60_FOC = DcBrushedMotor(12.0, 9.36, 476.1, 2, 6000.0)
dt = 0.02  #what's the periodic time OF DoubleJointedArmS.java ?
length1 = 46.25 * .0254  # in meters, so .0254 conversion factor
length2 = 41.8 * .0254
# Mass of segments
mass1 = 9.34 * 0.4536  #In KG
mass2 = 9.77 * 0.453

# Distance from pivot to CG for each segment
pivot_to_CG1 = 21.64 * 0.0254  #in meters
pivot_to_CG2 = 26.70 * 0.0254

# Moment of inertia about CG for each segment
MOI1 = 2957.05 * 0.0254 * 0.0254 * 0.4536  # In KgM^2
MOI2 = 2824.70 * 0.0254 * 0.0254 * 0.4536

# Gearing of each segment
gearing1 = 70.0  #Reduction is greater than 1.
gearing2 = 45.0

# Number of motors in each section
motor_count1 = 1  #How many motors on arm
motor_count2 = 1  #How many motors on elbow

# Motor Type. ALL ARM MOTORS MUST BE SAME
motor_type = MOTOR_KRAKEN_X60_FOC
# Gravity
gravity = 9.806

#Real controls and outputs.
voltages = np.zeros(4)
angleShoulder = 0  #auto updated on boot
angleElbow = 0  #auto updated on boot


def to_state(x, y, invert, arm):
    """
    :param arm: what arm to make the state for
    :param x: in meters for left/right
    :param y: in meters for top/bottom
    :param invert: should the elbow be left or right of the setpoint?
    :return: numpy array with arm theta, elbow theta, and two velocities at the setpoint (0)
    """
    theta1, theta2 = arm.constants.inv_kinematics(x, y, invert)
    return np.array([[theta1], [theta2], [0], [0]])


def initialize_arm(target_state):
    """
    Initialize the arm.
    """
    return DoubleJointedArm(dt, length1, length2, mass1, mass2, pivot_to_CG1, pivot_to_CG2, MOI1, MOI2, gearing1,
                            gearing2, motor_count1, motor_count2, motor_type, gravity, target_state)


def initialize_arm_with_encoders():
    """
    Initialize the arm's state based on encoder readings. does NOT accept given velocity, as latency compensation
    already consumes those values.
    """
    target_state = np.array([[angleShoulder], [angleElbow], [0],
                             [0], [0], [0]])
    arm = initialize_arm(target_state)
    # Set initial state based on encoder values
    arm.x = target_state  # Initial state [angle1, angle2, velocity1, velocity2]
    arm.observer.x_hat = arm.x  #override the Kalman filter to have the target_state (since it's our start pos)

    return arm


def arm_loop(arm):
    """
    Run the arm. Usually done in a separate thread.
    :param arm: arm to run
    :return: never, run in a thread.
    """
    #Uncomment the commented lines, and COMMENT the ones with COMMENT FOR MATPLOT for a visual output ON the pi
    #(almost never done)
    global voltages
    #fig, ax, arm_line, ref_line = initialize_plot_live(arm)
    while True:
        arm.update()
        #arm_line, ref_line = update_plot_live(arm, arm_line, arm.target_state, ref_line)
        volts = arm.getVolt()
        positions = arm.getPositions()
        voltages = np.concatenate((volts, positions), axis=0)
        time.sleep(arm.dt)  #COMMENT FOR MATPLOT
        #plt.draw()
        #plt.pause(arm.dt)


arm = None


def send_to_roborio(data, roborio_ip, roborio_port):
    """
    Send a given JSON to the roboRIO, and receive the response from the roboRIO.
    Any specific value from the response JSON will be set up here.
    :param data: JSON being sent to the RIO.
    :param roborio_ip: depending on if sim or not (10.1.35.2 or localhost)
    :param roborio_port: almost always 5802.
    """
    global angleShoulder, angleElbow, arm
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((roborio_ip, roborio_port))
        s.sendall((json.dumps(data) + '\n').encode())  #send the data
        try:
            data_from_robot = json.loads(s.recv(1024).decode())  #receive response, MUST be AFTER send
            #If you're using a confirmed send (response contains a marker) you'll need an else on your if check.
            if "DoubleJointedEncoders" in data_from_robot:
                encoders_string = data_from_robot["DoubleJointedEncoders"]
                encoders = [float(value) for value in encoders_string.split(",")]
                angleShoulder, angleElbow = encoders[0], encoders[1]
                #If we haven't yet created the arm, do it here.
                if arm is None:
                    arm = initialize_arm_with_encoders()
                    arm_thread = threading.Thread(target=arm_loop, args=(arm,))
                    arm_thread.daemon = True  # Allow the thread to be terminated when the main thread exits
                    arm_thread.start()
                arm.updatePosition(angleShoulder, angleElbow)
                data["gotEncoder"] = str("RECEIVED ENCODERS")
            else:
                if 'gotEncoder' in data:
                    data.pop('gotEncoder')
            if "modelInputs" in data_from_robot:
                m_input_string = data_from_robot["modelInputs"]
                m_input = [float(value) for value in m_input_string.split(",")]
                np_input = np.array(m_input)
                outputs = runValue(np_input)
                data["outputs"] = str(outputs)
            else:
                if 'outputs' in data:
                    data.pop('outputs')
            if "DoubleJointSetpoint" in data_from_robot:
                rawData = data_from_robot["DoubleJointSetpoint"]
                wantedPos = [float(value) for value in rawData.split(",")]
                newState = to_state(wantedPos[0], wantedPos[1], bool(wantedPos[2]), arm)
                arm.set_target_state(newState)
        except Exception:  #we use a global exception case to prevent crashing here, it gives ugly behaviour.
            print("FAILED TO READ")
            pass  #don't crash.


def latest_model():
    """
    Check the directory for models, and get the most recent one.
    :return: latest model like "model_v2.pkl"
    """
    versions = []

    # Create the directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    for filename in os.listdir(directory):
        match = model_naming.match(filename)
        if match:
            versions.append(int(match.group(1)))
    if versions:
        latest_version = max(versions)
        return f'model_v{latest_version}.pkl'
    else:
        return None


def load_latest_model():
    """
    Load the most recent model.
    :return: nothing, but update our model wrapper
    """
    global wrapper
    # Get list of subdirectories in the Models directory
    newest_model = latest_model()

    if not newest_model:
        print("No models found. Do not use runValue.")
    # Sort directories by creation time (modification time of the directory)
    else:
        newest_model_path = os.path.join(directory, newest_model)
        print("Using " + newest_model)
        wrapper = joblib.load(newest_model_path)


def handle_client(conn, addr):
    """
    Handle client connection to PyDriverStation, automatically downloading the model sent from PyDriverStation
    :param conn: what data is being sent
    :param addr: whatever IP and port the client is on (varies from call to call)
    :return: the new model, and save to ../OrangePi/Models/Model_v{index}.pkl
    """
    global wrapper
    print('Connected by', addr)
    data = b''
    while True:
        packet = conn.recv(1024)
        if not packet:
            break
        data += packet
    newest_model = latest_model()
    if newest_model:
        print("Current Latest Model is: " + newest_model)
        latest_version = int(model_naming.match(newest_model).group(1))
        new_version = latest_version + 1
    else:
        new_version = 1
    new_model = f'model_v{new_version}.pkl'
    new_model_path = os.path.join("../OrangePi/ModelsPI/", new_model)
    print(new_model_path)
    # Write data to a file in the specified directory
    with open(new_model_path, 'wb') as temp_file:
        temp_file.write(data)

    # Load the wrapper.pkl file using pickle
    with open(new_model_path, 'rb') as file:
        wrapper = joblib.load(file)


def runValue(value):
    """
    Calculate model output with given value
    :param value: numpy array with inputs, like [4.5,2.4,6]
    :return: numpy array with outputs
    """
    yhat = wrapper.predict([value])
    # summarize the prediction
    # print('Predicted: %s' % yhat[0])
    return yhat[0]


def main():
    HOST = '0.0.0.0'  # Listen on all available interfaces
    PORT = 5801  # Port to listen on for PyDriverStation
    data_to_robot = {'timestamp': '0'}
    roborio_ip = 'localhost'  # Replace with the actual IP address of the roboRIO (10.1.35.2)
    roborio_port = 5802  # Port on which the roboRIO is listening
    heartbeat = 0
    load_latest_model()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        print("BINDING...")
        s.bind((HOST, PORT))
        s.listen()
        s.settimeout(.001)
        print(f'Server listening on {HOST}:{PORT}')
        while True:
            heartbeat += 1
            try:
                conn, addr = s.accept()
                client_thread = threading.Thread(target=handle_client, args=(conn, addr))
                client_thread.start()
            except TimeoutError:
                pass
            except Exception:
                s.close()
                print("CRASHED BECAUSE CONNECTION TERMINATED... REBOOTING")
                raise ConnectionError
            data_to_robot['timestamp'] = str(heartbeat)
            data_to_robot['voltages'] = str(voltages) #3/4 are the wanted positions #5/6 are the (expected) velocities
            send_to_roborio(data_to_robot, roborio_ip, roborio_port)
            #data_to_robot.clear()

            # Send the received data to the roboRIO


# BEFORE YOU CALL MAIN MAKE SURE: (only use localhost if sim)
# 1. HOST (laptop) = 10.1.35.5  OR localhost
# 2. roborio_ip = 10.1.35.2 OR localhost
# 3. You have NOT touched send_to_roborio.
# 4. If you're using the Double Jointed Arm, be SURE to go into DoubleJointedArm.py and tweak "r_pos" DOWN until it
# breaks the REAL ARM, NOT SIM.
# Moving any lines in this function will CRASH THE PI (robot will keep working), I almost promise you.
main()
