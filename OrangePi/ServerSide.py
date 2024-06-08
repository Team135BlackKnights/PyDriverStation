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

model = GradientBoostingRegressor(n_estimators=100)
wrapper = MultiOutputRegressor(model)

model_naming = re.compile(r"model_v(\d+)\.pkl")
directory = "../OrangePi/ModelsPI/"
MOTOR_KRAKEN_X60_FOC = DcBrushedMotor(12.0, 9.36, 476.1, 2, 6000.0)  # create Kraken FOC
dt = 0.02
length1 = 46.25 * .0254  # in meters, so .0254
length2 = 41.8 * .0254
# Mass of segments
mass1 = 9.34 * 0.4536
mass2 = 9.77 * 0.453

# Distance from pivot to CG for each segment
pivot_to_CG1 = 21.64 * 0.0254
pivot_to_CG2 = 26.70 * 0.0254

# Moment of inertia about CG for each segment
MOI1 = 2957.05 * 0.0254 * 0.0254 * 0.4536
MOI2 = 2824.70 * 0.0254 * 0.0254 * 0.4536

# Gearing of each segment
gearing1 = 70.0
gearing2 = 45.0

# Number of motors in each gearbox
motor_count1 = 1
motor_count2 = 2
# Motor Type. ALL ARM MOTORS MUST BE SAME
motor_type = MOTOR_KRAKEN_X60_FOC
# Gravity
gravity = 9.806

#Real controls and outputs.
voltages = np.zeros(2)
startingAngleShoulder = 0  #placeholders.
startingAngleElbow = 0  #placeholders.
startingVelocityShoulder = 0  #placeholders.
startingVelocityElbow = 0  #placeholders.
angleShoulder = 0
angleElbow = 0
velocityShoulder = 0
velocityElbow = 0


def to_state(x, y, invert, arm):

    theta1, theta2 = arm.constants.inv_kinematics(x, y, invert)
    return np.array([[theta1], [theta2], [0], [0]])


def initialize_arm():
    """
    Initialize the arm.
    """
    return DoubleJointedArm(dt, length1, length2, mass1, mass2, pivot_to_CG1, pivot_to_CG2, MOI1, MOI2, gearing1,
                            gearing2, motor_count1, motor_count2, motor_type, gravity)


def initialize_arm_with_encoders():
    """
    Initialize the arm's state based on encoder readings.
    """
    arm = initialize_arm()

    # Set initial state based on encoder values
    arm.x = np.array([[startingAngleShoulder], [startingAngleElbow], [startingVelocityShoulder],
                      [startingVelocityElbow], [0], [0]])  # Initial state [angle1, angle2, velocity1, velocity2]
    arm.observer.x_hat = arm.x

    return arm


arm = initialize_arm_with_encoders()
# Initialize plot
def arm_loop(arm):
    global voltages
    #fig, ax, arm_line, ref_line = initialize_plot_live(arm)
    while True:
        arm.update()
        #arm_line, ref_line = update_plot_live(arm, arm_line, arm.target_state, ref_line)
        volts = arm.getVolt()
        positions = arm.getPositions()
        voltages = np.concatenate((volts, positions), axis=0)
        time.sleep(arm.dt)
        #plt.draw()
        #plt.pause(arm.dt)


def send_to_roborio(data, roborio_ip, roborio_port):
    global angleShoulder, angleElbow, velocityShoulder, velocityElbow
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((roborio_ip, roborio_port))
        s.sendall((json.dumps(data) + '\n').encode())
        data_from_robot = json.loads(s.recv(1024).decode())
        try:
            if "DoubleJointedEncoders" in data_from_robot:
                encoders_string = data_from_robot["DoubleJointedEncoders"]
                encoders = [float(value) for value in encoders_string.split(",")]
                arm.updatePosition(encoders[0], encoders[1])
                data["gotEncoder"] = str("RECEIVED ENCODERS")
                #arm.updateEncoders(encoders[0], encoders[1], encoders[2], encoders[3])
            else:
                if 'gotEncoder' in data:
                    data.pop('gotEncoder')
            if "modelDistance" in data_from_robot:
                m_distance = float(data_from_robot["modelDistance"])
                timeOld = time.time()
                outputs = runValue(m_distance)
                print("time to get val:" + str(time.time() - timeOld))
                data["outputs"] = str(outputs)
            else:
                if 'outputs' in data:
                    data.pop('outputs')
            if "DoubleJointSetpoint" in data_from_robot:
                rawData = data_from_robot["DoubleJointSetpoint"]
                wantedPos = [float(value) for value in rawData.split(",")]
                newState = to_state(wantedPos[0], wantedPos[1], bool(wantedPos[2]), arm)
                arm.set_target_state(newState)
        except Exception:
            print("FAILED TO READ")
            pass


def latest_model():
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
    global wrapper
    print('Connected by', addr)
    data = b''
    while True:
        packet = conn.recv(1024)
        if not packet:
            break
        data += packet
    newest_model = latest_model()
    print(newest_model)
    if newest_model:
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
    row = [[value]]
    yhat = wrapper.predict(row)
    # summarize the prediction
    # print('Predicted: %s' % yhat[0])
    return yhat[0]


def main():
    HOST = '0.0.0.0'  # Listen on all available interfaces
    PORT = 5801  # Port to listen on
    data_to_robot = {'timestamp': '0'}
    roborio_ip = 'localhost'  # Replace with the actual IP address of the roboRIO
    roborio_port = 5802  # Port on which the roboRIO is listening
    timestamp = 0
    load_latest_model()
    #go_to_state(arm, state2)
    #go_to_state(arm, state3, .1)
    arm_thread = threading.Thread(target=arm_loop, args=(arm,))
    arm_thread.daemon = True  # Allow the thread to be terminated when the main thread exits
    arm_thread.start()
    #arm_thread = threading.Thread(target=run_arm_simulation, args=(arm, t_rec, traj))
    #arm_thread.start()
    print("BINDING...")
    time.sleep(1)  #wait for RIO to boot.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        print("BINDING...")
        s.bind((HOST, PORT))
        s.listen()
        s.settimeout(.001)
        print(f'Server listening on {HOST}:{PORT}')
        while True:
            timestamp += 1
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
            data_to_robot['timestamp'] = str(timestamp)
            data_to_robot['voltages'] = str(voltages)
            send_to_roborio(data_to_robot, roborio_ip, roborio_port)
            #data_to_robot.clear()

            # Send the received data to the roboRIO


# BEFORE YOU CALL MAIN MAKE SURE: (only use localhost if sim)
# 1. HOST (laptop) = 10.1.35.5  OR localhost
# 2. roborio_ip = 10.1.35.2 OR localhost
# 3. You have NOT touched send_to_roborio.
# Moving any lines in this function will CRASH THE PI (robot will keep working), I almost promise you.
main()
