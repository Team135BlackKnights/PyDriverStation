#!/usr/bin/env python3

"""frccontrol example for a double-jointed arm."""

import sys

import matplotlib as mpl
from frccontrol import DcBrushedMotor
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
import time

import frccontrol as fct

if "--noninteractive" not in sys.argv:
    print("NON")
    mpl.use("TkAgg")


class DoubleJointedArm:
    """
    An frccontrol system representing a double-jointed arm.

    States: [joint angle 1, joint angle 2,
             joint angular velocity 1, joint angular velocity 2,
             input error 1, input error 2]
    Inputs: [voltage 1, voltage 2]
    """

    def __init__(self, dt, length1, length2, mass1, mass2, pivot_to_CG1, pivot_to_CG2, MOI1, MOI2, gearing1, gearing2,
                 motor_count1, motor_count2, motor_type, gravity):
        """
        Double-jointed arm subsystem.

        Keyword arguments:
        dt -- simulation step time
        """
        self.dt = dt

        self.constants = DoubleJointedArmConstants(length1, length2, mass1, mass2, pivot_to_CG1, pivot_to_CG2, MOI1,
                                                   MOI2, gearing1, gearing2, motor_count1, motor_count2, motor_type,
                                                   gravity)

        q_pos = 0.01745
        q_vel = 0.1745329
        q_error = 10
        r_pos = 0.05
        self.observer = fct.ExtendedKalmanFilter(
            6,
            2,
            self.f,
            self.h,
            [q_pos, q_pos, q_vel, q_vel, q_error, q_error],
            [r_pos, r_pos],
            self.dt,
        )

        self.x = np.zeros((6, 1))
        self.u = np.zeros((2, 1))
        self.y = np.zeros((2, 1))
        self.target_state = np.zeros((6, 1))
        self.u_min = np.array([[-12.0]])
        self.u_max = np.array([[12.0]])

    # pragma pylint: disable=unused-argument
    def set_target_state(self, target_state):
        """
        Set a new target state for the arm.

        Keyword arguments:
        target_state -- the new target state as a 6x1 numpy array
        """
        self.target_state = target_state

    def update(self):
        """
        Advance the model by one timestep.

        Keyword arguments:
        r -- the current reference
        next_r -- the next reference
        """
        self.x = fct.rkdp(self.f, self.x, self.u, self.dt)
        self.y = self.h(self.x, self.u)
        # self.y += np.array(
        #     [np.random.multivariate_normal(mean=[0, 0], cov=np.diag([1e-4, 1e-4]))]
        # ).T

        self.observer.predict(self.u, self.dt)
        self.observer.correct(self.u, self.y)

        u_ff = self.feedforward(self.observer.x_hat, np.zeros((6, 1)))

        A = fct.numerical_jacobian_x(
            6,
            6,
            self.f,
            self.observer.x_hat,
            self.feedforward(self.observer.x_hat, np.zeros((6, 1))),
        )
        B = fct.numerical_jacobian_u(
            6,
            2,
            self.f,
            self.observer.x_hat,
            self.feedforward(self.observer.x_hat, np.zeros((6, 1))),
        )
        u_fb = fct.LinearQuadraticRegulator(
            A[:4, :4],
            B[:4, :],
            [0.01745, 0.01745, 0.08726, 0.08726],
            [12.0, 12.0],
            self.dt,
        ).K @ (self.target_state[0:4] - self.observer.x_hat[0:4])

        # Voltage output from input error estimation. The estimate is in
        # Newton-meters, so has to be converted to volts.
        u_err = np.linalg.solve(self.constants.B, self.observer.x_hat[4:])

        self.u = np.clip(u_ff + u_fb - u_err, self.u_min, self.u_max)

    def updateEncoders(self, angle1, angle2, anglevelocity1, anglevelocity2):
        """
        Update the arm's matrices based on given angles and angular velocities.

        Keyword arguments:
        angle1 -- angle of joint 1 (in radians)
        angle2 -- angle of joint 2 (in radians)
        anglevelocity1 -- angular velocity of joint 1 (in radians/second)
        anglevelocity2 -- angular velocity of joint 2 (in radians/second)
        """
        self.x[:4] = np.array([[angle1], [angle2], [anglevelocity1], [anglevelocity2]])
    def updatePosition(self, angle1,angle2):
        self.x[:2] = np.array([[angle1], [angle2]])
        # Update the observer's estimated state x_hat
        #self.observer.x_hat[:4] = self.x[:4]

    def get_dynamics_matrices(self, x):
        """Gets the dynamics matrices for the given state.

        See derivation at:
        https://www.chiefdelphi.com/t/whitepaper-two-jointed-arm-dynamics/423060

        Keyword arguments:
        x -- current system state
        """
        theta1, theta2, omega1, omega2 = x[:4].flat
        c2 = np.cos(theta2)

        l1 = self.constants.l1
        r1 = self.constants.r1
        r2 = self.constants.r2
        m1 = self.constants.m1
        m2 = self.constants.m2
        I1 = self.constants.I1
        I2 = self.constants.I2
        g = self.constants.g

        hM = l1 * r2 * c2
        M = (
                m1 * np.array([[r1 * r1, 0], [0, 0]])
                + m2
                * np.array(
            [[l1 * l1 + r2 * r2 + 2 * hM, r2 * r2 + hM], [r2 * r2 + hM, r2 * r2]]
        )
                + I1 * np.array([[1, 0], [0, 0]])
                + I2 * np.array([[1, 1], [1, 1]])
        )

        hC = -m2 * l1 * r2 * np.sin(theta2)
        C = np.array([[hC * omega2, hC * omega1 + hC * omega2], [-hC * omega1, 0]])

        G = (
                g * np.cos(theta1) * np.array([[m1 * r1 + m2 * self.constants.l1, 0]]).T
                + g * np.cos(theta1 + theta2) * np.array([[m2 * r2, m2 * r2]]).T
        )

        return M, C, G

    def f(self, x, u):
        """
        Dynamics model.

        Keyword arguments:
        x -- state vector
        u -- input vector
        """
        M, C, G = self.get_dynamics_matrices(x)

        omega = x[2:4]

        # Motor dynamics
        torque = self.constants.A @ x[2:4] + self.constants.B @ u

        # dx/dt = [ω α 0]ᵀ
        return np.block(
            [[x[2:4]], [np.linalg.solve(M, torque - C @ omega - G)], [np.zeros((2, 1))]]
        )

    def getVolt(self):
        return self.u
    def getPositions(self):
        return self.x[0:2]

    def h(self, x, u):
        """
        Measurement model.

        Keyword arguments:
        x -- state vector
        u -- input vector
        """
        return x[:2, :]

    def feedforward(self, x, xdot):
        """
        Arm feedforward.
        """
        M, C, G = self.get_dynamics_matrices(x)
        omega = x[2:4]
        return np.linalg.solve(
            self.constants.B, M @ xdot[2:4] + C @ omega + G - self.constants.A @ omega
        )


class DoubleJointedArmConstants:
    """
    Double-jointed arm model constants.
    """

    def __init__(self, length1, length2, mass1, mass2, pivot_to_CG1, pivot_to_CG2, MOI1, MOI2, gearing1, gearing2,
                 motor_count1, motor_count2, motor_type, gravity):
        # Length of segments
        self.l1 = length1
        self.l2 = length2

        # Mass of segments
        self.m1 = mass1
        self.m2 = mass2

        # Distance from pivot to CG for each segment
        self.r1 = pivot_to_CG1
        self.r2 = pivot_to_CG2

        # Moment of inertia about CG for each segment
        self.I1 = MOI1
        self.I2 = MOI2

        # Gearing of each segment
        self.G1 = gearing1
        self.G2 = gearing2

        # Number of motors in each gearbox
        self.N1 = motor_count1
        self.N2 = motor_count2

        # Gravity
        self.g = gravity

        motor = motor_type

        # torque = A * velocity + B * voltage
        self.B = (
                np.array([[self.N1 * self.G1, 0], [0, self.N2 * self.G2]])
                * motor.Kt
                / motor.R
        )
        self.A = (
                -np.array(
                    [[self.G1 * self.G1 * self.N1, 0], [0, self.G2 * self.G2 * self.N2]]
                )
                * motor.Kt
                / motor.Kv
                / motor.R
        )

    def fwd_kinematics(self, x):
        """
        Forward kinematics for the given state.
        """
        theta1, theta2 = x[:2]

        joint2 = np.array([self.l1 * np.cos(theta1), self.l1 * np.sin(theta1)])
        end_eff = joint2 + np.array(
            [self.l2 * np.cos(theta1 + theta2), self.l2 * np.sin(theta1 + theta2)]
        )

        return joint2, end_eff

    def inv_kinematics(self, x, y, invert=False):
        """
        Inverse kinematics for a target position pos (x, y). The invert flag
        controls elbow direction.

        Keyword arguments:
        x -- x position
        y -- y position

        Returns:
        theta1 -- joint 1 angle
        theta2 -- joint 2 angle
        """
        theta2 = np.arccos(
            (x * x + y * y - (self.l1 * self.l1 + self.l2 * self.l2))
            / (2 * self.l1 * self.l2)
        )

        if invert:
            theta2 = -theta2

        theta1 = np.arctan2(y, x) - np.arctan2(
            self.l2 * np.sin(theta2), self.l1 + self.l2 * np.cos(theta2)
        )

        return theta1, theta2


def main():
    """Entry point."""

    dt = 0.02
    MOTOR_KRAKEN_X60_FOC = DcBrushedMotor(12.0, 9.36, 476.1, 2, 6000.0)  # create Kraken FOC
    dt = 0.02
    length1 = 46.25 * .0254  # in meters, so .0254
    length2 = 41.8 * .0254
    # Mass of segments
    mass1 = 9.34 * 0.4536
    mass2 = 9.77 * 0.4536

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
    motor_type = fct.MOTOR_FALCON_500
    # Gravity
    gravity = 9.806
    double_jointed_arm = DoubleJointedArm(dt, length1, length2, mass1, mass2, pivot_to_CG1, pivot_to_CG2, MOI1, MOI2,
                                          gearing1,
                                          gearing2, motor_count1, motor_count2, motor_type, gravity)

    def to_state(x, y, invert):
        theta1, theta2 = double_jointed_arm.constants.inv_kinematics(x, y, invert)
        return np.array([[theta1], [theta2], [0], [0], [0], [0]])

    state1 = to_state(1.5, -1, False)
    state2 = to_state(1.5, 1, True)
    state3 = to_state(-1.8, 1, False)

    double_jointed_arm.x = state1
    double_jointed_arm.observer.x_hat = state1
    double_jointed_arm.set_target_state(state2)
    fig, ax, arm_line, ref_line = initialize_plot_live(double_jointed_arm)
    while True:
        double_jointed_arm.update()
        arm_line = update_plot_live(double_jointed_arm, arm_line, double_jointed_arm.target_state, ref_line)
        print("Current state:", double_jointed_arm.u)
        plt.draw()
        plt.pause(dt)


def get_arm_joints_live(constants, state):
    """
    Get the x-y positions of all three robot joints: base, elbow, and end effector.
    """
    joint_pos, eff_pos = constants.fwd_kinematics(state)
    return np.array([0, joint_pos[0, 0], eff_pos[0, 0]]), np.array(
        [0, joint_pos[1, 0], eff_pos[1, 0]]
    )


def update_plot_live(arm, arm_line, ref_state, ref_line):
    xs, ys = get_arm_joints_live(arm.constants, arm.x[:2])
    xr, yr = get_arm_joints_live(arm.constants, ref_state[:2])
    arm_line.set_data(xs, ys)
    ref_line.set_data(xr, yr)
    return arm_line, ref_line


def initialize_plot_live(arm):
    """
    Initialize the plot for live simulation.
    """
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.axis("square")

    total_len = arm.constants.l1 + arm.constants.l2
    ax.set_xlim(-total_len, total_len)
    ax.set_ylim(-total_len, total_len)

    # Initial arm state
    xs, ys = get_arm_joints_live(arm.constants, arm.x[:4])
    arm_line, = ax.plot(xs, ys, "-o", label="State")

    # Initial reference state
    xs, ys = get_arm_joints_live(arm.constants, arm.target_state[:4])
    ref_line, = ax.plot(xs, ys, "--o", label="Setpoint")

    ax.legend(loc="lower left")
    return fig, ax, arm_line, ref_line


if __name__ == "__main__":
    main()