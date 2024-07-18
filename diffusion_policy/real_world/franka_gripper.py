import numpy as np
import franky
from scipy.spatial.transform import Rotation as R
from diffusion_policy.real_world.spacemouse import Spacemouse
import time

# Constants
MOVE_INCREMENT = 0.1
SPEED = 0.05  # [m/s]
FORCE = 20.0  # [N]
DT = 0.01  # Loop interval in seconds

gripper = franky.Gripper("172.16.0.2")


def toggle_gripper_state():
    current_width = gripper.width
    if current_width > 0.01:  # Assuming gripper is open if width > 1cm
        move_future = gripper.move_async(0.0, SPEED)  # Close the gripper
    else:
        move_future = gripper.open_async(SPEED)  # Open the gripper

    # Wait for the movement to complete with a timeout
    if move_future.wait(1):  # Wait for up to 1 second
        if move_future.get():
            print("Toggle gripper state successful")
        else:
            print("Toggle gripper state failed")
    else:
        gripper.stop()
        print("Toggle gripper state timed out")


def grasp_object():
    grasp_future = gripper.grasp_async(
        0.0, SPEED, FORCE, epsilon_inner=0.005, epsilon_outer=1.0)

    if grasp_future.wait(1):  # Wait for up to 1 second
        if grasp_future.get():
            print("Grasp successful")
            return True
        else:
            print("Grasp failed")
            return False
    else:
        gripper.stop()
        print("Grasp timed out")
        return False
