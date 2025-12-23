
import math
import numpy as np
import warnings
import gymnasium as gym
import numpy as np
import pybullet as p

from urdfenvs.robots.generic_urdf.generic_diff_drive_robot import GenericDiffDriveRobot
from urdfenvs.urdf_common.urdf_env import UrdfEnv

prev_lin_vel = 0.0
loc_error_int = 0.0
rot_error_int = 0.0

def limit_acceleration(v_des, v_prev, a_max, dt):
    dv = a_max * dt
    return max(v_prev - dv, min(v_des, v_prev + dv))
    
def calculate_base_vel(waypoint, robot_id):

    global prev_lin_vel, loc_error_int, rot_error_int

    dt = 0.01

    vel_cmd = [0, 0]

    Kp_loc = 0.05
    Ki_loc = 0.1

    Kp_rot = 2
    Ki_rot = 0.5
    
    MAX_LIN_VEL = 1.0
    MAX_LIN_ACC = 0.2  # m/s²
    MAX_LIN_INT = 2.0
    
    MAX_ROT_INT = 2.0
    MAX_ROT_VEL = 2.0
    
    loc_error_threshold = 0.3
    rot_error_threshold_1 = 2.5
    rot_error_threshold_2 = 0.5
    
    base_pose = p.getLinkState(robot_id, 0)
    base_location = np.array(base_pose[4])
    base_orientation = p.getEulerFromQuaternion(base_pose[5])[2]

    error_location = np.linalg.norm(waypoint - base_location)
    ref_orientation = math.atan2(
        waypoint[1] - base_location[1],
        waypoint[0] - base_location[0]
    ) - math.pi / 2

    # Proper angle wrapping
    error_orientation = (ref_orientation - base_orientation + math.pi) % (2 * math.pi) - math.pi

    rot_error_int += error_orientation * dt
    rot_error_int = max(-MAX_ROT_INT, min(rot_error_int, MAX_ROT_INT))

    v_des = (
        Kp_rot * error_orientation +
        Ki_rot * rot_error_int
    )
    vel_cmd[1] = max(-MAX_ROT_VEL, min(v_des, MAX_ROT_VEL))

     
    if (error_location > loc_error_threshold and abs(error_orientation) < math.radians(rot_error_threshold_1)) or \
            (error_location <= loc_error_threshold and abs(error_orientation) < math.radians(rot_error_threshold_2)):
        loc_error_int += error_location * dt
        loc_error_int = max(-MAX_LIN_INT, min(loc_error_int, MAX_LIN_INT))

        v_des = (
            Kp_loc * error_location +
            Ki_loc * loc_error_int
        )
        v_des = min(v_des, MAX_LIN_VEL)

        v_cmd = limit_acceleration(v_des, prev_lin_vel, MAX_LIN_ACC, dt)

        vel_cmd[0] = -v_cmd
        prev_lin_vel = v_cmd
    else:
        prev_lin_vel = 0.0
        loc_error_int = 0.0

    #print(f"err rot: {math.degrees(error_orientation):.2f}, " f"err loc: {error_location:.2f}, " f"vel rot: {vel_cmd[1]:.2f}, " f"vel loc: {vel_cmd[0]:.2f}, ")  

    return vel_cmd

    
