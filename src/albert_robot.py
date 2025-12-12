import warnings
import gymnasium as gym
import numpy as np
from urdfenvs.robots.generic_urdf.generic_diff_drive_robot import GenericDiffDriveRobot
from urdfenvs.urdf_common.urdf_env import UrdfEnv
from urdfenvs.scene_examples.goal import *


import matplotlib
matplotlib.use("TkAgg")   # Force an interactive backend
import matplotlib.pyplot as plt

import math
import time

import pybullet as p

def getJointStates(robot):
  joint_states = p.getJointStates(robot, range(p.getNumJoints(robot)))
  joint_positions = [state[0] for state in joint_states]
  joint_velocities = [state[1] for state in joint_states]
  joint_torques = [state[3] for state in joint_states]
  return joint_positions, joint_velocities, joint_torques


def getMotorJointStates(robot):
  joint_states = p.getJointStates(robot, range(p.getNumJoints(robot)))
  joint_infos = [p.getJointInfo(robot, i) for i in range(p.getNumJoints(robot))]
  joint_states = [j for j, i in zip(joint_states, joint_infos) if i[3] > -1]
  joint_names = [joint[1] for joint in joint_infos if joint[3] > -1]
  joint_positions = [state[0] for state in joint_states]
  joint_velocities = [state[1] for state in joint_states]
  joint_torques = [state[3] for state in joint_states]

  return joint_positions, joint_velocities, joint_torques, joint_names
  
  
  
### RRT Functions ###
def getMotorJointLimits(robot):
  joint_infos = [p.getJointInfo(robot, i) for i in range(p.getNumJoints(robot))]
  joint_names = [joint[1] for joint in joint_infos if joint[3] > -1]
  joint_limits= [(joint[8],joint[9]) for joint in joint_infos if joint[3] > -1]

  return joint_limits, joint_names

# Gen sample 

# Gen sample goal 

# Check collisions   
  
# Find nearest node
  
class TreeNode(object):
    def __init__(self, config, parent=None):
        self.parent = parent
        self.config = config
  
    def retrace(self):
        sequence = []
        node = self
        while node is not None:
            sequence.append(node.config)
            node = node.parent
        return sequence[::-1]


### Supporting Functions ###
def plot_2D_path(path):
    xs = path[:, 0]
    ys = path[:, 1]
    
    plt.figure(figsize=(6, 6))
    plt.plot(xs, ys, 'o-', label="Path")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_joint_state(joint_data):
    joint_data = np.asarray(joint_data)
    plt.figure(figsize=(6, 6))
    xs = range(joint_data.shape[0])
    
    for i in range(joint_data.shape[1]):
        if i > 2:
            plt.plot(xs, joint_data[:,i], 'o-', label=i)
    plt.grid(True)
    plt.legend()
    plt.show()

def update_sphere(robot_id, joint_index, sphere_id):
    # Get world pose of the link attached to the joint
    link_state = p.getLinkState(robot_id, joint_index)
    pos = link_state[4]  # world position
    orn = link_state[5]  # world orientation

    # Move the sphere to follow the joint
    p.resetBasePositionAndOrientation(sphere_id, pos, orn)

def draw_line(points, color=[1,0,0], width=2, life_time=0):
    for i in range(len(points)-1):
        p.addUserDebugLine(
            points[i],
            points[i+1],
            color,
            lineWidth=width,
            lifeTime=life_time
        )

def draw_waypoint(points, color=[0,0,1], pointSize=10):
    p.addUserDebugPoints(
        pointPositions=points,
        pointColorsRGB=[color] * len(points),
        pointSize=pointSize,
        lifeTime=0
    )


        
        


### Path Generation ###
def linear_interpolate(q0, q1, n_steps):
    q0 = np.array(q0, dtype=float)
    q1 = np.array(q1, dtype=float)
    traj = [(q0 + (q1 - q0) * (i / (n_steps-1))) for i in range(n_steps) ]
    return traj
    
    
def cubic_interpolate(q0, q1, n_steps):
    q0 = np.asarray(q0, dtype=float)
    q1 = np.asarray(q1, dtype=float)

    result = []
    for i in range(n_steps):
        t = i / (n_steps - 1)
        h = 3 * t**2 - 2 * t**3  # cubic smoothstep
        qi = (1 - h) * q0 + h * q1
        result.append(qi)

    return result
    
    
def plan_joint_path(way_points_cart, robot_id, ee_id, interp_steps):
    ikSolver = 0
    joint_poses = []
    joint_path_segments = []
    
    for i in range(len(way_points_cart)):

        joint_pose = p.calculateInverseKinematics(robot_id,
                      ee_id,
                      way_points_cart[i],
                      solver=ikSolver)[:7]
                      
        joint_pose = np.array(joint_pose)
        joint_poses.append(joint_pose)

    for i in range(len(joint_poses)-1):
        joint_path_segment = cubic_interpolate(joint_poses[i], joint_poses[i+1], interp_steps)
        
        joint_path_segment = np.array(joint_path_segment)
        joint_path_segments.append(joint_path_segment)

    joint_path = np.vstack(joint_path_segments)
    
    return joint_path



def run_albert(n_steps=10000, render=False, goal=True, obstacles=True):
    robots = [
        GenericDiffDriveRobot(
            urdf="albert.urdf",
            mode="vel",
            actuated_wheels=["wheel_right_joint", "wheel_left_joint"],
            castor_wheels=["rotacastor_right_joint", "rotacastor_left_joint"],
            wheel_radius = 0.08,
            wheel_distance = 0.494,
            spawn_rotation = 0,
            facing_direction = '-y',
        ),
    ]
    env: UrdfEnv = UrdfEnv(
        dt=0.01, robots=robots, render=render
    )
    dt=0.01
    ob = env.reset(
        pos=np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.5, 0.0, 1.8, 0.5])
    )   
    
    robot_id = env._robots[0]._robot         
        
    action = np.zeros(env.n())
    
    for _ in range(100):
        ob, *_ = env.step(action)
    
    joint_home_pose = [0.0, math.radians(-85), 0.0, math.radians(-160), 0.0, 1.8, 0.5]
    
    for idx in range(len(joint_home_pose)):
        p.resetJointState(robot_id, idx+7, joint_home_pose[idx]);
    
    
    for _ in range(100):
        ob, *_ = env.step(action)
        
    #getJointStates A1 to A7 = 7 to 14   
    #action A1 to A7 = 2 to 9  
    joint_indices = range(7,15)





    
## Visualisations
    sphere_visual = p.createVisualShape(
        shapeType=p.GEOM_SPHERE,
        radius=0.03,
        rgbaColor=[1, 0, 0, 1]
    )
    sphere_id = p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=sphere_visual,
        basePosition=[0, 0, 0],
        useMaximalCoordinates=False
    )
    
    
    for j in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, j)
        print(j, info[1].decode("utf-8"), info[2])
    

    
    ## Linear trajectory  
    
    mpos, mvel, mtorq, names = getMotorJointStates(robot_id) #returns length 13   
    print(f"{mpos}\n")
    print(f"{names}\n")
    
    limits, _ = getMotorJointLimits(robot_id)
    limits = limits[:7]
    print(f"{limits}\n")
    
    
    
    ########
    hand_id = 15
    
    
    qh_cart = list(p.getLinkState(robot_id,hand_id)[4])
       
    q0_cart = list([-0.2, -0.4, 1.5])
    
    q1_cart = list(q0_cart)
    q1_cart[0] += 0.4
    
    q2_cart = list(q1_cart)
    q2_cart[2] -= 0.4

    q3_cart = list(q2_cart)
    q3_cart[0] -= 0.4
    
    q4_cart = q0_cart
    
    way_points = []
    way_points.append(q0_cart)
    way_points.append(q1_cart)
    way_points.append(q2_cart)
    way_points.append(q3_cart)
    way_points.append(q0_cart)
    
    interp_steps = 100
    
    joint_path = plan_joint_path(way_points, robot_id, hand_id, interp_steps)
    draw_line(way_points)
    draw_waypoint(way_points)
    
    ########
    
    
    
     
    history = []
    path2D = []
    trajectory = []

    step_num = 0
    Kp = 10.0

    vel_lim = 2.0
    
    error_int = 0.0
    error_prev = 0.0
    
    for i in range(n_steps):
        action = np.zeros(env.n()) #11
    
    
        mpos, mvel, mtorq, names = getMotorJointStates(robot_id) #returns length 13  
        current_joint_pos = mpos[:7]
        current_cart = list(p.getLinkState(robot_id,hand_id)[4])
        
        error_joint = joint_path[step_num] - current_joint_pos
                
        error_mag = np.linalg.norm(np.array(error_joint))
        
        if error_mag < math.radians(1):
            step_num += 1
        
        error_int += (error_prev - error_joint) * dt
        error_prev = error_joint
        
        cmd_vel = Kp * error_joint  
        cmd_vel = np.clip(cmd_vel, -vel_lim, vel_lim)
        
        
        action[2:9] = cmd_vel
        
        print(f"{i}, {step_num}, {math.degrees(error_mag)}, {cmd_vel}") 
        
        ob, *_ = env.step(action)
        history.append(ob)
        update_sphere(robot_id, hand_id, sphere_id)
            

    	
    env.close()
    return history


if __name__ == "__main__":
    show_warnings = False
    warning_flag = "default" if show_warnings else "ignore"
    with warnings.catch_warnings():
        warnings.filterwarnings(warning_flag)
        run_albert(render=True)
