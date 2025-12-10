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
  joint_positions = [state[0] for state in joint_states]
  joint_velocities = [state[1] for state in joint_states]
  joint_torques = [state[3] for state in joint_states]
  return joint_positions, joint_velocities, joint_torques

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

def update_sphere(robot_id, joint_index, sphere_id):
    # Get world pose of the link attached to the joint
    link_state = p.getLinkState(robot_id, joint_index)
    pos = link_state[4]  # world position
    orn = link_state[5]  # world orientation

    # Move the sphere to follow the joint
    p.resetBasePositionAndOrientation(sphere_id, pos, orn)


### Path Generation ###
def linear_interpolate(q0, q1, n_steps):
    q0 = np.array(q0, dtype=float)
    q1 = np.array(q1, dtype=float)
    traj = [(q0 + (q1 - q0) * (i / (n_steps-1))) for i in range(n_steps) ]
    return traj

### Velocity Control ###
def compute_joint_vel(q_current, q_desired, dt, max_vel=1.0):
    """Convert joint angle error into joint velocity command."""
    q_current = np.array(q_current)
    q_desired = np.array(q_desired)
    vel = (q_desired - q_current) / dt
    # clip velocities to safe values
    return np.clip(vel, -max_vel, max_vel)




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
    
    ob = env.reset(
        pos=np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.5, 0.0, 1.8, 0.5])
    )   
    print(f"Initial observation : {ob}")
    
    robot_id = env._robots[0]._robot  
        
    action = np.zeros(env.n())
    
    #getJointStates A1 to A7 = 7 to 14   
    #action A1 to A7 = 2 to 9  
    joint_indices = range(7,15)

    num_joints = 24
    joint_num = 1
    hand_id = 15
    
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
    
    ## Single joint action   
    #action[joint_num + 1] = -0.5
    
    ## Linear trajectory   
    q0 = [0.0, 0.0, 0.0, -1.5, 0.0, 1.8, 0.5]
    
    q0_cart = list(p.getLinkState(robot_id,hand_id)[4])
    
    print(f"{q0_cart}\n")    
    
    q1_cart = list(q0_cart)
    q1_cart[0] += 0.4

    
    q2_cart = list(q1_cart)
    q2_cart[2] -= 0.4

    q3_cart = list(q2_cart)
    q3_cart[0] -= 0.4
    
    q4_cart = q1_cart
    
    traj_cart1 = linear_interpolate(q0_cart, q1_cart, 1000)
    traj_cart2 = linear_interpolate(q1_cart, q2_cart, 1000)
    traj_cart3 = linear_interpolate(q2_cart, q3_cart, 1000)
    traj_cart4 = linear_interpolate(q3_cart, q4_cart, 1000)
    
    traj_cart1 = np.array(traj_cart1)
    traj_cart2 = np.array(traj_cart2)
    traj_cart3 = np.array(traj_cart3)
    traj_cart4 = np.array(traj_cart4)
    
    traj_cart = np.vstack((traj_cart1,traj_cart2,traj_cart3,traj_cart4))

    print(traj_cart.shape)

## Visualisation
    start_pos = p.createVisualShape(
        shapeType=p.GEOM_SPHERE,
        radius=0.04,
        rgbaColor=[0, 1, 0, 1]
    )
    start_pos_id = p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=start_pos,
        basePosition=q0_cart,
        useMaximalCoordinates=False
    )
    
    mid_pos = p.createVisualShape(
        shapeType=p.GEOM_SPHERE,
        radius=0.04,
        rgbaColor=[0, 0, 1, 1]
    )
    mid_pos_id = p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=mid_pos,
        basePosition=q1_cart,
        useMaximalCoordinates=False
    )
    
    end_pos = p.createVisualShape(
        shapeType=p.GEOM_SPHERE,
        radius=0.04,
        rgbaColor=[1, 0, 0, 1]
    )
    end_pos_id = p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=end_pos,
        basePosition=q2_cart,
        useMaximalCoordinates=False
    )
    
    
    
    history = []
    path2D = []
    for i in range(n_steps):
        ob, *_ = env.step(action)
        history.append(ob)

        pos_rad = p.getJointStates(robot_id,[6+joint_num])[0][0]
        pos_deg = math.degrees(pos_rad)
        
        action = np.zeros(env.n()) #11
        if i < len(traj_cart):
            
            K = 10.0
            
            current_cart = list(p.getLinkState(robot_id,hand_id)[4])
            #error_cart = np.array(q1_cart) - current_cart            
            error_cart = traj_cart[i] - current_cart
            
            cmd_vel = K * error_cart           
            
            
            
            pos, vel, torqs = getJointStates(robot_id) #returns length 25            
            mpos, mvel, mtorq = getMotorJointStates(robot_id) #returns length 13            
            result = p.getLinkState(robot_id,
                                    hand_id,
                                    computeLinkVelocity=1,
                                    computeForwardKinematics=1)
            link_trn, link_rot, com_trn, com_rot, frame_pos, frame_rot, link_vt, link_vr = result
            zero_vec = [0.0] * len(mpos)
            jac_t, jac_r = p.calculateJacobian(robot_id, hand_id, com_trn, mpos, zero_vec, zero_vec) #19
            
           
            J = np.array(jac_t)
            dq = np.linalg.pinv(J) @ cmd_vel
            #dq = dq[-8:]
            print()            
            print(dq)

            action[2:7] = dq[2:7]
           
            
            
            #q_desired = p.calculateInverseKinematics(robot_id, hand_id, traj_cart[i])
            #q_desired = q_desired[5:]
            
            #q_current = [p.getJointState(robot_id, j)[0] for j in joint_indices]
            
            #joint_velocities = compute_joint_vel(q_current, q_desired, env.dt, max_vel=1.5)
            
            #print(joint_velocities)

            #action[2:10] = joint_velocities

        
        
        #hand_pos = p.getLinkState(robot_id,hand_id)
        #path2D.append(hand_pos[4][0:2])
       	#if pos_deg < -90:
       	#    break
        #print(f"{pos_deg}")
        
        
        
        ## Update stage
        update_sphere(robot_id, hand_id, sphere_id)
            
    #plot_2D_path(np.array(path2D))	

    	
    env.close()
    return history


if __name__ == "__main__":
    show_warnings = False
    warning_flag = "default" if show_warnings else "ignore"
    with warnings.catch_warnings():
        warnings.filterwarnings(warning_flag)
        run_albert(render=True)
