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

def draw_trajectory(points, color=[1,0,0], width=2, life_time=0):
    """
    points: list of [x, y, z] positions
    color: RGB (0–1)
    width: line thickness
    life_time: 0 = permanent
    """
    for i in range(len(points)-1):
        p.addUserDebugLine(
            points[i],
            points[i+1],
            color,
            lineWidth=width,
            lifeTime=life_time
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
    
    for _ in range(100):
        ob, *_ = env.step(action)
    
    
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
    ikSolver = 0    
    q0_cart = list(p.getLinkState(robot_id,hand_id)[4])
    
    mpos, mvel, mtorq, names = getMotorJointStates(robot_id) #returns length 13   
    print(f"{mpos}\n")
    print(f"{names}\n")
    
    q1_cart = list(q0_cart)
    q1_cart[0] += 0.4
    
    q2_cart = list(q1_cart)
    q2_cart[2] -= 0.4

    q3_cart = list(q2_cart)
    q3_cart[0] -= 0.4
    
    q4_cart = q0_cart
    
    jointPoses0 = p.calculateInverseKinematics(robot_id,
                      hand_id,
                      q0_cart,
                      solver=ikSolver)

    print(jointPoses0)     
    jointPoses0 = jointPoses0[:7]
    

    #jointPoses0[0] += math.radians(45)
                   
    print(jointPoses0)     
                      
    jointPoses1 = p.calculateInverseKinematics(robot_id,
                      hand_id,
                      q1_cart,
                      solver=ikSolver)[:7]
    jointPoses2 = p.calculateInverseKinematics(robot_id,
                      hand_id,
                      q2_cart,
                      solver=ikSolver)[:7]
    jointPoses3 = p.calculateInverseKinematics(robot_id,
                      hand_id,
                      q3_cart,
                      solver=ikSolver)[:7]
    jointPoses4 = p.calculateInverseKinematics(robot_id,
                      hand_id,
                      q4_cart,
                      solver=ikSolver)[:7]
                      
    jointPoses0 = np.asarray(jointPoses0)
    jointPoses1 = np.asarray(jointPoses1)
    jointPoses2 = np.asarray(jointPoses2)
    jointPoses3= np.asarray(jointPoses3)
    jointPoses4= np.asarray(jointPoses4)


    #for idx in range(len(jointPoses1)):
     #   p.resetJointState(robot_id, idx+7, jointPoses2[idx]);
                  
              
    
    
    
    traj_joint1 = cubic_interpolate(jointPoses0, jointPoses1, 500)
    traj_joint2 = cubic_interpolate(jointPoses1, jointPoses2, 500)
    traj_joint3 = cubic_interpolate(jointPoses2, jointPoses3, 500)
    traj_joint4 = cubic_interpolate(jointPoses3, jointPoses4, 500)
    
    traj_joint1 = np.array(traj_joint1)
    traj_joint2 = np.array(traj_joint2)
    traj_joint3 = np.array(traj_joint3)
    traj_joint4 = np.array(traj_joint4)
    
    traj_joint = np.vstack((traj_joint1,traj_joint2,traj_joint3,traj_joint4))

    print(len(q2_cart))
    
    way_points = []
    way_points.append(q0_cart)
    way_points.append(q1_cart)
    way_points.append(q2_cart)
    way_points.append(q3_cart)
    way_points.append(q0_cart)
    
    draw_trajectory(way_points)
    
    #plot_joint_state(traj_joint)
    
    
    
    

    

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
    
    
    joint_pose_data = []
    
    history = []
    path2D = []
    trajectory = []

    
    for i in range(n_steps):
        ob, *_ = env.step(action)
        history.append(ob)

        action = np.zeros(env.n()) #11
        
        if i < len(traj_joint):
            K = 1
            
            mpos, mvel, mtorq, names = getMotorJointStates(robot_id) #returns length 13  

            current_joint_pos = mpos[:7]
            current_cart = list(p.getLinkState(robot_id,hand_id)[4])

            error_joint = traj_joint[i] - current_joint_pos

            
            cmd_vel = K * error_joint          
            cmd_vel = np.clip(cmd_vel, -1.0, 1.0)

            #print(f"{names[0]}, {jointPoses0[0]}, {mpos[0]}, {error_joint[0]}")

            trajectory.append(current_cart)

            
            action[2:9] = cmd_vel
            

           
            
           
        
        
        
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
