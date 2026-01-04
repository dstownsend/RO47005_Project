import warnings
import gymnasium as gym
import numpy as np

import math
import time

import pybullet as p
	
def getJointStates(robot_id):
	joint_states = p.getJointStates(robot_id, range(p.getNumJoints(robot_id)))
	joint_positions = [state[0] for state in joint_states]
	joint_velocities = [state[1] for state in joint_states]
	joint_torques = [state[3] for state in joint_states]

	return joint_positions, joint_velocities, joint_torques
        
def getMotorJointStates(robot_id):
    joint_states = p.getJointStates(robot_id, range(p.getNumJoints(robot_id)))
    joint_infos = [p.getJointInfo(robot_id, i) for i in range(p.getNumJoints(robot_id))]
    joint_states = [j for j, i in zip(joint_states, joint_infos) if i[3] > -1]
    joint_names = [joint[1] for joint in joint_infos if joint[3] > -1]
    joint_positions = [state[0] for state in joint_states]
    joint_velocities = [state[1] for state in joint_states]
    joint_torques = [state[3] for state in joint_states]

    return joint_positions, joint_velocities, joint_torques, joint_names
    
    
def getMotorJointLimits(robot):
    joint_infos = [p.getJointInfo(robot, i) for i in range(p.getNumJoints(robot))]
    joint_names = [joint[1] for joint in joint_infos if joint[3] > -1]
    joint_limits= [(joint[8],joint[9]) for joint in joint_infos if joint[3] > -1]

    return joint_limits, joint_names
    
## Visualisation
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
    
    
    
    
    
    
