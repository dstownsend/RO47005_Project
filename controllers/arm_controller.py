import warnings
import gymnasium as gym
import numpy as np

import math

from global_planners.arm_helpers import getMotorJointStates

import pybullet as p


class ArmController():
    
    def __init__(self):

        self.ee_id = 15
        self.interp_steps = 50
        self.path = None
        self.step_num = 0
        self.Kp = 100.0
        self.vel_lim = 2.0
        self.goal_reached = False
    
    def compute_vel_path(self, robot_id):
        action = np.zeros(11) #11
    
        if self.step_num == len(self.path):
            self.goal_reached = True
            return action
    
        mpos, mvel, mtorq, names = getMotorJointStates(robot_id) #returns length 13  
        current_joint_pos = mpos[:7]
        current_cart = list(p.getLinkState(robot_id,self.ee_id)[4])

        error_joint = self.path[self.step_num] - current_joint_pos

        error_mag = np.linalg.norm(np.array(error_joint))

        if error_mag < math.radians(1):
            self.step_num += 1

        cmd_vel = self.Kp * error_joint  
        cmd_vel = np.clip(cmd_vel, -self.vel_lim, self.vel_lim)
        
        action[2:9] = cmd_vel

        return action
		
    def compute_vel_single(self, robot_id, desired_joint_pos):
        action = np.zeros(11) #11
        
        mpos, mvel, mtorq, names = getMotorJointStates(robot_id) #returns length 13  
        current_joint_pos = mpos[:7]
        current_cart = list(p.getLinkState(robot_id,self.ee_id)[4])

        error_joint = np.asarray(desired_joint_pos) - np.asarray(current_joint_pos)

        cmd_vel = self.Kp * error_joint  
        cmd_vel = np.clip(cmd_vel, -self.vel_lim, self.vel_lim)
        
        action[2:9] = cmd_vel

        return action
		
