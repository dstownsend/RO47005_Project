from .base import BaseGlobalPlanner

import numpy as np
import pybullet as p

import random
import math
import time

from global_planners.arm_helpers import draw_line, draw_waypoint, getMotorJointStates, getMotorJointLimits, getJointStates



class ArmRRTPlanner(BaseGlobalPlanner):
    
    def __init__(self):

        self.ee_id = 15
        self.max_iterations = 10**8
        self.step_size= math.radians(5)
        self.goal_threshold = 0.1 #[m] cart
        self.ikSolver = 0
        self.goal_bias = 0.5
        self.robot_id = 0
        self.joint_limits = None
        self.base_pose = None
        self.base_orientation = None
        self.tree = None
        
    def plan(self, robot_id, visualise=False):
        self.robot_id = robot_id
        total_path = []

        link_state = p.getLinkState(self.robot_id, self.ee_id)
        start_pos = link_state[4]  # world position
        
        pickup_cart = [-3.8, -1.6294126749038695, 0.5583768248558044]               
        dropoff_cart = [-4.300050067901611, -1.6294126749038695, 0.5583768248558044]

        base_pose = p.getLinkState(self.robot_id, 0)
        self.base_pose = base_pose[4]
        self.base_orientation = math.degrees(p.getEulerFromQuaternion(base_pose[5])[2])

        print(f"Base pose: {base_pose[5]}, base orientation: {self.base_orientation}")
	
        joint_limits, _ = getMotorJointLimits(self.robot_id)
        self.joint_limits = joint_limits[:7]


        # Path 1
        path1, cart1, samples_cart = self.rrt(start_pos, pickup_cart)

        draw_waypoint(cart1, color=[0,1,1])
        #draw_waypoint(samples_cart, color=[0,0,1])
        if path1 is not None:
            total_path.extend(path1)  

        # Path 2
        path2, cart2, samples_cart = self.rrt(pickup_cart, dropoff_cart)
        
        draw_waypoint(cart2, color=[1,1,0])
        #draw_waypoint(samples_cart, color=[1,0,1])
        if path2 is not None:
            total_path.extend(path2)   
                    
        return np.array(total_path)


        
        
    def rrt(self, start_cart, goal_cart):
        home_config, _, _= getJointStates(self.robot_id)    
    
        start_config = p.calculateInverseKinematics(self.robot_id,
                      self.ee_id,
                      start_cart,
                      solver=self.ikSolver)[:7]
        start_config = self.clamp_config(start_config)
                                   
        goal_config = p.calculateInverseKinematics(self.robot_id,
                      self.ee_id,
                      goal_cart,
                      solver=self.ikSolver)[:7]
        goal_config = self.clamp_config(goal_config)
        
        self.tree = [TreeNode(start_config, start_cart)]
        
        samples_cart = []
        
        for i in range(self.max_iterations):
            
            sample_config, sample_cart = self.gen_sample_config(start_config, goal_config)
            sample_config = self.clamp_config(sample_config)
            
            if sample_cart is not None:
                samples_cart.append(sample_cart)
            
            for idx in range(len(sample_config)):
                p.resetJointState(self.robot_id, idx+7, sample_config[idx])

            if self.has_contact_or_close(floor_id=0, margin=0.02):
                for idx in range(len(home_config)):
                    p.resetJointState(self.robot_id, idx, home_config[idx])
                continue

            nearest_node = self.get_nearest_node(sample_config)
            
            new_config = self.steer(nearest_node.config, sample_config)
            new_config = self.clamp_config(new_config)
            
            for idx in range(len(new_config)):
                p.resetJointState(self.robot_id, idx+7, new_config[idx])
            
            if self.has_contact_or_close(floor_id=0, margin=0.02):
                for idx in range(len(home_config)):
                    p.resetJointState(self.robot_id, idx, home_config[idx])
                continue
            
            link_state = p.getLinkState(self.robot_id, self.ee_id)
            new_cart = link_state[4]  # world position
        
            new_node = TreeNode(new_config, new_cart, parent=nearest_node)
            self.tree.append(new_node)
            
            if self.distance(new_cart, goal_cart) < self.goal_threshold:
                print(f"Goal reached in {i} iterations") 
                
                for idx in range(len(home_config)):
                    p.resetJointState(self.robot_id, idx, home_config[idx])
            
                sequence_config, sequence_cart = new_node.retrace(self.robot_id, self.ee_id, self.ikSolver)
                return sequence_config, sequence_cart, samples_cart

        for idx in range(len(home_config)):
            p.resetJointState(self.robot_id, idx, home_config[idx])

        print("RRT failed to find a path")        
        return None
        
        
    ###############
        
    def clamp_config(self, config):
        return config
        """Clamp a configuration to joint limits."""
    
        if self.joint_limits is None:
	        joint_limits, joint_names =  getMotorJointLimits(self.robot_id)
	        self.joint_limits = joint_limits[:7] 
    
        return [min(max(config[i], self.joint_limits[i][0]), self.joint_limits[i][1]) for i in range(len(self.joint_limits))]
        
    def gen_sample_config(self, start_config, goal_config):
        start = np.array(start_config)
        goal = np.array(goal_config)
        
        goal_std_deg=10.0
        goal_std = math.radians(goal_std_deg)
        
        if random.random() < self.goal_bias:
            sample_cart = None
            sample_config = np.random.normal(loc=goal, scale=goal_std, size=len(self.joint_limits))
        else:
            sample_cart = self.gen_sample_cartesian()
            sample_config = p.calculateInverseKinematics(self.robot_id,
                self.ee_id,
                sample_cart,
                solver=self.ikSolver)[:7]

        return sample_config, sample_cart
       
    def gen_sample_cartesian(self):

        bounds = np.array([
            [0.6, 0.8, 0.2],    # min xyz
            [ 0.6, 0.2, 2]       # max xyz
        ])

        return np.random.uniform(self.base_pose - bounds[0], self.base_pose + bounds[1])
    
    ###############
    
    def get_nearest_node(self, q):
        return min(self.tree, key=lambda node: self.distance(node.config, q))

    def distance(self, q1, q2):
        return np.linalg.norm(np.array(q1) - np.array(q2))
        
    def steer(self, q_from, q_to):
        q_to= np.array(q_to)
        q_from = np.array(q_from)

        direction = q_to - q_from
        dist = np.linalg.norm(direction)
        if dist < self.step_size:
            return q_to
        return q_from + (direction / dist) * self.step_size
        
    def has_contact_or_close(self, floor_id=0, margin=0.02):
        """
        Return True if robot is in contact with anything but the floor or
        within a safety margin (inflates obstacles).
        """
        p.performCollisionDetection()
        
        # Direct contacts first
        contacts = p.getContactPoints(bodyA=self.robot_id)
        for c in contacts:
            bodyB = c[2]
            if bodyB == floor_id:
                continue
            #print(f"Collision with {bodyB}")
            return True

        # End effector in base self collision       
        transpose_base = [0.0, 0.0, 0.25]
        base_radius = 0.45
        ee_radius = 0.1
        
        link_state = p.getLinkState(self.robot_id, self.ee_id)
        ee_cart = link_state[4]  # world position
        
        distance = self.distance(ee_cart, np.array(self.base_pose) + np.array(transpose_base))
        if distance <= ee_radius + base_radius:
            #print(f"Self collision")
            return True
        



        # Near contacts within margin
        other_bodies = [b for b in range(p.getNumBodies()) if b != self.robot_id and b != floor_id]
        for b in other_bodies:
            if p.getClosestPoints(bodyA=self.robot_id, bodyB=b, distance=margin):
                #print(f"Contacts within margin {bodyB}")
                return True
        return False
    
    

	
class TreeNode(object):

    def __init__(self, config, cart, parent=None):
        self.parent = parent
        self.config = config        
        self.cart = cart


    def retrace(self, robot_id, ee_id, ikSolver):
        sequence = []
        cart_sequence = []
        node = self
        while node is not None:
            sequence.append(np.array(node.config))
            cart_sequence.append(node.cart)
            node = node.parent
        return sequence[::-1], cart_sequence[::-1]


        
        
	
	

    
        
