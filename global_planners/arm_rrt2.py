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
        self.max_iterations = 10000
        self.step_size= math.radians(5)
        self.goal_threshold = math.radians(5)
        self.ikSolver = 0
        self.goal_bias = 0.5
        self.robot_id = 0
        
    def plan(self, robot_id, visualise=False):
        self.robot_id = robot_id
        total_path = []

        link_state = p.getLinkState(self.robot_id, self.ee_id)
        pos = link_state[4]  # world position


        pickup_cart = [-3.8, -1.6294126749038695, 0.7]
        pickup_config = p.calculateInverseKinematics(robot_id,
                          self.ee_id,
                          pickup_cart,
                          solver=self.ikSolver)[:7]
                          
        dropoff_cart = [-4.300050067901611, -1.6294126749038695, 0.7]
        dropoff_config = p.calculateInverseKinematics(robot_id,
                          self.ee_id,
                          dropoff_cart,
                          solver=self.ikSolver)[:7]

        #path1, cart1 = self.rrt(pos, pickup_cart)
        #path1, cart1, sample_points, new_points = self.rrt(pos, pickup_cart)
        path2, cart2, sample_points, new_points = self.rrt(pickup_cart, dropoff_cart)
        
        draw_waypoint(cart2, color=[0,1,1])
        draw_waypoint(sample_points, color=[0,0,1])
        draw_waypoint(new_points, color=[1,0,0])
        
        
        if path2 is not None:
            total_path.extend(path2)  
        #return np.array(total_path)
        


        
        if path2 is not None:
            total_path.extend(path2)   
            
        draw_waypoint(cart2, color=[1,1,0])

        return np.array(total_path)


        
        
    def rrt(self, start_cart, goal_cart):
    
        joint_positions, _, _= getJointStates(self.robot_id)
        
        start_config = p.calculateInverseKinematics(self.robot_id,
                      self.ee_id,
                      start_cart,
                      solver=self.ikSolver)[:7]
                      
        goal_config = p.calculateInverseKinematics(self.robot_id,
                      self.ee_id,
                      goal_cart,
                      solver=self.ikSolver)[:7]
        
        tree = [TreeNode(start_config, start_cart)]
        
        sample_points =[]
        new_points = []

        for i in range(self.max_iterations):
            
            if random.random() < self.goal_bias:
                sample_config = goal_config
            else:
                sample_config = genRandomSampleConfig(self.robot_id, start_config, goal_config)
                        
            
            for idx in range(len(sample_config)):
                p.resetJointState(self.robot_id, idx+7, sample_config[idx])
            link_state = p.getLinkState(self.robot_id, self.ee_id)
            pos = link_state[4]
            sample_points.append(pos)      
         
            nearest_node = getNearestNode(tree, sample_config)
           
            new_config = steer(nearest_node.config, sample_config, self.step_size)


            if edge_in_collision(self.robot_id, nearest_node.config, new_config):
                continue

            collision, new_cart = self.checkCollision(self.robot_id, new_config)
                        
            if collision:
                continue
            new_points.append(new_cart)            
            
            new_node = TreeNode(new_config, new_cart, parent=nearest_node)
            tree.append(new_node)
            print(len(tree))
            
            if distance(new_config, goal_config) < self.goal_threshold:
                print(f"Goal reached in {i} iterations") 
                
                for idx in range(len(joint_positions)):
                    p.resetJointState(self.robot_id, idx, joint_positions[idx])
            
                sequence_config, sequence_cart = new_node.retrace(self.robot_id, self.ee_id, self.ikSolver)
                return sequence_config, sequence_cart, sample_points, new_points

        for idx in range(len(joint_positions)):
            p.resetJointState(self.robot_id, idx, joint_positions[idx])

        print("RRT failed to find a path")        
        return None, [goal_cart,start_cart], sample_points, new_points

    def checkCollision(self, robot, config):
                
        for idx in range(len(config)):
            p.resetJointState(robot, idx+7, config[idx])
            
        link_state = p.getLinkState(self.robot_id, self.ee_id)
        pos = link_state[4]  # world position
                
        if collision_ignore_floor(self.robot_id, 0):
            print("Collision")
            return True, pos
        
        return False, pos


def collision_ignore_floor(robot_id, floor_id):
    p.performCollisionDetection()
    contacts = p.getContactPoints(bodyA=robot_id)
    for c in contacts:
        bodyA, bodyB = c[1], c[2]
        if bodyB == floor_id:
            continue  # ignore floor contact
        print(f"collision with {bodyB}")
        return True  # collision with something else
    return False
    
    
### RRT Functions ###


# Gen sample 
def genRandomSample():
    minX = -3 #3
    maxX = -5
    
    minY = -1 #1
    maxY = -4 #4
    
    minZ = 0
    maxZ = 2.5

    sample = [random.uniform(minX,maxX), random.uniform(minY,maxY), random.uniform(minZ,maxZ)]
    #sample = [-4,-3.5,0.0]
    return sample

def genRandomSampleConfig2(robot):


    joint_limits, joint_names = getMotorJointLimits(robot)
    joint_limits = joint_limits[:7]
    
    #sample = [random.uniform(joint[0], joint[1]) for joint in joint_limits]

    step = math.radians(5.0)
    sample =[]
    
    for idx in range(len(joint_limits)):
        n = int(round((joint_limits[idx][1] - joint_limits[idx][0]) / step))
        value = joint_limits[idx][0] + step * float(random.randint(0, n))
        sample.append(value)

    return sample
    
def genRandomSampleConfig(robot, start_config, goal_config):
    start = np.array(start_config)
    goal = np.array(goal_config)
    
    goal_std_deg=5.0
    goal_std = math.radians(goal_std_deg)
    goal_focus=0.5
    
    joint_limits, joint_names = getMotorJointLimits(robot)
    joint_limits = joint_limits[:7]
    step = math.radians(5.0)

    if random.random() < goal_focus:
        sample = np.random.normal(loc=goal, scale=goal_std, size=len(joint_limits))
    else:
        sample =[]  
        print(joint_limits)
        for idx in range(len(joint_limits)):
            n = int(round((joint_limits[idx][1] - joint_limits[idx][0]) / step))
            value = joint_limits[idx][0] + step * float(random.randint(0, n))
            sample.append(value)
            
    return sample
    
# Calc distance

def getNearestNode(tree, q):
    #return min(tree, key=lambda node: distance(node.cart, q))
    return min(tree, key=lambda node: distance(node.config, q))

def distance(q1, q2):
    return np.linalg.norm(np.array(q1) - np.array(q2))


def steer(q_from, q_to, step_size):
    q_to= np.array(q_to)
    q_from = np.array(q_from)
    
    direction = q_to - q_from
    dist = np.linalg.norm(direction)
    if dist < step_size:
        return q_to
    return q_from + (direction / dist) * step_size

def clamp_config(config, joint_limits):
    """Clamp a configuration to joint limits."""
    return [min(max(config[i], joint_limits[i][0]), joint_limits[i][1]) for i in range(len(joint_limits))]

def edge_in_collision(robot_id, q_from, q_to, steps=10, joint_limits=None, floor_id=0, margin=0.02):
    """Check interpolated edge for collisions (ignores floor contact) with optional safety margin."""
    if joint_limits is None:
	    joint_limits, joint_names = getMotorJointLimits(robot_id)
	    joint_limits = joint_limits[:7]

    saved = [p.getJointState(robot_id, j+7)[0] for j in range(len(q_from))]
    try:
        q_from = np.array(q_from)
        q_to = np.array(q_to)
        for i in range(steps + 1):
            alpha = i / steps
            q = (1 - alpha) * q_from + alpha * q_to
            q = clamp_config(q, joint_limits)
            for j, val in enumerate(q):
                p.resetJointState(robot_id, j+7, val)
            p.performCollisionDetection()
            if has_contact_or_close(robot_id, floor_id=floor_id, margin=margin):
                return True
        return False
    finally:
        for j, q in enumerate(saved):
            p.resetJointState(robot_id, j+7, q)


	
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


        
        
	
	

    
        
