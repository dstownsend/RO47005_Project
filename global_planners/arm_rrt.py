from .base import BaseGlobalPlanner

import numpy as np
import pybullet as p

import random
import math

from global_planners.arm_helpers import draw_line, draw_waypoint, getMotorJointStates, getMotorJointLimits, getJointStates



class ArmRRTPlanner(BaseGlobalPlanner):
    
    def __init__(self):

        
        self.ee_id = 15
        self.max_iterations = 10**8
        self.step_size= math.radians(5)
        self.goal_threshold = math.radians(20)
        self.ikSolver = 0
        self.goal_bias = 0.5
        self.robot_id = 0
        
    def plan(self, robot_id, visualise=False):
        self.robot_id = robot_id
        total_path = []

        link_state = p.getLinkState(self.robot_id, self.ee_id)
        pos = link_state[4]  # world position


        pickup_cart = [-3.8, -1.6294126749038695, 0.5583768248558044]
        pickup_config = p.calculateInverseKinematics(robot_id,
                          self.ee_id,
                          pickup_cart,
                          solver=self.ikSolver)[:7]
                          
        dropoff_cart = [-4.300050067901611, -1.6294126749038695, 0.5583768248558044]
        dropoff_config = p.calculateInverseKinematics(robot_id,
                          self.ee_id,
                          dropoff_cart,
                          solver=self.ikSolver)[:7]

        path1, cart1 = self.rrt(pos, pickup_cart)

        draw_waypoint(cart1, color=[0,1,1])
        
        if path1 is not None:
            total_path.extend(path1)  
        #return np.array(total_path)
        
        dropoff_cart = [-4.300050067901611, -1.6294126749038695, 0.5583768248558044]
        dropoff_config = p.calculateInverseKinematics(robot_id,
                          self.ee_id,
                          dropoff_cart,
                          solver=self.ikSolver)[:7]

        path2, cart2 = self.rrt(pickup_cart, dropoff_cart)
        
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
        
        for i in range(self.max_iterations):
            
            if random.random() < self.goal_bias:
                sample_cart = goal_cart
                sample_config = goal_config
            else:
                sample_cart = genRandomSample()
                sample_config = genRandomSampleConfig(self.robot_id)
            
            
            for idx in range(len(sample_config)):
                p.resetJointState(self.robot_id, idx+7, sample_config[idx])
            link_state = p.getLinkState(self.robot_id, self.ee_id)
            pos = link_state[4]
            #p.addUserDebugPoints(
             #   pointPositions=[pos],
              #  pointColorsRGB=[[0,0,1]],
               # pointSize=5,
             #   lifeTime=0
            #)
            
            #nearest_node = getNearestNode(tree, sample_cart)
            nearest_node = getNearestNode(tree, sample_config)
            
            #new_cart = steer(nearest_node.cart, sample_cart, self.step_size)
            new_config = steer(nearest_node.config, sample_config, self.step_size)

            #collision, new_config = self.checkCollision(self.robot_id, new_cart)            
            collision, new_cart = self.checkCollision(self.robot_id, new_config)
                        
            if collision:
                continue
                
                
            #p.addUserDebugPoints(
             #   pointPositions=[new_cart],
              #  pointColorsRGB=[[1,0,0]],
               # pointSize=5,
               # lifeTime=0
            #)

            
            
            new_node = TreeNode(new_config, new_cart, parent=nearest_node)
            tree.append(new_node)
            print(len(tree))
            
            if distance(new_config, goal_config) < self.goal_threshold:
                print(f"Goal reached in {i} iterations") 
                
                for idx in range(len(joint_positions)):
                    p.resetJointState(self.robot_id, idx, joint_positions[idx])
            
                return new_node.retrace(self.robot_id, self.ee_id, self.ikSolver)

        for idx in range(len(joint_positions)):
            p.resetJointState(self.robot_id, idx, joint_positions[idx])

        print("RRT failed to find a path")        
        return None

    def checkCollision(self, robot, config):
        
        joint_limits, joint_names =  getMotorJointLimits(self.robot_id)
        joint_limits = joint_limits[:7]
        
        #config = p.calculateInverseKinematics(self.robot_id,
         #                         self.ee_id,
          #                        cart,
           #                       solver=self.ikSolver)[:7]        

        joint_lim_tolerance_deg = 1
        
        for idx in range(len(config)):
            #if idx > 0 and (config[idx] < joint_limits[idx][0]+math.radians(joint_lim_tolerance_deg) or config[idx] > joint_limits[idx][1]-math.radians(joint_lim_tolerance_deg)):
                #print(f"Limits{idx}")
                #return True, None
            p.resetJointState(robot, idx+7, config[idx])
            
        link_state = p.getLinkState(self.robot_id, self.ee_id)
        pos = link_state[4]  # world position
            
        #if distance(cart, pos) > 0.01:     
         #   print("Distance")
          #  return True, None
            
        p.performCollisionDetection()
        contacts = p.getContactPoints(bodyA=robot)

                
        #if len(contacts) > 0:
        if collision_ignore_floor(self.robot_id, 0):
            print("Collision")
            return True, None
        
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

def genRandomSampleConfig(robot):


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


        
        
	
	

    
        
