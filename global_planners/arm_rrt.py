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
        self.goal_threshold = math.radians(5)
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

        path1, cart1, samples_cart = self.rrt(pos, pickup_cart)

        draw_waypoint(cart1, color=[0,1,1])
        draw_waypoint(samples_cart, color=[0,0,1])
        if path1 is not None:
            total_path.extend(path1)  
        #sreturn np.array(total_path)
        
        dropoff_cart = [-4.300050067901611, -1.6294126749038695, 0.5583768248558044]
        dropoff_config = p.calculateInverseKinematics(robot_id,
                          self.ee_id,
                          dropoff_cart,
                          solver=self.ikSolver)[:7]

        path2, cart2, samples_cart = self.rrt(pickup_cart, dropoff_cart)
        
        if path2 is not None:
            total_path.extend(path2)   
            
        draw_waypoint(cart2, color=[1,1,0])
        draw_waypoint(samples_cart, color=[1,0,1])
        
        return np.array(total_path)


        
        
    def rrt(self, start_cart, goal_cart):
    
        base_pose = p.getLinkState(self.robot_id, 0)
        base_pose = base_pose[4]
    
        joint_positions, _, _= getJointStates(self.robot_id)
        
        joint_limits, _ =  getMotorJointLimits(self.robot_id)
        joint_limits = joint_limits[:7]
        
        start_config = p.calculateInverseKinematics(self.robot_id,
                      self.ee_id,
                      start_cart,
                      solver=self.ikSolver)[:7]
        #start_config = clamp_config(start_config, joint_limits)
                      
        goal_config = p.calculateInverseKinematics(self.robot_id,
                      self.ee_id,
                      goal_cart,
                      solver=self.ikSolver)[:7]
        #goal_config = clamp_config(goal_config, joint_limits)
        
        tree = [TreeNode(start_config, start_cart)]
        
        samples_cart =[]
        
        for i in range(self.max_iterations):
            
            if random.random() < self.goal_bias:
                sample_cart = goal_cart
                sample_config = goal_config
            else:
                sample_cart = genRandomSample(start_cart, goal_cart)
                sample_config, sample_cart = genRandomSampleConfig(self.robot_id, self.ee_id, start_config, goal_config)
            
            #sample_config = clamp_config(sample_config, joint_limits)
            samples_cart.append(sample_cart)
            
            for idx in range(len(sample_config)):
                p.resetJointState(self.robot_id, idx+7, sample_config[idx])
            link_state = p.getLinkState(self.robot_id, self.ee_id)
            pos = link_state[4]
            
            #if pos[1] > base_pose[1]:
            #	continue
            
 #           p.addUserDebugPoints(
  #              pointPositions=[pos],
   #             pointColorsRGB=[[0,0,1]],
    #            pointSize=5,
     #           lifeTime=0
      #      )
            
            #nearest_node = getNearestNode(tree, sample_cart)
            nearest_node = getNearestNode(tree, sample_config)
            
            #new_cart = steer(nearest_node.cart, sample_cart, self.step_size)
            new_config = steer(nearest_node.config, sample_config, self.step_size)
            #new_config = clamp_config(new_config, joint_limits)

            #if edge_in_collision(self.robot_id, nearest_node.config, new_config, joint_limits=joint_limits):
            #    continue

            #collision, new_config = self.checkCollision(self.robot_id, new_cart)
            collision, new_cart = self.checkCollision(self.robot_id, new_config, joint_limits)
                        
            if has_contact_or_close(self.robot_id, floor_id=0, margin=0.02):
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
            
                sequence_config, sequence_cart = new_node.retrace(self.robot_id, self.ee_id, self.ikSolver)
                return sequence_config, sequence_cart, samples_cart

        for idx in range(len(joint_positions)):
            p.resetJointState(self.robot_id, idx, joint_positions[idx])

        print("RRT failed to find a path")        
        return None

    def checkCollision(self, robot, config, joint_limits=None, margin=0.02):
        
        if joint_limits is None:
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
            clamped = min(max(config[idx], joint_limits[idx][0]), joint_limits[idx][1])
            p.resetJointState(robot, idx+7, clamped)
        
        if is_self_collision(robot, config):
            return True, None
            
        link_state = p.getLinkState(self.robot_id, self.ee_id)
        pos = link_state[4]  # world position
            
        #if distance(cart, pos) > 0.01:     
         #   print("Distance")
          #  return True, None
            
        p.performCollisionDetection()
        if collision_ignore_floor(self.robot_id, 0, margin=margin):
            print("Collision")
            return True, None
        
        return False, pos


def is_self_collision(robot_id, config, floor_id=0, margin=0.02):
    """Return True if config is in self-collision (ignoring floor contact) with optional safety margin."""
    # Save current joint positions for the arm joints we overwrite
    saved = [p.getJointState(robot_id, j+7)[0] for j in range(len(config))]
    try:
        for j, q in enumerate(config):
            p.resetJointState(robot_id, j+7, q)
        p.performCollisionDetection()
        if has_contact_or_close(robot_id, floor_id=floor_id, margin=margin):
            return True
        return False
    finally:
        for j, q in enumerate(saved):
            p.resetJointState(robot_id, j+7, q)


def edge_in_collision(robot_id, q_from, q_to, steps=10, joint_limits=None, floor_id=0, margin=0.02):
    """Check interpolated edge for collisions (ignores floor contact) with optional safety margin."""
    if joint_limits is None:
        joint_limits = getMotorJointLimits(robot_id)[0][:7]

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


def collision_ignore_floor(robot_id, floor_id, margin=0.02):
    p.performCollisionDetection()
    return has_contact_or_close(robot_id, floor_id=floor_id, margin=margin)


def has_contact_or_close(robot_id, floor_id=0, margin=0.02):
    """
    Return True if robot is in contact with anything but the floor or
    within a safety margin (inflates obstacles).
    """
    # Direct contacts first
    contacts = p.getContactPoints(bodyA=robot_id)
    for c in contacts:
        bodyB = c[2]
        if bodyB == floor_id:
            continue
        return True

    # Near contacts within margin
    other_bodies = [b for b in range(p.getNumBodies()) if b != robot_id and b != floor_id]
    for b in other_bodies:
        if p.getClosestPoints(bodyA=robot_id, bodyB=b, distance=margin):
            return True
    return False
    
    
### RRT Functions ###


# Gen sample 
def genRandomSample(start_cart, goal_cart, padding=0.25, goal_std=0.05, goal_focus=0.8):
    """
    Sample cartesian points near the corridor between start and goal.
    - With probability goal_focus, draw from a Gaussian around goal_cart.
    - Otherwise, draw uniformly from a bounding box that encloses start/goal with padding.
    """
    start = np.array(start_cart)
    goal = np.array(goal_cart)

    # Bounding box around start/goal with some margin
    lower = np.minimum(start, goal) - padding
    upper = np.maximum(start, goal) + padding

    if random.random() < goal_focus:
        sample = np.random.normal(loc=goal, scale=goal_std, size=3)
    else:
        sample = np.array([random.uniform(lower[i], upper[i]) for i in range(3)])

    # Clamp to bounds to avoid drifting away
    sample = np.minimum(np.maximum(sample, lower), upper)
    return sample.tolist()

def genRandomSampleConfig2(robot, start_config, goal_config, padding_deg=20.0, goal_std_deg=5.0, goal_focus=0.5):
    

    joint_limits, joint_names = getMotorJointLimits(robot)
    joint_limits = joint_limits[:7]
    
    # Build a box around start/goal in joint space with padding
    start = np.array(start_config)
    goal = np.array(goal_config)
    padding = math.radians(padding_deg)
    lower = np.maximum(np.minimum(start, goal) - padding, [lim[0] for lim in joint_limits])
    upper = np.minimum(np.maximum(start, goal) + padding, [lim[1] for lim in joint_limits])

    if random.random() < goal_focus:
        std = math.radians(goal_std_deg)
        sample = np.random.normal(loc=goal, scale=std, size=len(joint_limits))
    else:
        sample = np.array([random.uniform(lower[i], upper[i]) for i in range(len(joint_limits))])

    # Clamp to joint limits and the padded box
    sample = np.minimum(np.maximum(sample, lower), upper)

    return sample.tolist()


def genRandomSampleConfig3(robot, start_config, goal_config):
    start = np.array(start_config)
    goal = np.array(goal_config)
    
    goal_std_deg=10.0
    goal_std = math.radians(goal_std_deg)
    goal_focus=0.2
    
    joint_limits, joint_names = getMotorJointLimits(robot)
    joint_limits = joint_limits[:7]
    step = math.radians(5.0)

    if random.random() < goal_focus:
        sample = np.random.normal(loc=goal, scale=goal_std, size=len(joint_limits))
    else:
        sample =[]  

        for idx in range(len(joint_limits)):
            n = int(round((joint_limits[idx][1] - joint_limits[idx][0]) / step))
            value = joint_limits[idx][0] + step * float(random.randint(0, n))
            sample.append(value)
            
    return sample


def genRandomSampleConfig(robot_id, ee_link, start_config, goal_config):
    start = np.array(start_config)
    goal = np.array(goal_config)
    
    goal_std_deg=10.0
    goal_std = math.radians(goal_std_deg)
    goal_focus=0.2
    
    joint_limits, joint_names = getMotorJointLimits(robot_id)
    joint_limits = joint_limits[:7]
    step = math.radians(5.0)

    #if random.random() < goal_focus:
     #   sample = np.random.normal(loc=goal, scale=goal_std, size=len(joint_limits))
    #else:

    sample_cart = sample_cartesian(robot_id)
    sample = ik_with_random_seed(robot_id, ee_link, sample_cart)


    return sample, sample_cart



def sample_cartesian(robot_id):
    base_pose = p.getLinkState(robot_id, 0)
    base_pose = base_pose[4]


    bounds = np.array([
        [0.6, 0.6, 0.2],  # min xyz
        [ 0.6,  0, 2]   # max xyz
    ])

    return np.random.uniform(base_pose - bounds[0], base_pose + bounds[1])

def get_arm_joints(robot_id):
    joint_indices = []
    lower_limits = []
    upper_limits = []
    joint_ranges = []

    for j in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, j)
        if info[2] == p.JOINT_REVOLUTE:
            joint_indices.append(j)
            ll, ul = info[8], info[9]
            lower_limits.append(ll)
            upper_limits.append(ul)
            joint_ranges.append(ul - ll)

    return (
        joint_indices,
        np.array(lower_limits),
        np.array(upper_limits),
        np.array(joint_ranges),
    )

def ik_with_random_seed(robot_id, ee_link, pos):
    
    joint_indices, ll, ul, jr = get_arm_joints(robot_id)

    seed = [np.random.uniform(-np.pi, np.pi) for _ in range(7)]
    return p.calculateInverseKinematics(
        robot_id,
        ee_link,
        pos
    )[:7]





def clamp_config(config, joint_limits):
    """Clamp a configuration to joint limits."""
    return [min(max(config[i], joint_limits[i][0]), joint_limits[i][1]) for i in range(len(joint_limits))]
    
    
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


        
        
	
	

    
        
