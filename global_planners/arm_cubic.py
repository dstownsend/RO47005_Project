from .base import BaseGlobalPlanner

import numpy as np
import pybullet as p
import time

from datetime import datetime, timezone

from global_planners.arm_helpers import draw_line, draw_waypoint


class ArmCubicPlanner(BaseGlobalPlanner):
    
    def __init__(self):

        
        self.ee_id = 15
        self.interp_steps = 50


    def cubic_interpolate(self, q0, q1, n_steps):
        q0 = np.asarray(q0, dtype=float)
        q1 = np.asarray(q1, dtype=float)

        result = []
        for i in range(n_steps):
            t = i / (n_steps - 1)
            h = 3 * t**2 - 2 * t**3  # cubic smoothstep
            qi = (1 - h) * q0 + h * q1
            result.append(qi)

        return result
        
    def quintic_interpolate(self, q0, q1, n_steps):
        q0 = np.asarray(q0, dtype=float)
        q1 = np.asarray(q1, dtype=float)

        result = []
        for i in range(n_steps):
            t = i / (n_steps - 1)
            h = 6*t**5 - 15*t**4 + 10*t**3  # quintic smoothstep
            qi = (1 - h) * q0 + h * q1
            result.append(qi)

        return result

    def plan_joint_path(self, waypoints_cart, robot_id):
        ikSolver = 0
        joint_poses = []
        joint_path_segments = []
        
        for i in range(len(waypoints_cart)):

            joint_pose = p.calculateInverseKinematics(robot_id,
                          self.ee_id,
                          waypoints_cart[i],
                          solver=ikSolver)[:7]
                          
            joint_pose = np.array(joint_pose)
            joint_poses.append(joint_pose)

        for i in range(len(joint_poses)-1):
            joint_path_segment = self.quintic_interpolate(joint_poses[i], joint_poses[i+1], self.interp_steps)
            
            joint_path_segment = np.array(joint_path_segment)
            joint_path_segments.append(joint_path_segment)

        joint_path = np.vstack(joint_path_segments)
        
        return joint_path
    
    
    def plan(self, robot_id, pickup_cart, dropoff_cart, visualise=False):
        start = time.perf_counter()

        current_pose_cart = list(p.getLinkState(robot_id, self.ee_id)[4])
        
                
        p1_cart = list(current_pose_cart)
        p1_cart[1] = pickup_cart[1]   
        p1_cart[2] += 0.1   
        
        p2_cart = list(p1_cart)
        p2_cart[0] = pickup_cart[0]

        p3_cart = list(pickup_cart)
       
        p4_cart = list(p3_cart)
        p4_cart[2] += 0.5  
        
        p5_cart = list(p4_cart)
        p5_cart[0] = dropoff_cart[0]

        p6_cart = list(dropoff_cart)
                
        waypoints = []
        waypoints.append(current_pose_cart)
        waypoints.append(p1_cart)        
        waypoints.append(p2_cart)
        waypoints.append(p3_cart)
        waypoints.append(p4_cart)
        waypoints.append(p5_cart)
        waypoints.append(p6_cart)
        
        if visualise:
            draw_line(waypoints)
            draw_waypoint(waypoints)    
            
            
        joint_path = self.plan_joint_path(waypoints, robot_id)

        end = time.perf_counter()
        elapsed_ms = (end - start) * 1000

        utc_ms = int(time.time() * 1000)
        result = f"{utc_ms},Arm Cubic, Duration [ms],{elapsed_ms:.3f}\n"
        with open("ArmResults.txt", "a", encoding="utf-8") as file:
            file.write(result)


        return joint_path

	
	


        
        
	
	

    
        
