import numpy as np
import logging
import time

import gymnasium as gym
from urdfenvs.robots.generic_urdf.generic_diff_drive_robot import GenericDiffDriveRobot
from urdfenvs.urdf_common.urdf_env import UrdfEnv

from global_planners import global_planner_rrt, arm_cubic, arm_rrt #,dumb_global_planner
from controllers.arm_controller import ArmController
from global_planners.arm_helpers import getMotorJointStates, draw_waypoint, draw_line

import math
import pybullet as p

# from local_planners import dumb_local_planner
from local_planners import mpc

from environment.scene_builder import apply_scenario_to_env, refresh_dynamic_obstacle_states
from environment.scenarios import get_scenario

N_STEPS = 100000
BASE_START = (4,4)
BASE_GOAL = (-4,-1)
BASE_GOAL_ORIENTATION = 0 # degrees about global Z axis 
ARM_PICKUP = [-3.7, -1.6, 0.6]   
ARM_DROPOFF = [-4.3, -1.6, 0.6]
MANUAL_PATH = False  

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def main():
# 0. Setup environment
    env, robots, obstacles = create_env_with_obstacles(scenario_name="one_static") #empty, one_static, dynamic_and_static # random static
    ob, *_ = env.step(np.zeros(11))

    history = []
    phase = "move_base"
    
# 1. Setup planners
    ########## Set variables for RRT_planners ################
    X_dimensions = np.array([(-4, 4),(-4, 4)]) # change once we know the actual dim of the final workspace
    #X_dimensions = np.array([(-5,10),(-5,10)]) # change once we know the actual dim of the final workspace
        
    obstacles_rrt = sphere_to_square(obstacles) # Possibly change type or dim here
    obstacles_rrt = dilate_obstacles(obstacles_rrt,0.23365)  # 0.3365 m  is max halfwidth of albert base, check if all distances in env are in m
    # print(obstacles)
    q = 0.1
    r = 0.01
    max_samples = 3000
    rewire_count = 20
    
    prc = 0.1
    
    ########## Set class for RRT_planners ################
    RRT_planner = global_planner_rrt.RRT_planner(X_dimensions,obstacles_rrt,q,r,max_samples,rewire_count)
    
    local_planner = mpc.create_mpc_planner()
    
    # 2a. Navigation global plan 
    global_path = RRT_planner.plan(rrt_type = 'rrt_star_bidirectional_plus_heuristic', x_init = BASE_START, x_goal = BASE_GOAL, prc = prc, plot_bool=True)
    logger.info(f"GLOBAL PATH IS: {global_path}")
    
    global_path_waypoints =[]
    for point in global_path:
        point_3d = [point[0], point[1], 0.5]
        global_path_waypoints.append(point_3d)
    draw_line(global_path_waypoints)
    
    action = np.zeros(env.n())
    next_vertex = global_path.pop(0)
    goal_state = np.append(next_vertex, 0.0)
    
    ##### Arm #####
    # Initialise Planners
    arm_global_planner = arm_cubic.ArmCubicPlanner()
    arm_global_planner_rrt = arm_rrt.ArmRRTPlanner()
    
    # Initialise Controllers
    arm_controller = ArmController()
    
    # Configure arm and base collision links       
    link_transformation = np.identity(4)
    link_transformation[0:3, 3] = np.array([0.0, -0.0, 0.25])
    env.add_collision_link(
        robot_index=0,
        link_index=0,
        shape_type="sphere",
        size=[0.45],
        link_transformation=link_transformation,
    )
    
    link_transformation = np.identity(4)
    env.add_collision_link(
        robot_index=0,
        link_index=15,
        shape_type="sphere",
        size=[0.1],
        link_transformation=link_transformation,
    )
    
    robot_id = env._robots[0]._robot 
          
    # Generate arm path       
    if MANUAL_PATH:
        arm_controller.path = arm_global_planner.plan(robot_id, None, visualise=True) 
    else:
        # Move robot to the base goal location
        pos = np.array([-8.0, -5.0, math.radians(-90),
                        0.0, math.radians(0), 0.0, math.radians(-160), 0.0, math.radians(160), math.radians(50),
                        0.02, 0.02], dtype=float)
        ob = env.reset(pos=pos)
        for _ in range(100):
            ob, *_ = env.step(np.zeros(11))
        # Create RRT plan
        arm_controller.path = arm_global_planner_rrt.plan(robot_id, visualise=True)

    # Move robot to the base start goal locaition
    pos = np.array([8.0, 5.0, math.radians(90),
                    0.0, math.radians(0), 0.0, math.radians(-160), 0.0, math.radians(160), math.radians(50),
                    0.02, 0.02], dtype=float)
    ob = env.reset(pos=pos)
    for _ in range(100):
        ob, *_ = env.step(np.zeros(11))
        
    # Get arm pose to hold during base movement
    mpos, mvel, mtorq, names = getMotorJointStates(robot_id) #returns length 13  
    desired_arm_joint_pos = mpos[:7]
        
    # Main loop
    for step in range(N_STEPS):
        if phase == "move_base":
            # 2b. Navigation local replan (dynamic obs)
            
            current_state = ob["robot_0"]["joint_state"]["position"][:3]

            cost = local_planner.ocp_solver.get_cost()
            if cost < 10:
                logger.warning(f"REACHED WAYPOINT {goal_state}")
                
                if not global_path:
                    logger.warning("REACHED FINAL GOAL")
                    phase = "rotate_base"
                    continue
                    
                next_vertex = global_path.pop(0)
                goal_state = np.append(next_vertex, 0.0)
                
            vehicle_control = local_planner.plan(current_state, goal_state, obstacles)
            logger.debug(f"vehicle control: {vehicle_control}")
            
            action = arm_controller.compute_vel_single(robot_id, desired_arm_joint_pos)
            action[:2] = vehicle_control
                        
        elif phase == "rotate_base":
        
            base_pose = p.getLinkState(robot_id, 0)
            base_location = base_pose[4][:2]
            print(base_location)
            
            base_orientation = p.getEulerFromQuaternion(base_pose[5])[2]
            error_rot = math.radians(BASE_GOAL_ORIENTATION) - base_orientation
        
            error_loc_Y = np.array(BASE_GOAL[1]) - base_location[1]

            K_rot = 0.5
            K_loc = 0.5
            rotate_vel = K_rot * error_rot
            location_Y_vel = K_loc * error_loc_Y           
            
            action = arm_controller.compute_vel_single(robot_id, desired_arm_joint_pos)
            action[1] = rotate_vel
            
            if abs(error_rot) < math.radians(2):
                action[0] = location_Y_vel
                if abs(error_loc_Y) < 0.05:
                    phase = "move_arm"
        

        elif phase == "move_arm":
            action[:2] = np.array([0.0,0.0])
            #logger.info("in phase: move_arm")
    
            # 3. Manipulation task
            if not arm_controller.goal_reached:               
                action = arm_controller.compute_vel_path(robot_id)
                #action = np.zeros(env.n()) 
            else:
               action = np.zeros(env.n()) 
               
        
        # Fix gripper finger joint to avoid API bug
        p.resetJointState(robot_id, 16, 0.01);
        p.resetJointState(robot_id, 17, 0.01);     
        
        # Sync dynamic obstacle state for planners
        obstacles = refresh_dynamic_obstacle_states(env, obstacles)

        if obstacles["dynamic"]:
            dyn_pos = obstacles["dynamic"][0].get("position")
            # print(f"Dynamic obstacle 0 position @t={env.t():.2f}: {dyn_pos}")

        # Simulation step
        #action = np.zeros(env.n())
        #action[0] = 0.2 # drive forward
        ob, reward, terminated, truncated, info  = env.step(action)
        logger.debug(f"robot pos: {ob['robot_0']['joint_state']['position'][:3]}")
        if terminated:
            logger.info(info)
            break
        history.append(ob)

    env.close()

def create_env_with_obstacles(
        randomize=False,
        scenario_name: str = "empty",
        middle_wall_height = "low"):

    robots = [
        GenericDiffDriveRobot(
            urdf="albert.urdf",
            mode="vel",
            actuated_wheels=["wheel_right_joint", "wheel_left_joint"],
            castor_wheels=["rotacastor_right_joint", "rotacastor_left_joint"],
            wheel_radius = 0.08,
            wheel_distance = 0.494,
            spawn_rotation = 1.570796,
            facing_direction = 'x',
        ),
    ]

    env: UrdfEnv = UrdfEnv(
        dt=0.01, robots=robots, render=True
    )

    # [x, y, yaw, j1, j2, j3, j4, j5, j6, j7, finger1, finger2]
    pos = np.array([4.0, 4.0, math.radians(90),
                    0.0, math.radians(0), 0.0, math.radians(-160), 0.0, math.radians(160), math.radians(50),
                    0.02, 0.02], dtype=float)
    ob = env.reset(pos=pos)
        		
    #TODO: create wall and static obs

    # Get Scenario Config
    scenario_cfg = {}
    scenario_cfg = get_scenario(scenario_name)
    
    # Spawn in walls and obstacles returns obstacle dictionary (walls, static, dynamic)
    obs = apply_scenario_to_env(env, scenario_cfg)

    # Camera perspectives
    env.reconfigure_camera(8.0, 180.0, -90.01, (0, 0, 0)) # Birds Eye
    # env.reconfigure_camera(2.0, -50.0, -50.01, (4, 4, 0)) # Spawn Config Joints
    # env.reconfigure_camera(8.0, 0.0, -90.01, (0, 0, 0)) # Birds Eye
    # env.reconfigure_camera(2.0, -50.0, -50.01, (-4, -1, 0)) # Spawn Config Joints
    # env.reconfigure_camera(2.0, -0.0, 0.0, (4, 4, 1)) # Spawn Config Side

    return env, robots, obs
    
def sphere_to_square(obstacles):
    obs = obstacles['static']
    sq_obs = []
    for el in obs:
        x_min = el['position'][0]-1.3*el['radius']
        y_min = el['position'][1]-1.3*el['radius']
        x_max = el['position'][0]+1.3*el['radius']
        y_max = el['position'][1]+1.3*el['radius']
        sq_obs.append((x_min,y_min,x_max,y_max))
    return np.array(sq_obs) if len(sq_obs)>0 else None
    


def dilate_obstacles(obstacles, dilation):
    def add_dilation(tupl, dilation):
        return (tupl[0] - dilation, tupl[1] - dilation, tupl[2] + dilation, tupl[3] + dilation)
    
    wall_obstacles = np.array([
        (-5.05, -5.0, -4.95, 10.0),   
        (9.95, -5.0, 10.05, 10.0),   
        (-5.0, -5.05, 10.0, -4.95),   
        (-5.0, 9.95, 10.0, 10.05),    
    ])
    
    hub_wall_obstacles = np.array([
        (-3.01, -5.0, -2.99, -2),
        (-4, -3.00, -3.01, -1.99),
    ])
    all_obstacles = np.concatenate((obstacles, wall_obstacles, hub_wall_obstacles), axis=0)
    dilated_obstacles = [add_dilation(el, dilation) for el in all_obstacles]
    return np.array(dilated_obstacles)

if __name__ == "__main__":
    main()
