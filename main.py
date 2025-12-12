import numpy as np
import logging
import time

import gymnasium as gym
from urdfenvs.robots.generic_urdf.generic_diff_drive_robot import GenericDiffDriveRobot
from urdfenvs.urdf_common.urdf_env import UrdfEnv

from global_planners import global_planner_rrt, arm_cubic #,dumb_global_planner
from controllers.arm_controller import ArmController
from global_planners.arm_helpers import getMotorJointStates

import math
import pybullet as p

# from local_planners import dumb_local_planner
from local_planners import mpc

from environment.scene_builder import apply_scenario_to_env, refresh_dynamic_obstacle_states
from environment.scenarios import get_scenario

N_STEPS = 10000
BASE_START = (4,4)
BASE_GOAL = (-4,-1)
# ARM_START =
# ARM_GOAL =
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def main():
    # 0. Setup environment
    env, robots, obstacles = create_env_with_obstacles(scenario_name="one_static") #empty, one_static, dynamic_and_static
    ob = env.reset(
        pos=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.5, 0.0, 1.8, 0.5])
    )[0]
    # env, robots, obstacles = create_env_with_obstacles(scenario_name="empty")
    history = []
    phase = "move_base"
    
# 1. Setup planners
    ########## Set variables for RRT_planners ################
    X_dimensions = np.array([(-4.5,4.5),(-4.5,4.5)]) # change once we know the actual dim of the final workspace
    # print(obstacles)
    obstacles_rrt = sphere_to_square(obstacles) # Possibly change type or dim here   
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
    global_path = RRT_planner.plan(rrt_type = 'rrt_star', x_init = BASE_START, x_goal = BASE_GOAL, prc = prc, plot_bool=True)
    logger.info(f"GLOBAL PATH IS: {global_path}")
    
    action = np.zeros(env.n())
    next_vertex = global_path.pop(0)
    goal_state = np.append(next_vertex, 0.0)
    
    arm_global_planner = arm_cubic.ArmCubicPlanner()
    arm_controller = ArmController()
    robot_id = env._robots[0]._robot 
    
    for _ in range(100):
        ob, *_ = env.step(np.zeros(11))
    joint_home_pose = [0.0, math.radians(-0), 0.0, math.radians(-160), 0.0, math.radians(160), math.radians(50)]

    for idx in range(len(joint_home_pose)):
        p.resetJointState(robot_id, idx+7, joint_home_pose[idx]);
        
    for _ in range(100):
        ob, *_ = env.step(np.zeros(11))
        
    mpos, mvel, mtorq, names = getMotorJointStates(robot_id) #returns length 13  
    desired_arm_joint_pos = mpos[:7]

    # Main loop
    for step in range(N_STEPS):
        if phase == "move_base":
            # 2b. Navigation local replan (dynamic obs)
            # TODO: once done, switch phase to move_arm
            current_state = ob["robot_0"]["joint_state"]["position"][:3]
            # goal_state = np.array([-4.0, 1.0, 0.0])
            cost = local_planner.ocp_solver.get_cost()
            if cost < 10:
                logger.warning(f"REACHED WAYPOINT {goal_state}")
                if not global_path:
                    logger.warning("REACHED FINAL GOAL")
                    phase = "move_arm"
                    continue
                next_vertex = global_path.pop(0)
                goal_state = np.append(next_vertex, 0.0)
            vehicle_control = local_planner.plan(current_state, goal_state, obstacles)
            logger.debug(f"vehicle control: {vehicle_control}")
            
            action = arm_controller.compute_vel_single(robot_id, desired_arm_joint_pos)
            
            action[:2] = vehicle_control

            # print(vehicle_control)
            # logger.info("in phase: move_base")
            # pass

        elif phase == "move_arm":
            action[:2] = np.array([0.0,0.0])
            logger.info("in phase: move_arm")
    
            # 3. Manipulation task
            if not arm_controller.goal_reached:
                if arm_controller.path is None:
                    arm_controller.path = arm_global_planner.plan(robot_id, None, visualise=True) 
                
                action = arm_controller.compute_vel_path(robot_id)
            else:
               action = np.zeros(env.n()) 

            
        
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
        scenario_name: str = "empty"):

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
    pos = np.array([4.0, 4.0, 0.0,
                    0.0, 0.7, 0.0, -1.0, 0.0, 1.0, 0.0,
                    0.02, 0.02], dtype=float)
    ob = env.reset(pos=pos)
        
    #TODO: create wall and static obs

    # Get Scenario Config
    scenario_cfg = {}
    if randomize:
        _, scenario_cfg = get_random_training_scenario()
    else:
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
    
if __name__ == "__main__":
    main()
