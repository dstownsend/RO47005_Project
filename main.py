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

from controllers.simple_base_controller import calculate_base_vel

# from local_planners import dumb_local_planner
from local_planners import mpc

from environment.scene_builder import apply_scenario_to_env, refresh_dynamic_obstacle_states
from environment.scenarios import get_scenario

N_STEPS = 100000
BASE_START = (4, 4)
BASE_GOAL = (-4,-1)
BASE_GOAL_ORIENTATION = 0 # degrees about global Z axis 
BASE_CONTROLLER_MPC = True # True for MPC, False for PI
RANDOM_SEED = 14 # None (randomize), or int. Set to get repeatable env.

BASE_POSE_FOR_ARM = [-4,-2.6]
ARM_PICKUP = [-3.7, -3.1, 0.6]   
ARM_DROPOFF = [-4.3, -3.1, 0.6]
ARM_PATH_MANUAL = False
MID_WALL_HEIGHT = "middle" # low, middle, high

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def main():
# 0. Setup environment
    env, robots, obstacles = create_env_with_obstacles(scenario_name="random_static") #empty, one_static, dynamic_and_static # random_static
    ob, *_ = env.step(np.zeros(11))

    history = []
    phase = "move_base"
    
# 1. Setup planners
    ########## Set variables for RRT_planners ################
    X_dimensions = np.array([(-4, 4),(-4, 4)]) # change once we know the actual dim of the final workspace
    #X_dimensions = np.array([(-5,10),(-5,10)]) # change once we know the actual dim of the final workspace
        
    obstacles_rrt = sphere_to_square(obstacles) # Possibly change type or dim here
    obstacles_rrt = dilate_obstacles(obstacles_rrt,0.4)  # 0.3365 m  is max halfwidth of albert base, check if all distances in env are in m #0.23365
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
    global_path = RRT_planner.plan(rrt_type = 'rrt_star_bidirectional_plus_heuristic', x_init = BASE_START, x_goal = BASE_GOAL, prc = prc, plot_bool=False)
    logger.info(f"GLOBAL PATH IS: {global_path}")
       
    global_path_waypoints =[]
    for point in global_path:
        point_3d = [point[0], point[1], 0.5]
        global_path_waypoints.append(point_3d)
    draw_line(global_path_waypoints)
    
    draw_line([(BASE_GOAL[0],BASE_GOAL[1],0.5),(BASE_POSE_FOR_ARM[0],BASE_POSE_FOR_ARM[1],0.5)], color = [0,0,1])
    
        
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

    draw_waypoint([ARM_PICKUP, ARM_DROPOFF], color=[0,0,1], pointSize = 5)
    
    
    robot_id = env._robots[0]._robot 
                  
    # Get arm pose to hold during base movement
    mpos, mvel, mtorq, names = getMotorJointStates(robot_id) #returns length 13  
    desired_arm_joint_pos = mpos[:7]
        
    # Main loop
    start = time.time()
    for step in range(N_STEPS):
        if phase == "move_base":
            # 2b. Navigation local replan (dynamic obs)
            
            if BASE_CONTROLLER_MPC:
                current_state = ob["robot_0"]["joint_state"]["position"][:3]
                cost = local_planner.ocp_solver.get_cost()
                
                if cost < 10:
                    logger.warning(f"REACHED WAYPOINT {goal_state}")
                    
                    if not global_path:
                        logger.warning("REACHED RRT TRAVEL FINAL GOALs")
                        phase = "rotate_base"
                        time_taken = time.time() - start
                        with open("results.txt", "a") as f:
                            f.write(f"MPC, seed {RANDOM_SEED}: {time_taken:.4f} seconds\n")
                        logger.info(f"MPC, seed {RANDOM_SEED}: {time_taken:.4f} seconds")

                        continue
                        
                    next_vertex = global_path.pop(0)
                    goal_state = np.append(next_vertex, 0.0)
                    
                vehicle_control = local_planner.plan(current_state, goal_state, obstacles)
                logger.debug(f"vehicle control: {vehicle_control}")
                
                action = np.zeros(env.n())
                action[:2] = vehicle_control
            else:
                action = np.zeros(env.n())
                action[:2] = calculate_base_vel(goal_state, robot_id)
                
                base_pose = p.getLinkState(robot_id, 0)
                base_location = base_pose[4]
                
                if np.linalg.norm(goal_state - base_location) < 0.15:
                    logger.warning(f"REACHED WAYPOINT {goal_state}")
                    
                    if not global_path:
                        logger.warning("REACHED RRT TRAVEL FINAL GOAL")
                        phase = "rotate_base"
                        time_taken = time.time() - start
                        with open("results.txt", "a") as f:
                            f.write(f"PI, seed {RANDOM_SEED}: {time_taken:.4f} seconds\n")
                        logger.info(f"PI, seed {RANDOM_SEED}: {time_taken:.4f} seconds")
                        continue
                    
                    next_vertex = global_path.pop(0)
                    goal_state = np.append(next_vertex, 0.0)
                
            # Fix arm joint poisitions to avoid API bug
            for joint_index in range(len(desired_arm_joint_pos)):
                 p.resetJointState(robot_id, joint_index+7, desired_arm_joint_pos[joint_index]);
                        
        elif phase == "rotate_base":
            
            base_pose = p.getLinkState(robot_id, 0)
            base_location = base_pose[4][:2]
            #print(base_location)
            
            base_orientation = p.getEulerFromQuaternion(base_pose[5])[2]
        
            error_rot = (math.radians(BASE_GOAL_ORIENTATION) - base_orientation + math.pi) % (2 * math.pi) - math.pi
            error_loc = BASE_POSE_FOR_ARM[1] - base_location[1]

            K_rot = 0.5
            K_loc = 0.5
            rotate_vel = K_rot * error_rot
            location_vel = -K_loc * error_loc
            
            action = np.zeros(env.n())
            action[1] = rotate_vel
                    
            if abs(error_rot) < math.radians(0.5):
                action[0] = location_vel
                
                if abs(error_loc) < 0.05:
                    phase = "move_arm"
                    logger.warning("REACHED BASE POSE FOR ARM")
                    
                    
            # Fix arm joint poisitions to avoid API bug
            for joint_index in range(len(desired_arm_joint_pos)):
                 p.resetJointState(robot_id, joint_index+7, desired_arm_joint_pos[joint_index]);
        
        elif phase == "move_arm":
    
            # 3. Manipulation task
            if not arm_controller.goal_reached:
                if arm_controller.path is None:
                    if ARM_PATH_MANUAL:
                        arm_controller.path = arm_global_planner.plan(robot_id,  ARM_PICKUP, ARM_DROPOFF, visualise=True) 
                    else:
                        arm_controller.path = arm_global_planner_rrt.plan(robot_id, ARM_PICKUP, ARM_DROPOFF, visualise=True)
                        # Reset arm configuration 
                        for joint_index in range(len(desired_arm_joint_pos)):
                             p.resetJointState(robot_id, joint_index+7, desired_arm_joint_pos[joint_index]);
                        
                action = arm_controller.compute_vel_path(robot_id)
                
                K_loc = 0.5
                base_pose = p.getLinkState(robot_id, 0)
                base_location = base_pose[4]
                p.resetBasePositionAndOrientation(
                    robot_id,
                    base_location,
                    [0,0,0,1]
                )
                            
            else:
                action = np.zeros(env.n())
                phase = "complete"
                logger.warning("SEQUENCE COMPLETE")
                
               
        
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
    pos = np.array([BASE_START[0], BASE_START[1], math.radians(90),
                    0.0, math.radians(0), 0.0, math.radians(-160), 0.0, math.radians(160), math.radians(50),
                    0.02, 0.02], dtype=float)
    ob = env.reset(pos=pos)
        		
    #TODO: create wall and static obs

    # Get Scenario Config
    scenario_cfg = {}
    scenario_cfg = get_scenario(scenario_name)
    
    # Spawn in walls and obstacles returns obstacle dictionary (walls, static, dynamic)
    if MID_WALL_HEIGHT == "high":
        obs = apply_scenario_to_env(env, scenario_cfg, mid_wall_height = "high", seed=RANDOM_SEED)
    elif MID_WALL_HEIGHT == "middle":
        obs = apply_scenario_to_env(env, scenario_cfg, mid_wall_height = "middle", seed=RANDOM_SEED)
    else:
        obs = apply_scenario_to_env(env, scenario_cfg, mid_wall_height = "low", seed=RANDOM_SEED)

    # Camera perspectives
    #env.reconfigure_camera(8.0, 180.0, -90.01, (0, 0, 0)) # Birds Eye
    # env.reconfigure_camera(2.0, -50.0, -50.01, (4, 4, 0)) # Spawn Config Joints
    env.reconfigure_camera(8.0, 0.0, -90.01, (0, 0, 0)) # Birds Eye
    # env.reconfigure_camera(2.0, -50.0, -50.01, (-4, -1, 0)) # Spawn Config Joints
    # env.reconfigure_camera(2.0, -0.0, 0.0, (4, 4, 1)) # Spawn Config Side

    spawn_boxes()


    return env, robots, obs
    
def sphere_to_square(obstacles):
    obs = obstacles['static']
    sq_obs = []
    for el in obs:
        x_min = el['position'][0]-1*el['radius']
        y_min = el['position'][1]-1*el['radius']
        x_max = el['position'][0]+1*el['radius']
        y_max = el['position'][1]+1*el['radius']
        sq_obs.append((x_min,y_min,x_max,y_max))
    return np.array(sq_obs) if len(sq_obs)>0 else None
    

def spawn_boxes():
    # Desired position and orientation
    position_1 = [-3.5, -3.5, 0.25]  # x, y, z (z should be half height if sitting on ground)
    position_2 = [-4.5, -3.5, 0.25]  # x, y, z (z should be half height if sitting on ground)
    orientation = p.getQuaternionFromEuler([0, 0, 0])

    # Load your open-top box URDF
    box_id_1 = p.loadURDF(
        "environment/open_top_box.urdf",
        basePosition=position_1,
        baseOrientation=orientation,
        useFixedBase=True   # set False if you want it to fall
    )
    
    box_id_2 = p.loadURDF(
        "environment/open_top_box.urdf",
        basePosition=position_2,
        baseOrientation=orientation,
        useFixedBase=True   # set False if you want it to fall
    )


def dilate_obstacles(obstacles, dilation):
    def add_dilation(tupl, dilation):
        return (tupl[0] - dilation, tupl[1] - dilation, tupl[2] + dilation, tupl[3] + dilation)
    
    wall_obstacles = np.array([
        (-5.1, -5.1, -5.0, 5.0),   
        (5.0, -5.0, 5.1, 5.0),   
        (-5.0, -5.1, 5.0, -5.0),   
        (-5.0, 5.0, 5.0, 5.1),    
    ])
    
    hub_wall_obstacles = np.array([
        (-5.0, -5.0, -3, -2),
    ])
    
    if obstacles is None:
        all_obstacles = np.concatenate((wall_obstacles, hub_wall_obstacles), axis=0)
    else:
        all_obstacles = np.concatenate((obstacles, wall_obstacles, hub_wall_obstacles), axis=0)
    dilated_obstacles = [add_dilation(el, dilation) for el in all_obstacles]
    return np.array(dilated_obstacles)

if __name__ == "__main__":
    main()
