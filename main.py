import numpy as np
import logging

import gymnasium as gym
from urdfenvs.robots.generic_urdf.generic_diff_drive_robot import GenericDiffDriveRobot
from urdfenvs.urdf_common.urdf_env import UrdfEnv

from global_planners import rrtstar #,dumb_global_planner
# from local_planners import dumb_local_planner
from local_planners import mpc

from environment.scene_builder import apply_scenario_to_env, refresh_dynamic_obstacle_states
from environment.scenarios import get_scenario, get_random_training_scenario

N_STEPS = 1000
BASE_START = (0,0)
BASE_GOAL = (5,5)
# ARM_START =
# ARM_GOAL =
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def main():
    # 0. Setup environment
    env, robots, obstacles = create_env_with_obstacles(scenario_name="dynamic_cylinders")
    # env, robots, obstacles = create_env_with_obstacles(scenario_name="empty")
    history = []
    phase = "move_base"
    
    # 1. Setup planners
    global_planner = rrtstar.RRTStar()
    local_planner = mpc.create_mpc_planner()
    logger.info("Global path: ")
    
    # 2a. Navigation global plan
    # global_path = global_planner.plan(BASE_START, BASE_GOAL, obstacles)
    
    # Main loop
    for step in range(N_STEPS):
        if phase == "move_base":
            # 2b. Navigation local replan (dynamic obs)
            # TODO: get control from local planner, fill action
            # TODO: once done, switch phase to move_arm
            current_state = np.array([0.0, 0.0, 0.0])
            goal_state = np.array([1.0, 1.0, 0.0])
            vehicle_control = local_planner.plan(current_state, goal_state, None)
            # print(vehicle_control)
            # action[:2] = vehicle_control
            # logger.info("in phase: move_base")
            # pass

        elif phase == "move_arm":            
            # 3. Manipulation task
            # TODO: get control and fill action
            # joint_velocities = some_arm_planner.plan()
            # action[2:9] = joint_velocities
            pass
        
        # Sync dynamic obstacle state for planners
        obstacles = refresh_dynamic_obstacle_states(env, obstacles)

        if obstacles["dynamic"]:
            dyn_pos = obstacles["dynamic"][0].get("position")
            # print(f"Dynamic obstacle 0 position @t={env.t():.2f}: {dyn_pos}")


        # Simulation step
        action = np.zeros(env.n())
        #action[0] = 0.2 # drive forward
        ob, reward, terminated, truncated, info  = env.step(action)
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
            spawn_rotation = 0,
            facing_direction = '-y',
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
    env.reconfigure_camera(8.0, 0.0, -90.01, (0, 0, 0)) # Birds Eye
    # env.reconfigure_camera(2.0, -50.0, -50.01, (4, 4, 0)) # Spawn Config Joints
    # env.reconfigure_camera(2.0, -0.0, 0.0, (4, 4, 1)) # Spawn Config Side

    return env, robots, obs
    
    
if __name__ == "__main__":
    main()
