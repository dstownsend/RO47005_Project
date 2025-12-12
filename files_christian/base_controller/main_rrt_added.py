
import numpy as np
import logging

import gymnasium as gym
from urdfenvs.robots.generic_urdf.generic_diff_drive_robot import GenericDiffDriveRobot
from urdfenvs.urdf_common.urdf_env import UrdfEnv


from global_planner_rrt import RRT_planner
# from local_planners import dumb_local_planner


N_STEPS = 1000
BASE_START = (0,0)
BASE_GOAL = (5,5)
# ARM_START =
# ARM_GOAL =
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)





def main():
    # 0. Setup environment
    env, robots, obstacles = create_env_with_obstacles()
    history = []
    phase = "move_base"
    
    # 1. Setup planners
    ########## Set variables for RRT_planners ################
    X_dimensions = np.array([(0,100),(0,100)]) # change once we know the actual dim of the final workspace
    obstacles = obstacles # Possibly change type or dim here   
    q = 8
    r = 1
    max_samples = 1024
    rewire_count = 32
    
    prc = 0.1
    
    ########## Set class for RRT_planners ################
    RRT_planner = RRT_planner(X_dimensions,obstacles,q,r,max_samples,rewire_count)
    
    # local_planner = mpc.MPC()
    logger.info("Global path: ")
    
    # 2a. Navigation global plan 
    global_path = RRT_planner.plan(rrt_type = 'basic_rrt', x_init = BASE_START, x_goal = BASE_GOAL, prc = prc)
    
    # Main loop
    for step in range(N_STEPS):
        if phase == "move_base":
            # 2b. Navigation local replan (dynamic obs)
            # TODO: get control from local planner, fill action
            # TODO: once done, switch phase to move_arm
            # vehicle_control = local_planner.plan()
            # action[:2] = vehicle_control
            # logger.info("in phase: move_base")
            pass

        elif phase == "move_arm":            
            # 3. Manipulation task
            # TODO: get control and fill action
            # joint_velocities = some_arm_planner.plan()
            # action[2:9] = joint_velocities
            pass
        
        # Simulation step
        action = np.zeros(env.n())
        action[0] = 0.2 # drive forward
        ob, reward, terminated, truncated, info  = env.step(action)
        if terminated:
            logger.info(info)
            break
        history.append(ob)
        
    env.close()

def create_env_with_obstacles(dynamic_obs=False, randomize=False):
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

    ob = env.reset(
        pos=np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.5, 0.0, 1.8, 0.5])
    )
    
    #TODO: create wall and static obs
    from urdfenvs.scene_examples.obstacles import sphereObst1
    obstacles = [sphereObst1]
    for obstacle in obstacles: 
        env.add_obstacle(obstacle)
    
    if dynamic_obs:
        #TODO: create dynamic obs
        pass
    
    if randomize:
        # TODO: some logic to give random env 
        pass
    
    return env, robots, obstacles
    
    
if __name__ == "__main__":
    main()