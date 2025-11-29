import warnings
import gymnasium as gym
import numpy as np
import pybullet

from urdfenvs.robots.generic_urdf.generic_diff_drive_robot import GenericDiffDriveRobot
from urdfenvs.urdf_common.urdf_env import UrdfEnv

from global_planner_plusplot import global_planner
from local_planner import local_planner

def run_albert(n_steps_max=10000, render=False, goal=True, obstacles=True,input_pose = None, final_pose=None):
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
        dt=0.01, robots=robots, render=render
    )

    ob = env.reset(
        pos=np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.5, 0.0, 1.8, 0.5])
    )
    

    action = np.zeros(env.n())

    n_waypoints = 10
    waypoints = global_planner(input_pose,final_pose,n_waypoints)

    n_it = 0 
    eta = 0.1 
    run = True

    while run and n_it < n_steps_max:
        robot_id = env._robots[0]._robot  
        pos, orn = pybullet.getBasePositionAndOrientation(robot_id, physicsClientId=env._cid)
        x,y,theta = pos[0],pos[1],orn[2]
        
        state=np.array([x,y,0])
        goal_x = 0 
        goal_y = -2.5
        action[0] = 0.5

        if ((x-goal_x)**2+(y-goal_y)**2)**0.5 < 0.1:
            action[0] = 0
        
        #action[0],action[1] = local_planner(waypoints,state,threshold=0.1)


        ob, *_ = env.step(action)
        n_it += 1 

    action[0],action[1] = 0,0

    env.close()



if __name__ == "__main__":
    show_warnings = False
    warning_flag = "default" if show_warnings else "ignore"
    with warnings.catch_warnings():
        warnings.filterwarnings(warning_flag)
        run_albert(render=True,input_pose = np.array([0.0, 0.0, 0.0]),final_pose = np.array([0.0, -2.5, 0.0]))