import warnings
import gymnasium as gym
import numpy as np
import pybullet

from urdfenvs.robots.generic_urdf.generic_diff_drive_robot import GenericDiffDriveRobot
from urdfenvs.urdf_common.urdf_env import UrdfEnv

from global_planner_plusplot_line import global_planner_spline, plot_waypoints
from local_planner import local_planner

def draw_waypoints_in_pybullet(waypoints, env,
                               z_height=0.05,
                               color=(1, 0, 0),
                               line_width=2.0):
    """
    Draw a polyline (series of connected waypoints) in PyBullet.

    waypoints : Nx2 or Nx3 array
    env       : UrdfEnv instance
    """
    cid = env._cid  # PyBullet client ID used internally by UrdfEnv

    for i in range(len(waypoints) - 1):
        x1, y1 = waypoints[i][0], waypoints[i][1]
        x2, y2 = waypoints[i + 1][0], waypoints[i + 1][1]

        start = [x1, y1, z_height]
        end = [x2, y2, z_height]

        pybullet.addUserDebugLine(
            start,
            end,
            lineColorRGB=color,
            lineWidth=line_width,
            lifeTime=0,  # Draw permanently
            physicsClientId=cid,
        )


def draw_global_frame_in_pybullet(env, axis_length=1.0, line_width=3.0):
    """
    Draw the global XYZ coordinate frame at the world origin in PyBullet.
    X: red, Y: green, Z: blue.
    """
    cid = env._cid

    # X axis (red)
    pybullet.addUserDebugLine(
        [0.0, 0.0, 0.0],
        [axis_length, 0.0, 0.0],
        lineColorRGB=[1.0, 0.0, 0.0],
        lineWidth=line_width,
        lifeTime=0,
        physicsClientId=cid,
    )

    # Y axis (green)
    pybullet.addUserDebugLine(
        [0.0, 0.0, 0.0],
        [0.0, axis_length, 0.0],
        lineColorRGB=[0.0, 1.0, 0.0],
        lineWidth=line_width,
        lifeTime=0,
        physicsClientId=cid,
    )

    # Z axis (blue)
    pybullet.addUserDebugLine(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, axis_length],
        lineColorRGB=[0.0, 0.0, 1.0],
        lineWidth=line_width,
        lifeTime=0,
        physicsClientId=cid,
    )


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

    # draw global XYZ frame in the sim
    draw_global_frame_in_pybullet(env)

    action = np.zeros(env.n())

    # Set nr waypoints and create waypoints 
    n_waypoints = 10
    waypoints = global_planner_spline(input_pose,final_pose,n_waypoints)
    plot_waypoints(waypoints)

    draw_waypoints_in_pybullet(
        waypoints,
        env,
        z_height=0.05,
        color=(1, 0, 0),   # Red path
        line_width=2.0
    )


    # Set initial conditiosn 
    n_it = 0 
    target_idx = 1
    run = True
    
    while run and n_it < n_steps_max:
        robot_id = env._robots[0]._robot  
        pos, orn = pybullet.getBasePositionAndOrientation(robot_id, physicsClientId=env._cid)
        heading_r = pybullet.getEulerFromQuaternion(orn)[2]
        
        x_r,y_r= pos[0],pos[1]
        state_r=np.array([x_r,y_r,heading_r])
        
        action[0],action[1],target_idx = local_planner(
            waypoints,
            target_idx,
            state_r,
            threshold_pos=0.1,
            threshold_head=0.1,
            K_p_lat=1,
            K_p_head=1
        )
        
        ob, *_ = env.step(action)
        n_it += 1 

    env.close()


if __name__ == "__main__":
    show_warnings = False
    warning_flag = "default" if show_warnings else "ignore"
    with warnings.catch_warnings():
        warnings.filterwarnings(warning_flag)
        run_albert(render=True,
                   input_pose = np.array([0.0, 0.0, 0.0]), # Starting position waypoints (start)
                   final_pose = np.array([2.0, -2.5, 0.0]) # Final position waypoints (goal)
                   )
