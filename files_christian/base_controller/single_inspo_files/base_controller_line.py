
import math
import numpy as np
import warnings
import gymnasium as gym
import numpy as np
import pybullet

from urdfenvs.robots.generic_urdf.generic_diff_drive_robot import GenericDiffDriveRobot
from urdfenvs.urdf_common.urdf_env import UrdfEnv

from global_planner_plusplot_line import global_planner, plot_waypoints
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
    waypoints = global_planner(input_pose,final_pose,n_waypoints)
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

def error_calc(wp_target, wp_previous, state, threshold_pos):
    x1, y1, heading1 = wp_previous
    x2, y2, heading2 = wp_target
    x_robot, y_robot, heading_robot = state

    # account for the fact that we get spawed with +x dir in -y dir global
    heading_robot -= np.pi/2

    print()
    print(f'xyhead robot   {x_robot, y_robot, heading_robot}')
    print(f'xyhead previous{x1, y1, heading1}')
    print(f'target xyhead  {x2, y2, heading2}')

    # Segment vector
    dx = x2 - x1
    dy = y2 - y1
    denom = math.hypot(dx, dy)

    # Lateral error (cross-track)
    if denom > 1e-6:
        e_lateral = ((x_robot - x1) * dy - (y_robot - y1) * dx) / denom
    else:
        # Pure rotation segment (same x,y) -> no lateral error
        e_lateral = 0.0

    # Heading error (target - robot), then wrap to [-pi, pi]
    e_heading = heading2 - heading_robot
    e_heading = (e_heading + math.pi) % (2 * math.pi) - math.pi

    print(f'lateral error: {e_lateral}')
    print(f'heading error: {e_heading}')
    return e_lateral, e_heading


def local_planner(reference_trajectory,
                  target_idx,
                  state,
                  threshold_pos=0.2,
                  threshold_head=0.3,
                  K_p_lat=1.0,
                  K_p_head=1.0):
    """
    Returns:
        u : linear velocity (body frame)
        w : angular velocity (yaw rate)
        target_idx : possibly updated waypoint index
    """

    n = len(reference_trajectory)

    # Safety clamp on index (you start at 1, using 0 as "previous")
    if target_idx <= 0:
        target_idx = 1
    if target_idx >= n:
        target_idx = n - 1

    # Current target and previous waypoint
    wp_target = reference_trajectory[target_idx]
    wp_previous = reference_trajectory[target_idx - 1]

    # Errors
    e_lateral, e_heading = error_calc(wp_target, wp_previous, state, threshold_pos)

    x_robot, y_robot, heading_robot = state
    x2, y2, heading2 = wp_target

    # Distance to target waypoint
    dist_to_target = math.hypot(x_robot - x2, y_robot - y2)

    # Segment length (to detect rotate-only segments)
    dx_seg = x2 - wp_previous[0]
    dy_seg = y2 - wp_previous[1]
    seg_len = math.hypot(dx_seg, dy_seg)

    # --------------------------
    # Waypoint advancement logic
    # --------------------------
    if seg_len < 1e-6:
        # Pure rotation waypoint: advance when heading is good
        if abs(e_heading) < threshold_head and target_idx < n - 1:
            target_idx += 1
            wp_target = reference_trajectory[target_idx]
            wp_previous = reference_trajectory[target_idx - 1]
            e_lateral, e_heading = error_calc(wp_target, wp_previous, state, threshold_pos)
            x2, y2, heading2 = wp_target
            dist_to_target = math.hypot(x_robot - x2, y_robot - y2)
            dx_seg = x2 - wp_previous[0]
            dy_seg = y2 - wp_previous[1]
            seg_len = math.hypot(dx_seg, dy_seg)
    else:
        # Translation waypoint: advance based on distance to target
        if dist_to_target < threshold_pos and target_idx < n - 1:
            target_idx += 1
            wp_target = reference_trajectory[target_idx]
            wp_previous = reference_trajectory[target_idx - 1]
            e_lateral, e_heading = error_calc(wp_target, wp_previous, state, threshold_pos)
            x2, y2, heading2 = wp_target
            dist_to_target = math.hypot(x_robot - x2, y_robot - y2)
            dx_seg = x2 - wp_previous[0]
            dy_seg = y2 - wp_previous[1]
            seg_len = math.hypot(dx_seg, dy_seg)

    # --------------------------
    # Control law
    # --------------------------

    # 1) Pure rotation segments: only fix heading
    if seg_len < 1e-6:
        u = 0.0
        w = K_p_head * e_heading
        return u, w, target_idx

    # 2) On straight segments:
    #    if heading error is large, rotate in place first
    big_heading = math.radians(45)  # e.g. > 45° = too misaligned
    if abs(e_heading) > big_heading:
        u = 0.0
        w = K_p_head * e_heading
        return u, w, target_idx

    # 3) Normal tracking: combine lateral + heading
    w_lateral = K_p_lat * e_lateral   # minus sign to reduce cross-track
    w_heading = K_p_head * e_heading
    w = w_lateral + w_heading

    # Forward speed when heading error is reasonable
    u = 0.5
    if abs(e_heading) > threshold_head:
        u = 0.0

    # if abs(e_lateral-e_heading) < 0.1:
    #     e_lateral = 0 # set this one to zero to let the sim break the singularity

    return u, w, target_idx

if __name__ == "__main__":
    show_warnings = False
    warning_flag = "default" if show_warnings else "ignore"
    with warnings.catch_warnings():
        warnings.filterwarnings(warning_flag)
        run_albert(render=True,                             # set to false if you only want the plot
                   input_pose = np.array([0.0, 0.0, 0.0]), # Starting position waypoints (start)
                   final_pose = np.array([2.0, -2.5, 0.0]) # Final position waypoints (goal)
                   )

