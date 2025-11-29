import matplotlib
matplotlib.use("TkAgg")   # Force an interactive backend

import numpy as np
import matplotlib.pyplot as plt

# pose = [x, y, theta] - numpy array
def global_planner(input_pose, final_pose, n_waypoints):
    """
    Super simple diff-drive-feasible planner:
    1) Rotate in place to face the goal
    2) Drive straight toward the goal
    3) Rotate in place to match final heading
    """
    x0, y0, th0 = input_pose
    x1, y1, th1 = final_pose

    # Direction from start to goal
    alpha = np.arctan2(y1 - y0, x1 - x0)

    # Split waypoints between: rotate1, straight, rotate2
    # (very simple split: 1/4, 1/2, 1/4)
    n_rot1 = max(2, n_waypoints // 4)
    n_trans = max(2, n_waypoints // 2)
    n_rot2 = max(1, n_waypoints - n_rot1 - n_trans)

    waypoints = []

    # 1) Rotate in place from th0 to alpha (x, y fixed) 
    # ths1 = np.linspace(th0, alpha, n_rot1)
    # for th in ths1:
    #     waypoints.append([x0, y0, th])

    # 2) Drive straight from (x0, y0) to (x1, y1) with heading alpha 
    s_vals = np.linspace(0.0, 1.0, n_trans)
    for s in s_vals:
        x = x0 + s * (x1 - x0)
        y = y0 + s * (y1 - y0)
        waypoints.append([x, y, alpha])

    # # 3) Rotate in place at goal to match th1
    # ths2 = np.linspace(alpha, th1, n_rot2)
    # for th in ths2:
    #     waypoints.append([x1, y1, th])

    waypoints = np.array(waypoints)

    return waypoints  # numpy array


def plot_waypoints(waypoints):
    xs = waypoints[:, 0]
    ys = waypoints[:, 1]
    thetas = waypoints[:, 2]

    plt.figure(figsize=(6, 6))
    plt.plot(xs, ys, 'o-', label="Path")

    plt.scatter(xs[0], ys[0], color='green', s=120, label="Start")
    plt.scatter(xs[-1], ys[-1], color='red', s=120, label="Goal")

    # Draw heading arrows
    for x, y, th in waypoints:
        dx = 0.2 * np.cos(th)
        dy = 0.2 * np.sin(th)
        plt.arrow(x, y, dx, dy, head_width=0.08, length_includes_head=True)

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Diff-Drive Feasible Path (Rotate–Straight–Rotate)")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.show()


# input_pose = np.array([0.0, 0.0, 0.0])
# final_pose = np.array([5.0, 3.0, 0.0])
# wps = global_planner(input_pose, final_pose, 20)

# plot_waypoints(wps)
