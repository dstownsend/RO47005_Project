import warnings
import gymnasium as gym
import numpy as np
from urdfenvs.robots.generic_urdf.generic_diff_drive_robot import GenericDiffDriveRobot
from urdfenvs.urdf_common.urdf_env import UrdfEnv
from multiprocessing import Process, Pipe
from urdfenvs.keyboard_input.keyboard_input_responder import Responder
from pynput.keyboard import Key

def run_albert(conn, n_steps=10000, render=True, goal=True, obstacles=True):
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
    action = np.zeros(env.n())
    # action[0] = 0.2
    # action[1] = 0.0
    # action[5] = -0.1
    ob = env.reset(
        pos=np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.5, 0.0, 1.8, 0.5])
    )
    print(f"Initial observation : {ob}")
    history = []
    for _ in range(n_steps):
        conn.send({"request_action": True, "kill_child": False})
        keyboard_data = conn.recv()
        action[0:2] = keyboard_data["action"]
        ob, *_ = env.step(action)
        print(f"robot pos: {ob['robot_0']['joint_state']['position'][:3]}")
        history.append(ob)
    env.close()
    conn.send({"request_action": False, "kill_child": True})
    return history


if __name__ == "__main__":
    # show_warnings = False
    # warning_flag = "default" if show_warnings else "ignore"
    # with warnings.catch_warnings():
    #     warnings.filterwarnings(warning_flag)
    #     run_albert(render=True)

    # setup multi threading with a pipe connection
    parent_conn, child_conn = Pipe()

    # create parent process
    p = Process(target=run_albert, args=(parent_conn,))

    # create Responder object
    responder = Responder(child_conn)

    # unlogical key bindings
    custom_on_press = {
        Key.left: np.array([-1.0, 0.0]),
        Key.space: np.array([1.0, 0.0]),
        Key.page_down: np.array([1.0, 1.0]),
        Key.page_up: np.array([-1.0, -1.0]),
        
    }

    responder.setup(default_action=np.array([0.0, 0.0]))
    # responder.setup(custom_on_press=custom_on_press)

    # start parent process
    p.start()

    # start child process which keeps responding/looping
    responder.start(p)

    # kill parent process
    p.kill()
