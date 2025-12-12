import os
import pybullet
import numpy as np
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.urdf_common.urdf_env import UrdfEnv

from environment.scene_builder import (
    build_room_walls,
    build_static_cylinder,
    build_moving_sphere,
    build_moving_cylinder,
)
from environment.scenarios import get_scenario, get_random_training_scenario


def apply_scenario_to_env(env: UrdfEnv, scenario_cfg: dict):
    # 1) Walls (always the same)
    for wall in build_room_walls(wall_length=10.0):
        env.add_obstacle(wall)

    # 2) Static obstacles
    for i, s_cfg in enumerate(scenario_cfg["static"]):
        cyl = build_static_cylinder(
            name=f"static_{i}",
            position=s_cfg["position"],
            radius=s_cfg.get("radius", 0.5),
        )
        env.add_obstacle(cyl) # Put this outside of the function, rather return

    # 3) Dynamic obstacles
    for j, d_cfg in enumerate(scenario_cfg["dynamic"]):
        dynamic_type = d_cfg.get("type", "sphere")
        if dynamic_type == "cylinder":
            dyn = build_moving_cylinder(
                name=f"dynamic_{j}",
                trajectory_exprs=d_cfg["trajectory"],
                radius=d_cfg.get("radius", 0.5),
                height=d_cfg.get("height", 2.0),
            )
        else:
            dyn = build_moving_sphere(
                name=f"dynamic_{j}",
                trajectory_exprs=d_cfg["trajectory"],
                radius=d_cfg.get("radius", 0.5),
            )
        env.add_obstacle(dyn)


def run_point_robot(
    n_steps=1000,
    render=False,
    scenario_name: str | None = None,
    random_training: bool = False,
    record_video: bool = False,
    recordings_dir: str = "recordings",
    recording_name: str | None = None,
    max_time: float | None = None,
):
    robots = [GenericUrdfReacher(urdf="pointRobot.urdf", mode="vel")]
    env = UrdfEnv(dt=0.01, robots=robots, render=render)
    video_logging_id = None

    if record_video and render:
        os.makedirs(recordings_dir, exist_ok=True)
        if recording_name is None:
            recording_name = f"{scenario_name or 'scenario'}"
        video_path = os.path.join(recordings_dir, f"{recording_name}.mp4")
        video_logging_id = pybullet.startStateLogging(
            pybullet.STATE_LOGGING_VIDEO_MP4, video_path
        )

    # Choose scenario
    if random_training:
        chosen_name, scenario_cfg = get_random_training_scenario()
        print(f"Using random training scenario: {chosen_name}")
    else:
        if scenario_name is None:
            scenario_name = "empty"  # default
        scenario_cfg = get_scenario(scenario_name)
        print(f"Using scenario: {scenario_name}")

    # Build environment
    apply_scenario_to_env(env, scenario_cfg)

    # Optional goal:
    # from urdfenvs.scene_examples.goal import splineGoal
    # env.add_goal(splineGoal)

    action = np.array([0.1, 0.0, 0.0])
    pos0 = np.array([3.0, 4.0, 0.0])
    vel0 = np.array([0.0, 0.0, 0.0])
    ob = env.reset(pos=pos0, vel=vel0)
    print(f"Initial observation: {ob}")

    env.reconfigure_camera(8.0, 0.0, -90.01, (0, 0, 0))
    history = []
    steps_limit = n_steps
    if max_time is not None:
        steps_limit = min(steps_limit, int(max_time / env.dt))

    for _ in range(steps_limit):
        ob, _, terminated, _, info = env.step(action)
        if terminated:
            print(info)
            break
        history.append(ob)

    if record_video and render and video_logging_id is not None:
        pybullet.stopStateLogging(video_logging_id)

    env.close()
    return history


if __name__ == "__main__":
    recordings_dir = os.path.join(os.getcwd(), "PDM", "project", "recordings")
    # a) no obstacles
    # run_point_robot(render=True, scenario_name="empty")

    # b) basic with only one static obstacle
    run_point_robot(render=True, scenario_name="only_static")

    # c) basic with only dynamic obstacle
    # run_point_robot(render=True, scenario_name="only_dynamic")

    # d) basic with one static & one dynamic
    # run_point_robot(
    #     render=True,
    #     recordings_dir=recordings_dir,
    #     record_video=False,
    #     scenario_name="static_and_dynamic",
    #     max_time=6)

    # e) pick 1 of the 5 training scenarios at random
    # run_point_robot(render=True, random_training=True)
