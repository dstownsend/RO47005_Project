import math

import sympy as sp
from mpscenes.obstacles.box_obstacle import BoxObstacle
from mpscenes.obstacles.cylinder_obstacle import CylinderObstacle
from mpscenes.obstacles.dynamic_sphere_obstacle import DynamicSphereObstacle
from mpscenes.obstacles.dynamic_cylinder_obstacle import DynamicCylinderObstacle
from mpscenes.obstacles.dynamic_obstacle import DynamicObstacle
from urdfenvs.urdf_common.urdf_env import UrdfEnv


def _linear_trajectory_exprs(
    start,
    end,
    speed: float,
    start_time: float = 0.0,
    stop_at_end: bool = True,
    heading: float | None = None,
):
    """
    Build piecewise-linear trajectory expressions using Heaviside clamps.

    Args:
        start: start position.
        end: end position.
        speed: linear speed along heading (units/s).
        start_time: time at which motion begins.
        stop_at_end: clamp motion after reaching end.
        heading: if provided (rad), overrides direction in xy-plane.
    Returns:
        (exprs, meta): exprs list usable by sympy; meta contains start, velocity,
        heading unit vector, duration and end_time.
    """
    start = list(start)
    end = list(end)
    dims = max(len(start), len(end))
    start += [0.0] * (dims - len(start))
    end += [0.0] * (dims - len(end))

    delta = [e - s for s, e in zip(start, end)]
    distance = math.sqrt(sum(d * d for d in delta))

    if heading is None and distance > 1e-9:
        direction = [d / distance for d in delta]
    elif heading is not None:
        direction = [math.cos(heading), math.sin(heading)] + [0.0] * (dims - 2)
    else:
        direction = [0.0] * dims

    duration = distance / speed if speed > 1e-9 else 0.0
    end_time = start_time + duration

    exprs = []
    for s_val, dir_comp in zip(start, direction):
        v_comp = dir_comp * speed
        expr = f"{s_val}"
        if abs(v_comp) > 1e-12:
            expr += (
                f" + ({v_comp})*(t - {start_time})*sp.Heaviside(t - {start_time})"
            )
            if stop_at_end and duration > 0.0:
                expr += (
                    f" - ({v_comp})*(t - {end_time})*sp.Heaviside(t - {end_time})"
                )
        exprs.append(expr)

    velocity_vec = [d * speed for d in direction]
    heading_norm = math.sqrt(sum(comp * comp for comp in velocity_vec))
    heading_vec = (
        [comp / heading_norm for comp in velocity_vec]
        if heading_norm > 1e-9
        else [0.0] * dims
    )

    meta = {
        "start": start,
        "velocity": velocity_vec,
        "heading": heading_vec,
        "duration": duration,
        "end_time": end_time,
        "start_time": start_time,
    }
    return exprs, meta


def build_room_walls(
    wall_length: float = 10.0,
    wall_height: float = 2.0,
    thickness: float = 0.1,
    z_center: float = 1.0,
    name_prefix: str = "wall",
    wall_rgba: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0),
):
    walls = []

    # +x wall
    walls.append(
        BoxObstacle(
            name=f"{name_prefix}_+x",
            content_dict={
                "type": "box",
                "geometry": {
                    "position": [wall_length / 2.0, 0.0, z_center],
                    "width": wall_length,
                    "height": wall_height,
                    "length": thickness,
                },
                "low": {"position": [wall_length / 2.0, 0.0, z_center]},
                "high": {"position": [wall_length / 2.0, 0.0, z_center]},
                "rgba": list(wall_rgba),
            },
        )
    )

    # +y wall
    walls.append(
        BoxObstacle(
            name=f"{name_prefix}_+y",
            content_dict={
                "type": "box",
                "geometry": {
                    "position": [0.0, wall_length / 2.0, z_center],
                    "width": thickness,
                    "height": wall_height,
                    "length": wall_length,
                },
                "low": {"position": [0.0, wall_length / 2.0, z_center]},
                "high": {"position": [0.0, wall_length / 2.0, z_center]},
                "rgba": list(wall_rgba),
            },
        )
    )

    # -y wall
    walls.append(
        BoxObstacle(
            name=f"{name_prefix}_-y",
            content_dict={
                "type": "box",
                "geometry": {
                    "position": [0.0, -wall_length / 2.0, z_center],
                    "width": thickness,
                    "height": wall_height,
                    "length": wall_length,
                },
                "low": {"position": [0.0, -wall_length / 2.0, z_center]},
                "high": {"position": [0.0, -wall_length / 2.0, z_center]},
                "rgba": list(wall_rgba),
            },
        )
    )

    # -x wall
    walls.append(
        BoxObstacle(
            name=f"{name_prefix}_-x",
            content_dict={
                "type": "box",
                "geometry": {
                    "position": [-wall_length / 2.0, 0.0, z_center],
                    "width": wall_length,
                    "height": wall_height,
                    "length": thickness,
                },
                "low": {"position": [-wall_length / 2.0, 0.0, z_center]},
                "high": {"position": [-wall_length / 2.0, 0.0, z_center]},
                "rgba": list(wall_rgba),
            },
        )
    )
    
    # +x hub wall
    walls.append(
        BoxObstacle(
            name=f"hub_wall_+x",
            content_dict={
                "type": "box",
                "geometry": {
                    "position": [-3.0, -3.5, 1],
                    "width": 3,
                    "height": 2,
                    "length": 0.1,
                },
                "low": {"position": [4.0, -3.5, 1.0]},
                "high": {"position": [4.0, -3.5, 1.0]},
                "rgba": list(wall_rgba),
            },
        )
    )

    # +y hub wall obstacle for arm RRT planner
    walls.append(
        BoxObstacle(
            name=f"hub_wall_+y",
            content_dict={
                "type": "box",
                "geometry": {
                    # "position": [4.0, 2.0, 2.0], # position for testing height directly at spawn
                    "position": [-4.0, -2.0, 1.5], # position at hub
                    "width": 0.1,
                    "height": 1.0,
                    "length": 2.0,
                },
                "low": {"position": [-4.0, -2.0, 1.5]},
                "high": {"position": [-4.0, -2.0, 1.5]},
                "rgba": list(wall_rgba),
            },
        )
    )

    return walls


def build_static_cylinder(name: str, position, radius=0.5, height=2.0, rgba=(0.1, 0.3, 0.3, 1.0)):
    content_dict = {
        "type": "cylinder",
        "movable": False,
        "geometry": {
            "position": list(position),
            "radius": radius,
            "height": height,
        },
        "rgba": list(rgba),
    }
    return CylinderObstacle(name=name, content_dict=content_dict)


def build_moving_sphere(
    name: str,
    trajectory_exprs,
    radius: float = 0.5,
    rgba=(1.0, 0.0, 0.0, 1.0),
):
    content_dict = {
        "type": "sphere",
        "geometry": {
            "trajectory": trajectory_exprs,
            "radius": radius,
        },
        "rgba": list(rgba),
        "movable": False,
    }
    return DynamicSphereObstacle(name=name, content_dict=content_dict)


def build_moving_cylinder(
    name: str,
    trajectory_exprs,
    radius: float = 0.5,
    height: float = 2.0,
    rgba=(1.0, 0.0, 0.0, 1.0),
):
    content_dict = {
        "type": "cylinder",
        "geometry": {
            "trajectory": trajectory_exprs,
            "radius": radius,
            "height": height,
        },
        "rgba": list(rgba),
        "movable": False,
    }
    return DynamicCylinderObstacle(name=name, content_dict=content_dict)


def apply_scenario_to_env(env: UrdfEnv, scenario_cfg: dict):
    """
    Adds walls and configured obstacles to the environment and returns a
    dictionary representation of the static/dynamic obstacles for planners.

    Returns:
        dict: {
            "wall": [
                {"position": [...], "width": w, "height": h, "length": l},
            ],
            "static": [{"center": [...], "radius": r}],
            "dynamic": [
                {
                    "start": [...],
                    "velocity": [...],
                    "heading": [...],  # unit vector, zeros if stationary
                    "radius": r,
                    "type": "sphere" | "cylinder",
                    "rgba": [r, g, b, a],
                }
            ],
        }
    """
    obstacles_dict = {"wall": [], "static": [], "dynamic": []}
    t_sym = sp.symbols("t")
    velocity_dt = 1e-2  # small step to numerically approximate velocity

    # 1) Walls and Goal Hub
    for wall in build_room_walls(wall_length=10.0):
        env.add_obstacle(wall)
        obstacles_dict["wall"].append(
            {
                "position": wall.position().tolist(),
                "width": wall.width(),
                "height": wall.height(),
                "length": wall.length(),
            }
        )
    

    # 2) Static obstacles
    for i, s_cfg in enumerate(scenario_cfg["static"]):
        cyl = build_static_cylinder(
            name=f"static_{i}",
            position=s_cfg["position"],
            radius=s_cfg.get("radius", 0.5),
        )
        env.add_obstacle(cyl)
        obstacles_dict["static"].append(
            {
                "position": list(s_cfg["position"]),
                "radius": s_cfg.get("radius", 0.5),
            }
        )

    # 3) Dynamic obstacles
    for j, d_cfg in enumerate(scenario_cfg["dynamic"]):
        dynamic_type = d_cfg.get("type", "sphere")
        dynamic_rgba = d_cfg.get("rgba", (1.0, 0.0, 0.0, 1.0))

        if "trajectory" in d_cfg:
            # Backward-compatible manual expressions.
            trajectory_exprs = d_cfg["trajectory"]
            traj_meta = None
        else:
            # Preferred: derive from start/end/speed (optionally heading, start_time).
            trajectory_exprs, traj_meta = _linear_trajectory_exprs(
                start=d_cfg["start"],
                end=d_cfg["end"],
                speed=d_cfg.get("speed", d_cfg.get("velocity", 0.0)),
                start_time=d_cfg.get("start_time", 0.0),
                stop_at_end=d_cfg.get("stop_at_end", True),
                heading=d_cfg.get("heading"),
            )

        if dynamic_type == "cylinder":
            dyn = build_moving_cylinder(
                name=f"dynamic_{j}",
                trajectory_exprs=trajectory_exprs,
                radius=d_cfg.get("radius", 0.5),
                height=d_cfg.get("height", 2.0),
                rgba=dynamic_rgba,
            )
        else:
            dyn = build_moving_sphere(
                name=f"dynamic_{j}",
                trajectory_exprs=trajectory_exprs,
                radius=d_cfg.get("radius", 0.5),
                rgba=dynamic_rgba,
            )
        prev_ids = set(env.get_obstacles().keys())
        env.add_obstacle(dyn)
        new_ids = set(env.get_obstacles().keys()) - prev_ids
        dyn_obst_id = next(iter(new_ids)) if new_ids else None

        # Build simple trajectory functions to extract start/velocity info
        if traj_meta:
            start_pos = traj_meta["start"]
            velocity = traj_meta["velocity"]
            heading = traj_meta["heading"]
        else:
            exprs = [
                sp.sympify(expr, locals={"t": t_sym, "sp": sp})
                for expr in trajectory_exprs
            ]
            lambdas = [sp.lambdify(t_sym, expr, "numpy") for expr in exprs]

            start_pos = [float(fn(0.0)) for fn in lambdas]
            pos_after_dt = [float(fn(velocity_dt)) for fn in lambdas]
            velocity = [
                (after - start) / velocity_dt
                for after, start in zip(pos_after_dt, start_pos)
            ]
            heading_norm = math.sqrt(sum(comp**2 for comp in velocity))
            heading = (
                [comp / heading_norm for comp in velocity]
                if heading_norm > 1e-9
                else [0.0] * len(velocity)
            )

        obstacles_dict["dynamic"].append(
            {
                "position": start_pos,
                "velocity": velocity,
                "heading": heading,
                "radius": d_cfg.get("radius", 0.5),
                "type": dynamic_type,
                "id": dyn_obst_id,
                **(
                    {
                        "duration": traj_meta["duration"],
                        "end_time": traj_meta["end_time"],
                        "start_time": traj_meta["start_time"],
                    }
                    if traj_meta
                    else {}
                ),
                "rgba": list(dynamic_rgba),
            }
        )

    return obstacles_dict


def refresh_dynamic_obstacle_states(env: UrdfEnv, obstacles_dict: dict):
    """
    Update the dynamic obstacles section of the obstacles_dict with the current
    position/velocity/heading using the live trajectories in env._obsts.
    """
    if "dynamic" not in obstacles_dict:
        return obstacles_dict

    for dyn_entry in obstacles_dict["dynamic"]:
        dyn_id = dyn_entry.get("id")
        if dyn_id is None:
            continue

        obs = env.get_obstacles().get(dyn_id)
        if obs is None or not isinstance(obs, DynamicObstacle):
            continue

        t_now = env.t() if hasattr(env, "t") else 0.0
        pos = obs.position(t=t_now).tolist()
        vel = obs.velocity(t=t_now).tolist()
        heading_norm = math.sqrt(sum(v * v for v in vel))
        heading = (
            [v / heading_norm for v in vel] if heading_norm > 1e-9 else [0.0] * len(vel)
        )

        dyn_entry["position"] = pos
        dyn_entry["velocity"] = vel
        dyn_entry["heading"] = heading

    return obstacles_dict
