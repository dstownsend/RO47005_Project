from global_planners.create_rrt_comparison import RRT_type_comparison
from global_planners.global_planner_rrt import RRT_planner
import numpy as np
import random
from rich.table import Table
from rich.console import Console

X_dimensions = np.array([(0, 10), (0, 10)])

# Original regions are defined in [-5,5] world coordinates
RANDOM_STATIC_REGIONS = [
    ( 2.75, -4.25,  4.25,  0.25, 0.75),
    ( 0.25, -4.25,  1.25,  0.25, 0.75),
    (-2.25, -4.25, -1.25,  0.25, 0.75),
    ( 0.25,  1.75,  1.25,  4.25, 0.75),
    (-2.25,  1.75, -1.25,  4.25, 0.75),
    (-4.25,  1.25, -3.25,  4.25, 0.75),
]

def center_radius_to_rect(cx, cy, r):
    return (cx - r, cy - r, cx + r, cy + r)

def shift_minus5_5_to_0_10(x, y):
    return x + 5.0, y + 5.0

def generate_random_static_rectangles(seed=None):
    rng = random.Random(seed)
    rects = []

    for lx, ly, hx, hy, r in RANDOM_STATIC_REGIONS:
        cx = rng.uniform(lx, hx)
        cy = rng.uniform(ly, hy)

        xmin, ymin, xmax, ymax = center_radius_to_rect(cx, cy, r)

        xmin, ymin = shift_minus5_5_to_0_10(xmin, ymin)
        xmax, ymax = shift_minus5_5_to_0_10(xmax, ymax)

        rects.append((xmin, ymin, xmax, ymax))

    return np.array(rects, dtype=float)






obstacles = generate_random_static_rectangles()


# width x dir = 2x 0.327 = 0.654
# length y dir = 0.237−(−0.3365) = 0.5735
# maximum 'length' to consider = 0.3365

def add_dilation(tupl,dilation):
    x_min_new = tupl[0] - dilation
    y_min_new = tupl[1] - dilation
    x_max_new = tupl[2] + dilation
    y_max_new = tupl[3] + dilation
    new_tuple = (x_min_new,y_min_new,x_max_new,y_max_new)
    return new_tuple

dilated_obstacles = []
for el in obstacles:
    dilated_el = add_dilation(el,0.03365)
    dilated_obstacles.append(dilated_el)

wall_obstacles = np.array([
    (-0.05, 0.0, 0.05, 10.0),    # -x wall
    (9.95, 0.0, 10.05, 10.0),   # +x wall
    (0.0, -0.05, 10.0, 0.05),   # -y wall
    (0.0, 9.95, 10.0, 10.05),   # +y wall
])

hub_wall_obstacles = np.array([
    (7.99, 8.0, 8.01, 10),
    (8.0, 7.99, 8.5, 8.01),
])

for el in wall_obstacles:
    dilated_obstacles.append(add_dilation(el,0.03365))
for el in hub_wall_obstacles:
    dilated_obstacles.append(add_dilation(el,0.03365))

dilated_obstacles = np.array(dilated_obstacles)



q = 0.1
r = 0.01
max_samples = 1000
rewire_count = 32
prc = 0.1

x_init = (0.5, 0.5)
x_goal = (9.0, 9.0)


layers = RRT_type_comparison(
    X_dimensions, dilated_obstacles, q, r,
    max_samples, rewire_count,
    x_init, x_goal, prc
)

layers.plot_global_paths()
lengths, samples = layers.determine_metrics()
lengths = np.round(lengths,2)


planner = RRT_planner(
    X_dimensions, dilated_obstacles, q, r,
    max_samples, rewire_count
)

t_basic, cpu_basic = planner.rrt_runtime_cpu(x_init, x_goal, prc)
t_star, cpu_star = planner.rrt_star_runtime_cpu(x_init, x_goal, prc)
t_star_bd, cpu_star_bd = planner.rrt_star_bd_runtime_cpu(x_init, x_goal, prc)
t_star_bd_h, cpu_star_bd_h = planner.rrt_star_bd_h_runtime_cpu(x_init, x_goal, prc)
t_connect, cpu_connect = planner.rrt_connect_runtime_cpu(x_init, x_goal, prc)

# ===================== MULTI-TRIAL METRICS (drop-in) =====================

ALG_NAMES = ["basic", "star", "star_bd", "star_bd_h", "connect"]

N_TRIALS = 0
BASE_SEED = 0
DILATION = 0.03365

def build_dilated_obstacles(seed=None, dilation=DILATION):
    obstacles = generate_random_static_rectangles(seed=seed)
    dilated = [add_dilation(el, dilation) for el in obstacles]

    wall_obstacles = np.array([
        (-0.05, 0.0, 0.05, 10.0),
        (9.95, 0.0, 10.05, 10.0),
        (0.0, -0.05, 10.0, 0.05),
        (0.0, 9.95, 10.0, 10.05),
    ], dtype=float)

    hub_wall_obstacles = np.array([
        (7.99, 8.0, 8.01, 10.0),
        (8.0, 7.99, 8.5, 8.01),
    ], dtype=float)

    for el in wall_obstacles:
        dilated.append(add_dilation(el, dilation))
    for el in hub_wall_obstacles:
        dilated.append(add_dilation(el, dilation))

    return np.array(dilated, dtype=float)

def safe_lengths_from_layers(layers_obj, k=len(ALG_NAMES)):
    try:
        lengths, _ = layers_obj.determine_metrics()
        lengths = np.asarray(lengths, dtype=float)
        ok = np.isfinite(lengths) & (lengths > 0.0)
        lengths = np.where(ok, lengths, 0.0)
        return lengths, ok
    except Exception:
        return np.zeros(k), np.zeros(k, dtype=bool)

def safe_runtime(fn, *args):
    try:
        t, _ = fn(*args)
        return float(t) if np.isfinite(t) and t > 0 else 0.0
    except Exception:
        return 0.0

len_hist = np.zeros((N_TRIALS, len(ALG_NAMES)))
time_hist = np.zeros((N_TRIALS, len(ALG_NAMES)))
succ_len  = np.zeros((N_TRIALS, len(ALG_NAMES)), dtype=bool)

for i in range(N_TRIALS):
    seed_i = BASE_SEED + i if BASE_SEED is not None else None
    dilated_obstacles = build_dilated_obstacles(seed_i)

    layers = RRT_type_comparison(
        X_dimensions, dilated_obstacles, q, r,
        max_samples, rewire_count,
        x_init, x_goal, prc
    )

    lengths, ok = safe_lengths_from_layers(layers)
    len_hist[i] = lengths
    succ_len[i] = ok

    planner = RRT_planner(
        X_dimensions, dilated_obstacles, q, r,
        max_samples, rewire_count
    )

    time_hist[i] = [
        safe_runtime(planner.rrt_runtime_cpu, x_init, x_goal, prc),
        safe_runtime(planner.rrt_star_runtime_cpu, x_init, x_goal, prc),
        safe_runtime(planner.rrt_star_bd_runtime_cpu, x_init, x_goal, prc),
        safe_runtime(planner.rrt_star_bd_h_runtime_cpu, x_init, x_goal, prc),
        safe_runtime(planner.rrt_connect_runtime_cpu, x_init, x_goal, prc),
    ]

# -------- averages EXCLUDING FAILED TRIALS ----------
avg_len = []
avg_time = []
success_rate = np.mean(succ_len, axis=0)

for j in range(len(ALG_NAMES)):
    valid_idx = succ_len[:, j]

    avg_len.append(
        np.mean(len_hist[valid_idx, j]) if np.any(valid_idx) else 0.0
    )

    avg_time.append(
        np.mean(time_hist[valid_idx, j]) if np.any(valid_idx) else 0.0
    )

avg_len = np.round(avg_len, 2)
avg_time = np.round(avg_time, 4)

print("\n\n\n\n")
c = Console()
t = Table(title=f"Results over {N_TRIALS} trials (failed runs excluded from averages)")
[t.add_column(x, justify="right") for x in ["Metric"] + ALG_NAMES]
t.add_row("Avg path length", *map(str, avg_len))
t.add_row("Avg time (s)", *map(lambda x: f"{x:.4f}", avg_time))
t.add_row("Success rate (length)", *[f"{100*v:.1f}%" for v in success_rate])
c.print(t)
print("\n\n\n\n")

# =================== END MULTI-TRIAL METRICS ===================



# print()
# print()
# print()
# print()

# c=Console(); t=Table(title="Results")
# [t.add_column(x, justify="right") for x in ["Metric","basic","star","star_bd","star_bd_h","connect"]]
# t.add_row("Path length",*map(str,lengths))
# #t.add_row("Samples",*map(str,samples))
# t.add_row("Time (s)",*map(lambda x:f"{x:.4f}",[t_basic,t_star,t_star_bd,t_star_bd_h,t_connect]))
# #t.add_row("CPU (%)",*map(lambda x:f"{x:.2f}",[cpu_basic,cpu_star,cpu_star_bd,cpu_star_bd_h,cpu_connect]))
# c.print(t)

# print()
# print()
# print()
# print()