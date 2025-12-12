from files_christian.base_controller.create_rrt_comparison import RRT_type_comparison
import numpy as np

X_dimensions = np.array([(0, 10), (0, 10)])

# obstacles must be [x_min, y_min, x_max, y_max]
obstacles = np.array([
    (2, 2, 4, 4),
    (6, 6, 8, 8)
])

q = 0.5               # step size
r = 0.01               # neighbor radius for RRT*
max_samples = 1000
rewire_count = 100
prc = 0.1             # goal bias

x_init = (0.5, 0.5)
x_goal = (9.0, 9.0)

layers = RRT_type_comparison(
    X_dimensions, obstacles, q, r,
    max_samples, rewire_count,
    x_init, x_goal, prc
)

layers.plot_global_paths()
lengths, samples = layers.determine_metrics()

print()
print()
print("Path lengths [basic, star, star_bd, star_bd_h, connect]:")
print(lengths)
print("Samples used   [basic, star, star_bd, star_bd_h, connect]:")
print(samples)
