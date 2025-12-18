import numpy as np

def get_lookahead_point(global_path, current_position, lookahead_distance):
    current_position = np.array(current_position)
    dist_to_next_point = np.linalg.norm(np.array(global_path[0]) - current_position)
    if dist_to_next_point >= lookahead_distance:
        lookahead_point = current_position + (global_path[0] - current_position)/dist_to_next_point * lookahead_distance
        return lookahead_point
    else:
        next_wp = np.array(global_path.pop(0))
        if len(global_path) == 0:
            return next_wp  # no more points, return current position
        else:
            unit_vector = (global_path[0] - next_wp) / np.linalg.norm(np.array(global_path[0]) - next_wp) 
            lookahead_point = next_wp + unit_vector * lookahead_distance / 2
            return lookahead_point

print(get_lookahead_point([ [2,2]], [1.5,2], 1.0))