import numpy as np

def get_lookahead_point(global_path, current_position, lookahead_distance):
    """Get a lookahead point at a fixed distance ahead, else the last point.

    Parameters
    ----------
    global_path : list of list of float
        The global path defined by waypoints.
    current_position : list of float
        The current position of the robot.
    lookahead_distance : float
        The distance to look ahead along the path.

    Returns
    -------
    list of float
        The lookahead point coordinates.
    """
    current_position = np.array(current_position)
    next_wp = np.array(global_path[0])

    dist_to_next_point = np.linalg.norm(next_wp - current_position)
    if dist_to_next_point >= lookahead_distance: # lookahead point is before next point, add unit vector of lookahead dist
        lookahead_point = current_position + (next_wp - current_position)/dist_to_next_point * lookahead_distance
        return lookahead_point
    else: # lookahead point is after next point
        if len(global_path) == 1:
            return next_wp  # no more points, return next_wp
        else:
            # we assume the next next wp is after lookahead dist. If not, this should still compute but lookahead point will not actually be on global path.
            next_next_wp = np.array(global_path[1])
            unit_vector = (next_next_wp - next_wp) / np.linalg.norm(next_next_wp - next_wp)
            lookahead_point = next_wp + unit_vector * (lookahead_distance - dist_to_next_point)
            return lookahead_point
        
def generate_reference_to_lookahead(
    x0,
    lookahead_point,
    N,
    Ts,
    v_ref
):
    """
    Generate a straight-line state reference trajectory from the
    current state to a lookahead point.

    State is assumed to be [x, y, theta].

    Parameters
    ----------
    x0 : array_like, shape (3,)
        Current state [x, y, theta]
    lookahead_point : array_like, shape (2,)
        Lookahead point [x, y]
    N : int
        MPC horizon length
    Ts : float
        Sampling time
    v_ref : float
        Desired forward speed

    Returns
    -------
    xref_traj : ndarray, shape (N+1, 3)
        State reference trajectory
    """
    x0 = np.asarray(x0, dtype=float)
    p0 = x0[:2]
    theta0 = x0[2]
    pL = np.asarray(lookahead_point, dtype=float)

    # Direction and distance to lookahead
    dvec = pL - p0
    dist = np.linalg.norm(dvec)

    # Degenerate case: already at lookahead
    if dist < 1e-6:
        xref = np.zeros((N+1, 3))
        xref[:, 0] = p0[0]
        xref[:, 1] = p0[1]
        xref[:, 2] = theta0
        return xref

    direction = dvec / dist
    theta_ref = np.arctan2(direction[1], direction[0])

    step = v_ref * Ts

    xref_traj = np.zeros((N+1, 3))

    for k in range(N+1):
        s = min(k * step, dist)
        pos = p0 + s * direction

        xref_traj[k, 0] = pos[0]
        xref_traj[k, 1] = pos[1]
        xref_traj[k, 2] = theta_ref

    return xref_traj



print(get_lookahead_point([ [1,1],[1,2],[2,2] ], [0,1], 2.5))
print(get_lookahead_point([ [1,1] ], [0,1], 2.5))