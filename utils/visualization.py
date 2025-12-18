import pybullet as p
from sympy import li

def draw_points_and_path(waypoints, lifetime=0):
    # convert 2D waypoints -> 3D (small z offset so points are visible)
    waypoints_3d = [[float(x), float(y), 0.01] for (x, y) in waypoints]

    # draw waypoints 
    p.addUserDebugPoints(
        pointPositions=waypoints_3d,
        pointColorsRGB=[[0, 0, 1]] * len(waypoints_3d),
        pointSize=10,
        lifeTime=lifetime
    )
    # draw path lines
    for a, b in zip(waypoints_3d, waypoints_3d[1:]):
        p.addUserDebugLine(a, b, [0, 1, 0], lineWidth=3, lifeTime=lifetime)
        
        
class PathUpdater:
    def __init__(self):
        self.line_ids = [] # to replace for faster rendering

    def draw_path(self, traj):
        traj3d = [[float(x), float(y), 0.01] for x, y in traj]
        if len(self.line_ids) == 0:
            # save ids if first time drawing
            self.line_ids = [
                p.addUserDebugLine(a, b, [1, 0, 0], lineWidth=2)
                for a, b in zip(traj3d, traj3d[1:])
            ]
        else:
            for lid, a, b in zip(self.line_ids, traj3d, traj3d[1:]):
                p.addUserDebugLine(
                    a, b, [1, 0, 0], 2,
                    replaceItemUniqueId=lid,
                )