# Robot Planners
Mobile manipulator (Albert robot) in pybullet environment. 
- Motion planners for navigation around obstacles 
- Motion planners for 7DoF arm movement
## Setup
1. Ubuntu22, python3.10, venv
```bash
sudo apt install python3-venv python3-pip
git clone git@github.com:dstownsend/RO47005_Project.git
cd RO47005_Project
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```
2. [Acados (for mpc)](https://docs.acados.org/python_interface/index.html#installation)

## Quickstart
Remember to activate the venv before running the following commands.
1. Run example env script
```bash
python3 examples/point_robot.py
```
1. Run example albert script. Press the arrow keys to move, ESC to quit.
```bash
python3 examples/albert.py
```
2. Run main script
```bash
python3 main.py
```

---

## RRT Base Planner Parameters

| Code Parameter | Value | Description |
|---------------|-------|-------------|
| `q` | 0.1 | Tree expansion step size. Maximum distance a new node is steered toward a sampled state per iteration. |
| `r` | 0.01 | Goal tolerance radius. The planner terminates successfully when a node is within this Euclidean distance of the goal. |
| `max_samples` | 1000 | Maximum number of sampled nodes allowed before declaring failure. |
| `rewire_count` | 32 | Number of nearby nodes evaluated for rewiring in RRT* to improve path optimality. |
| `prc` | 0.1 | Goal bias probability. Fraction of iterations in which the goal is directly sampled. |

---
