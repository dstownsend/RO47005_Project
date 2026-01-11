# Project repo for RO47005 Group 43
## Maintainers
Lau Wei Cheng, Josh (6379184)  
Plevier, Christian (5041120)  
Sackmann, Luca (6376754)  
Townsend, Dylan (6315577)

## Setup and installation
This project was developed with Ubuntu22, python3.10.
1. Create venv and install python packages
    ```bash
    sudo apt install python3-venv python3-pip
    git clone git@github.com:dstownsend/RO47005_Project.git
    cd RO47005_Project
    python3 -m venv venv
    source venv/bin/activate
    pip3 install -r requirements.txt
    ```
2. Install Acados following [this](https://docs.acados.org/installation/index.html) and [this](https://docs.acados.org/python_interface/index.html#installation) link. Note that the first time you run Acados, you will get a prompt in your console "Do you wish to set up Tera renderer automatically?" Enter Y.

3. Install RRT library. 
    ```bash
    source venv/bin/activate # make sure venv is activated
    git clone git@github.com:motion-planning/rrt-algorithms.git
    cd rrt-algorithms
    pip3 install -e .
    ```

## Quickstart
1. Activate the venv
    ```bash
    source venv/bin/activate
    ```
2. Run main script. Pybullet should start and you should see the robot navigating through the env (see gif below)
    ```bash
    python3 main.py # from project root
    ```
- To see an image of the generated global path, set in `global_path = RRT_planner.plan(..., plot_bool=True)` in `main.py`.
- You may interact with the camera view in pybullet using mouse control (be careful not to accidently drag and move the robot)
- More videos can be found at [this link](https://tud365-my.sharepoint.com/:f:/r/personal/lsackmann_tudelft_nl/Documents/RO47005_Project/Videos?csf=1&web=1&e=JciZgA) (log in required)
![full run](media/MPC_Overall.gif)  
*Top down view of base movement*
![full run](media/RRT_MiddleWall.gif)  
*Close up view of Manipulator task*


## Obstacle avoidance
You can run the tests for obstacle avoidance in a standalone scenario (currently implemented in different branches).
### Static obstacle avoidance
Switch to the branch `feat/static-obs`. In `main.py`, ensure that `scenario_name="one_static"`, you can refer to and update the configurations [here](environment/scenarios.py).
```bash
source venv/bin/activate
git switch feat/static-obs
python3 main.py
```
### Dynamic obstacle avoidance
Switch to the branch `feat/dynamic-obs`. In `main.py`, ensure that `scenario_name="one_dynamic"`, you can refer to and update the configurations in `environment/scenarios.py`.
```bash
source venv/bin/activate
git switch feat/dynamic-obs
python3 main.py
```

### MPC Parameters
The MPC implementation and parameters can be found [here](local_planners/mpc.py) 
| Parameter | Description |
|---------------|--------------------|
| NUM_HORIZON_STEPS | Number of steps in planning horizon over total time horizon (next parameter) |
| TIME_HORIZON_S | Total planning time horizon in seconds. |
| Q_MAT | Cost (penalty) for tracking a position [x,y,angular displacement]|
| Q_MAT_E | Cost for tracking the terminal state in the planning horizon |
| R_MAT | Cost for control effort. [forward velocity, angular velocity] |
| INFLATION_M | Inflation radius of obstacles in meters.|
| OBS_SOFT_COST | Cost for hitting an obstacle |
## RRT Base Planner Parameters

| Code Parameter | Value | Description |
|---------------|-------|-------------|
| `q` | 0.1 | Tree expansion step size. Maximum distance a new node is steered toward a sampled state per iteration. |
| `r` | 0.01 | Goal tolerance radius. The planner terminates successfully when a node is within this Euclidean distance of the goal. |
| `max_samples` | 1000 | Maximum number of sampled nodes allowed before declaring failure. |
| `rewire_count` | 32 | Number of nearby nodes evaluated for rewiring in RRT* to improve path optimality. |
| `prc` | 0.1 | Goal bias probability. Fraction of iterations in which the goal is directly sampled. |

---
