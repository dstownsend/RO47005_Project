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