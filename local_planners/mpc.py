import numpy as np
import logging
from scipy.linalg import block_diag

from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from casadi import SX, vertcat, sin, cos

from .base import BaseLocalPlanner

NUM_HORIZON_STEPS = 50
TIME_HORIZON_S = 2.0
V_MAX_M_S = 0.3
OMEGA_MAX_RAD_S = 0.5
# Cost to minimize distance to goal and control effort
Q_MAT = np.diag([1e3, 1e3, 1e-3])  # [x,y,theta]
Q_MAT_E = np.diag([1e3, 1e3, 1e1])  # [x,y,theta]
R_MAT = np.diag([1e1, 1e1])  # [v, theta_d]
logger = logging.getLogger(__name__)


class MPC(BaseLocalPlanner):
    
    def __init__(self, robot_model):
        self.robot_model = robot_model
        self.ocp = self._create_ocp()
        self.ocp_solver = AcadosOcpSolver(self.ocp)
        
        nx = self.robot_model.x.rows()
        nu = self.robot_model.u.rows()
        ny = nx + nu
        self.yref_e = np.zeros((nx,))
        self.yref = np.zeros((ny,))

    def plan(self, current_state, goal_state, map_data):
        # TODO: add constraints to ocp based on obstacles
        # Set goal (or traj) in solver
        self.yref[:len(goal_state)] = goal_state # here we leave control refs at 0
        self.yref_e = goal_state
        for j in range(self.ocp_solver.N):
            self.ocp_solver.set(j, "yref", self.yref)
        self.ocp_solver.set(self.ocp_solver.N, "yref", self.yref_e)
        
        logger.debug("\t", self.ocp_solver.get_cost())
        logger.debug(current_state)
        
        control = self.ocp_solver.solve_for_x0(current_state)
        return control
    
    def _create_ocp(self) -> AcadosOcp:    
        ocp = AcadosOcp()
        ocp.model = self.robot_model
        # prediction horizon
        ocp.solver_options.N_horizon = NUM_HORIZON_STEPS
        ocp.solver_options.tf = TIME_HORIZON_S
        
        # set options
        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM" 
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "ERK" # required for explicit model
        ocp.solver_options.nlp_solver_type = "SQP_RTI" # sometimes no solutions without rti
        ocp.solver_options.nlp_solver_max_iter = 1000 # default max_iter is 100, errors if no solution found
        
        # Cost
        ocp.cost.cost_type = "LINEAR_LS"
        ocp.cost.cost_type_e = "LINEAR_LS"
        
        ocp.cost.W = block_diag(Q_MAT, R_MAT)
        ocp.cost.W_e = Q_MAT_E
        
        nx = self.robot_model.x.rows()
        nu = self.robot_model.u.rows()
        ny = nx + nu
        ny_e = nx # only x without u at terminal

        Vx = np.zeros((ny, nx))
        Vx[:nx, :nx] = np.eye(nx)
        ocp.cost.Vx = Vx
        ocp.cost.Vx_e = np.eye(ny_e)

        Vu = np.zeros((ny, nu))
        Vu[nx:, :] = np.eye(nu)
        ocp.cost.Vu = Vu

        ocp.cost.yref = np.zeros((ny,))
        ocp.cost.yref_e = np.zeros((ny_e,))

        # Constraints
        ocp.constraints.lbu = np.array([-V_MAX_M_S, -OMEGA_MAX_RAD_S])
        ocp.constraints.ubu = np.array([V_MAX_M_S, OMEGA_MAX_RAD_S])
        ocp.constraints.idxbu = np.array([0, 1]) # V applies to 0th control, omega to 1st control

        ocp.constraints.x0 = np.array([0.0, 0.0, 0.0]) # Just to initialize, will be set at each plan() call
        # ocp.constraints.lh = np.array([0.0]) #obs avoidance, must be >=0
        # ocp.constraints.uh = np.array([1e8])   # large to indicate no upper bound

        return ocp
    

def create_mpc_planner() -> MPC:
    """Main function to call to get an mpc planner. Other objects required by MPC are initialized here.

    Returns
    -------
    MPC
        The local planner to call plan() with.
    """
    robot_model = create_robot_model()
    return MPC(robot_model)
    
def create_robot_model() -> AcadosModel:
    """Kinematic nonlinear unicycle model.

    Returns
    -------
    AcadosModel
        model required by acados for MPC formulation. Should be fed into OCP.
    """
    model_name = "unicycle_kinematic"

    # set up states & controls
    x = SX.sym("x")
    y = SX.sym("y")
    theta = SX.sym("theta")

    state_vector = vertcat(x, y, theta)

    # set up controls
    v = SX.sym("x_d")
    theta_d = SX.sym("theta_d")
    control_vector = vertcat(v, theta_d)

    # "explicit" kinematics model. the implicit one is formulated as f_impl = xdot - f_expl = 0
    f_expl = vertcat(v * cos(theta), v * sin(theta), theta_d)

    # For obstacle avoidance
    # p = SX.sym("p", 3)  # [x_obs, y_obs, r_safe]
    
    model = AcadosModel()
    model.f_expl_expr = f_expl
    model.x = state_vector
    model.u = control_vector
    # model.con_h_expr = ((x - p[0])**2 + (y - p[1])**2) - p[2]**2  # should be >= 0
    # model.p = p # set this in the ocp before each solve
    model.name = model_name

    model.t_label = "$t$ [s]"
    model.x_labels = ["$x$", "$y$", "$\\theta$"]
    model.u_labels = ["$v$", "$\\dot{\\theta}$"]

    return model
