import numpy as np

from rrt_algorithms.rrt.rrt_star import RRTStar
from rrt_algorithms.rrt.rrt_connect import RRTConnect
from rrt_algorithms.rrt.rrt import RRT
from rrt_algorithms.rrt.rrt_star_bid_h import RRTStarBidirectionalHeuristic
from rrt_algorithms.rrt.rrt_star_bid import RRTStarBidirectional
from rrt_algorithms.search_space.search_space import SearchSpace
from rrt_algorithms.utilities.plotting import Plot


class RRT_planner:
    def __init__(self, X_dimensions, obstacles, q, r, max_samples, rewire_count):
        self.X_dimensions = X_dimensions
        self.obstacles = obstacles
        self.q = q
        self.r = r
        self.max_samples = max_samples
        self.rewire_count = rewire_count    

    def plot_rrt(self, X, obstacles, x_init, x_goal, path, rrt_type, rrt):
        plot = Plot(f"{rrt_type}")        
        plot.plot_tree(X, rrt.trees)
        if path is not None:
            plot.plot_path(X, path)
        if obstacles is not None:
            plot.plot_obstacles(X, obstacles)
        plot.plot_start(X, x_init)
        plot.plot_goal(X, x_goal)
  
        plot.draw(auto_open=True)

    def rrt(self, x_init, x_goal, prc, plot_bool=False):
        X = SearchSpace(self.X_dimensions, self.obstacles)
        rrt = RRT(X, self.q, x_init, x_goal, self.max_samples, self.r, prc)
        path = rrt.rrt_search()
        if plot_bool:
            self.plot_rrt(X, self.obstacles, x_init, x_goal, path, 'basic_rrt', rrt)
        return path
    
    def rrt_star(self, x_init, x_goal, prc, plot_bool=False):
        X = SearchSpace(self.X_dimensions, self.obstacles)
        rrt = RRTStar(X, self.q, x_init, x_goal, self.max_samples, self.r, prc, self.rewire_count)
        path = rrt.rrt_star()
        if plot_bool:
            self.plot_rrt(X, self.obstacles, x_init, x_goal, path, 'rrt_star', rrt)
        return path

    def rrt_star_bd(self, x_init, x_goal, prc, plot_bool=False):
        X = SearchSpace(self.X_dimensions, self.obstacles)
        rrt = RRTStarBidirectional(
            X, self.q, x_init, x_goal, self.max_samples, self.r, prc, self.rewire_count
        )
        path = rrt.rrt_star_bidirectional()
        if plot_bool:
            self.plot_rrt(X, self.obstacles, x_init, x_goal, path, 'rrt_star_bidirectional', rrt)
        return path
    
    def rrt_star_bd_h(self, x_init, x_goal, prc, plot_bool=False):
        X = SearchSpace(self.X_dimensions, self.obstacles)
        rrt = RRTStarBidirectionalHeuristic(
            X, self.q, x_init, x_goal, self.max_samples, self.r, prc, self.rewire_count
        )
        path = rrt.rrt_star_bid_h()
        if plot_bool:
            self.plot_rrt(X, self.obstacles, x_init, x_goal, path, 'rrt_star_bidirectional_plus_heuristic', rrt)
        return path
    
    def rrt_connect(self, x_init, x_goal, prc, plot_bool=False):
        X = SearchSpace(self.X_dimensions, self.obstacles)
        rrt = RRTConnect(X, self.q, x_init, x_goal, self.max_samples, self.r, prc)
        path = rrt.rrt_connect()
        if plot_bool:
            self.plot_rrt(X, self.obstacles, x_init, x_goal, path, 'rrt_connect', rrt)
        return path



    def plan(self, rrt_type, x_init, x_goal, prc, plot_bool=False):
        """
        Plan from start to goal and choose which global RRT-based planner to execute.

        Args:
            rrt_type (str): Type of planner to use. Options:
                - 'basic_rrt': Standard RRT
                - 'rrt_star': RRT*
                - 'rrt_star_bidirectional': Bidirectional RRT*
                - 'rrt_star_bidirectional_plus_heuristic': Bidirectional RRT* with heuristic
                - 'rrt_connect': RRT-Connect
            x_init: Start state.
            x_goal: Goal state.
            prc (float): Probability of sampling the goal (goal-bias).
            plot_bool (bool): Whether to plot the resulting tree and path.

        Returns:
            list | None: Path as a list of nodes, or None if planning fails.
        """

        if rrt_type == 'basic_rrt':
            return self.rrt(x_init, x_goal, prc, plot_bool)

        elif rrt_type == 'rrt_star':
            return self.rrt_star(x_init, x_goal, prc, plot_bool)

        elif rrt_type == 'rrt_star_bidirectional':
            return self.rrt_star_bd(x_init, x_goal, prc, plot_bool)

        elif rrt_type == 'rrt_star_bidirectional_plus_heuristic':
            return self.rrt_star_bd_h(x_init, x_goal, prc, plot_bool)

        elif rrt_type == 'rrt_connect':
            return self.rrt_connect(x_init, x_goal, prc, plot_bool)

        else:
            print(f"Unknown planner type: {rrt_type}")
            return None
        
    def rrt_current_samples(self, x_init, x_goal, prc):
        X = SearchSpace(self.X_dimensions, self.obstacles)
        rrt = RRT(X, self.q, x_init, x_goal, self.max_samples, self.r, prc)
        rrt.rrt_search()
        return rrt.samples_taken

    def rrt_star_current_samples(self, x_init, x_goal, prc):
        X = SearchSpace(self.X_dimensions, self.obstacles)
        rrt = RRTStar(X, self.q, x_init, x_goal, self.max_samples, self.r, prc, self.rewire_count)
        rrt.rrt_star()
        return rrt.samples_taken

    def rrt_star_bd_current_samples(self, x_init, x_goal, prc):
        X = SearchSpace(self.X_dimensions, self.obstacles)
        rrt = RRTStarBidirectional(
            X, self.q, x_init, x_goal, self.max_samples, self.r, prc, self.rewire_count
        )
        rrt.rrt_star_bidirectional()
        return rrt.samples_taken

    def rrt_star_bd_h_current_samples(self, x_init, x_goal, prc):
        X = SearchSpace(self.X_dimensions, self.obstacles)
        rrt = RRTStarBidirectionalHeuristic(
            X, self.q, x_init, x_goal, self.max_samples, self.r, prc, self.rewire_count
        )
        rrt.rrt_star_bid_h()
        return rrt.samples_taken

    def rrt_connect_current_samples(self, x_init, x_goal, prc):
        X = SearchSpace(self.X_dimensions, self.obstacles)
        rrt = RRTConnect(X, self.q, x_init, x_goal, self.max_samples, self.r, prc)
        rrt.rrt_connect()
        return rrt.samples_taken

