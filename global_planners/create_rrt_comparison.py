from .global_planner_rrt import RRT_planner

from rrt_algorithms.utilities.plotting import Plot
from rrt_algorithms.search_space.search_space import SearchSpace
import math
import plotly
import plotly.graph_objects as go



class RRT_type_comparison:
    def __init__(self,X_dimensions,obstacles,q,r,max_samples,rewire_count,x_init,x_goal,prc):
        self.X_dimensions = X_dimensions
        self.obstacles = obstacles
        self.q = q 
        self.r = r 
        self.max_samples = max_samples
        self.rewire_count = rewire_count
        self.prc = prc 

        self.x_init = x_init
        self.x_goal = x_goal
    
    def path_results(self):
        RRT_planners = RRT_planner(
            self.X_dimensions,
            self.obstacles,
            self.q,
            self.r,
            self.max_samples,
            self.rewire_count
        )
        path_base = RRT_planners.rrt(self.x_init,self.x_goal,self.prc,plot_bool=False)
        path_star = RRT_planners.rrt_star(self.x_init,self.x_goal,self.prc,plot_bool=False)
        path_star_bd = RRT_planners.rrt_star_bd(self.x_init, self.x_goal, self.prc, plot_bool=False)
        path_star_bd_h = RRT_planners.rrt_star_bd_h(self.x_init, self.x_goal, self.prc, plot_bool=False)
        path_connect = RRT_planners.rrt_connect(self.x_init, self.x_goal, self.prc, plot_bool=False)
        return path_base,path_star,path_star_bd,path_star_bd_h,path_connect

    def plot_global_paths(self):
        path_base, path_star, path_star_bd, path_star_bd_h, path_connect = self.path_results()

        fig = go.Figure()

        # obstacles (assumes [x0, y0, x1, y1])
        for x0, y0, x1, y1 in self.obstacles:
            fig.add_trace(go.Scatter(
                x=[x0, x1, x1, x0, x0],
                y=[y0, y0, y1, y1, y0],
                fill="toself",
                mode="lines",
                line=dict(color="purple"),
                showlegend=False
            ))

        paths  = [path_base, path_star, path_star_bd, path_star_bd_h, path_connect]
        names  = ["RRT", "RRT*", "RRT* BD", "RRT* BD-H", "RRT-Connect"]
        colors = ["red", "blue", "green", "orange", "black"]

        for path, name, color in zip(paths, names, colors):
            if path is None:
                continue
            xs = [p[0] for p in path]
            ys = [p[1] for p in path]
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=name,
                                    line=dict(color=color)))

        # start + goal
        fig.add_trace(go.Scatter(x=[self.x_init[0]], y=[self.x_init[1]],
                                mode="markers", marker=dict(color="orange", size=10),
                                name="Start"))
        fig.add_trace(go.Scatter(x=[self.x_goal[0]], y=[self.x_goal[1]],
                                mode="markers", marker=dict(color="green", size=10),
                                name="Goal"))

        fig.update_layout(title="RRT Paths", xaxis_title="x", yaxis_title="y")
        fig.show()


    def determine_metrics(self):
        # get all paths internally
        path_base, path_star, path_star_bd, path_star_bd_h, path_connect = self.path_results()

        def path_length(points):
            if points is None or len(points) < 2:
                return 0.0 # the length is not 0.0 but this allows us to easily iterate later on 
            total = 0.0
            for i in range(1, len(points)):
                x1, y1 = points[i-1]
                x2, y2 = points[i]
                total += math.hypot(x2 - x1, y2 - y1)
            return total

        paths = [path_base, path_star, path_star_bd, path_star_bd_h, path_connect]
        lengths = [path_length(path) for path in paths]

        # use existing helpers to get *sample counts*
        RRT_planners = RRT_planner(
            self.X_dimensions,
            self.obstacles,
            self.q,
            self.r,
            self.max_samples,
            self.rewire_count
        )
        samples_base = RRT_planners.rrt_current_samples(self.x_init,self.x_goal,self.prc)
        samples_star = RRT_planners.rrt_star_current_samples(self.x_init,self.x_goal,self.prc)
        samples_star_bd = RRT_planners.rrt_star_bd_current_samples(self.x_init, self.x_goal, self.prc)
        samples_star_bd_h = RRT_planners.rrt_star_bd_h_current_samples(self.x_init, self.x_goal, self.prc)
        samples_connect = RRT_planners.rrt_connect_current_samples(self.x_init, self.x_goal, self.prc)

        samples = [samples_base, samples_star, samples_star_bd, samples_star_bd_h, samples_connect]

        return lengths, samples
