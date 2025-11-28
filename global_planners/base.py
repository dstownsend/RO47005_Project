from abc import ABC, abstractmethod


class BaseGlobalPlanner(ABC):
    """Abstract base class for all global planners."""

    @abstractmethod
    def plan(self, start, goal, map_data):
        """
        Compute a path (waypoints) from start to goal given map_data.
        Must be implemented by subclasses.
        """
        pass