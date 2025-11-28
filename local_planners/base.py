from abc import ABC, abstractmethod


class BaseLocalPlanner(ABC):
    """Abstract base class for all local planners."""

    @abstractmethod
    def plan(self, start, goal, map_data):
        """
        Compute a plan from start to goal given map_data.
        Must be implemented by subclasses.
        """
        pass