from .base import BaseGlobalPlanner
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class RRTStar(BaseGlobalPlanner):
    
    def __init__(self):
        logger.info("test from rrtstar")
    
    def plan(self, start, goal, map_data):
        pass