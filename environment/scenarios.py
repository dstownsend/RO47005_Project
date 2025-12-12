import math
import random

# Each scenario: lists of configs for static and dynamic obstacles.
# You can define as many as you like.

SCENARIOS = {
    # 0) Basic environment
    "empty": {
        "static": [],
        "dynamic": [],
    },
    "one_static": {
        "static": [
            {"position": [0.0, 0.0, 0.0], "radius": 1.5},
        ],
        "dynamic": [],
    },
    "one_dynamic": {
        "static": [],
        "dynamic": [
            {
                # Starts at start_time, moves at speed toward end, stops at final position.
                "start": [-2.0, 0.0, 0.1],
                "end": [4.0, -3.0, 0.1],
                "speed": 5,
                "start_time": 2.0,
                "radius": 0.5,
            },
        ],
    },
    "one_random_static": {
        "static": [
            {
                "radius": 0.75,
                "height": 2.0,
                "rgba": [0.8, 0.2, 0.2, 1.0],
                # Randomly place within x/y in [-5, 5]; keep z, radius, height fixed.
                "low": {"position": [-5.0, -5.0, 0.0], "radius": 0.75, "height": 2.0},
                "high": {"position": [5.0, 5.0, 0.0], "radius": 0.75, "height": 2.0},
                "randomize": True,
            },
        ],
        "dynamic": [],
    },
    "random_static": {
        "static": [
            {
                "radius": 0.75,
                "height": 2.0,
                "rgba": [0.8, 0.2, 0.2, 1.0],
                # Randomly place within x/y in [-5, 5]; keep z, radius, height fixed.
                "low": {"position": [2.75, -4.25, 0.0], "radius": 0.75, "height": 2.0},
                "high": {"position": [4.25, 0.25, 0.0], "radius": 0.75, "height": 2.0},
                "randomize": True,
            },
            {
                "radius": 0.75,
                "height": 2.0,
                "rgba": [0.8, 0.2, 0.2, 1.0],
                # Randomly place within x/y in [-5, 5]; keep z, radius, height fixed.
                "low": {"position": [0.25, -4.25, 0.0], "radius": 0.75, "height": 2.0},
                "high": {"position": [1.25, 0.25, 0.0], "radius": 0.75, "height": 2.0},
                "randomize": True,
            },
            {
                "radius": 0.75,
                "height": 2.0,
                "rgba": [0.8, 0.2, 0.2, 1.0],
                # Randomly place within x/y in [-5, 5]; keep z, radius, height fixed.
                "low": {"position": [-2.25, -4.25, 0.0], "radius": 0.75, "height": 2.0},
                "high": {"position": [-1.25, 0.25, 0.0], "radius": 0.75, "height": 2.0},
                "randomize": True,
            },
            {
                "radius": 0.75,
                "height": 2.0,
                "rgba": [0.8, 0.2, 0.2, 1.0],
                # Randomly place within x/y in [-5, 5]; keep z, radius, height fixed.
                "low": {"position": [0.25, 1.75, 0.0], "radius": 0.75, "height": 2.0},
                "high": {"position": [1.25, 4.25, 0.0], "radius": 0.75, "height": 2.0},
                "randomize": True,
            },
            {
                "radius": 0.75,
                "height": 2.0,
                "rgba": [0.8, 0.2, 0.2, 1.0],
                # Randomly place within x/y in [-5, 5]; keep z, radius, height fixed.
                "low": {"position": [-2.25, 1.75, 0.0], "radius": 0.75, "height": 2.0},
                "high": {"position": [-1.25, 4.25, 0.0], "radius": 0.75, "height": 2.0},
                "randomize": True,
            },
            {
                "radius": 0.75,
                "height": 2.0,
                "rgba": [0.8, 0.2, 0.2, 1.0],
                # Randomly place within x/y in [-5, 5]; keep z, radius, height fixed.
                "low": {"position": [-4.25, 1.25, 0.0], "radius": 0.75, "height": 2.0},
                "high": {"position": [-3.25, 4.25, 0.0], "radius": 0.75, "height": 2.0},
                "randomize": True,
            },
        ],
        "dynamic": [],
    },
    "static_and_dynamic": {
        "static": [
            {"position": [0.0, 3.0, 0.0], "radius": 1.0},
            {"position": [0.0, 1.0, 0.0], "radius": 1.0},
            {"position": [-0.5, 4.5, 0.0], "radius": 0.5},
            {"position": [0.5, 4.5, 0.0], "radius": 0.5}, 
            {"position": [2.0, 0.0, 0.0], "radius": 1.0}, # lower set of obstacles
            {"position": [0.0, -3.0, 0.0], "radius": 1.0},
            {"position": [-0.5, -4.5, 0.0], "radius": 0.5},
            {"position": [0.5, -4.5, 0.0], "radius": 0.5},
            {"position": [-2.0, -2, 0.0], "radius": 1.0}, # upper set of obstacles
        ],
        "dynamic": [
            {
                # Starts at start_time, moves at speed toward end, stops at final position.
                "start": [-2.0, 0.0, 0.1],
                "end": [4.0, -3.0, 0.1],
                "speed": 5,
                "start_time": 2.0,
                "radius": 0.5,
            },
        ],
    },
    "dynamic_cylinders": {
        "static": [],
        "dynamic": [
            {
                # Starts at start_time, moves at speed toward end, stops at final position.
                "start": [-2.0, 0.0, 0.1],
                "end": [4.0, -3.0, 0.1],
                "speed": 5,
                "start_time": 2.0,
                "radius": 0.5,
            },
            {
                # Starts at start_time, moves at speed toward end, stops at final position.
                "start": [-2.0, 1.0, 0.1],
                "end": [3.0, 3.0, 0.1],
                "speed": 5,
                "start_time": 2.0,
                "radius": 0.5,
            },
        ],
    },

}

def get_scenario(name: str):
    return SCENARIOS[name]
