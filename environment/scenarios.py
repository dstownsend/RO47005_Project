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

    # 5 "real" training scenarios, just examples:
    "train_1": { # only static
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
                "start": [-2.0, -1.0, 0.1],
                "end": [4.0, -3.0, 0.1],
                "speed": 5,
                "start_time": 2.0,
                "radius": 0.5,
            },
        ],
    },
    "train_2": {
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
                "start": [-2.0, -1.0, 0.1],
                "end": [4.0, -3.0, 0.1],
                "speed": 5,
                "start_time": 2.0,
                "radius": 0.5,
            },
        ],
    },
    "train_3": {
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
                "start": [-2.0, -1.0, 0.1],
                "end": [4.0, -3.0, 0.1],
                "speed": 5,
                "start_time": 2.0,
                "radius": 0.5,
            },
        ],
    },
    "train_4": {
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
                "start": [-2.0, -1.0, 0.1],
                "end": [4.0, -3.0, 0.1],
                "speed": 5,
                "start_time": 2.0,
                "radius": 0.5,
            },
        ],
    },
    "train_5": {
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
                "start": [-2.0, -1.0, 0.1],
                "end": [4.0, -3.0, 0.1],
                "speed": 5,
                "start_time": 2.0,
                "radius": 0.5,
            },
        ],
    },
}


TRAINING_SCENARIOS = ["train_1", "train_2", "train_3", "train_4", "train_5"]


def get_scenario(name: str):
    return SCENARIOS[name]


def get_random_training_scenario():
    name = random.choice(TRAINING_SCENARIOS)
    return name, SCENARIOS[name]
