import numpy as np

NUM_UAV = 5
DT = 0.15
MAX_STEPS = 180
NEIGHBOR_NUM = 2
ROBOT_RADIUS = 0.12
D_SAFE = 0.28
DREF = 0.55
VREF = 0.80
VMAX = 1.05
AMAX = 1.40
UREF = np.array([1.0, 0.0])
X_GOAL = 7.0
Y_MIN = -2.55
Y_MAX = 2.75
STARTS = np.array(
    [[0.0, -1.00], [0.0, -0.50], [0.0, 0.00], [0.0, 0.50], [0.0, 1.00]], dtype=float
)
INITIAL_VELOCITY = np.array([0.10, 0.0])
OBSTACLES = np.array(
    [
        [4.80, -2.10, 0.42],
        [4.80, -1.40, 0.42],
        [4.80, -0.70, 0.42],
        [4.80, 1.70, 0.42],
        [4.80, 2.40, 0.42],
    ],
    dtype=float,
)
OBS_MARGIN = 0.06
K_COHESION = 0.24
K_REPULSION = 0.75
K_FRICTION = 0.70
K_OBSTACLE = 0.22
OBSTACLE_INFLUENCE = 0.78
MAX_OBSTACLE_EFFECT = 1.80
BOUNDARY_INFLUENCE = 0.35
K_BOUNDARY = 0.18
HORIZON = 16
W_SEP = 2.5
W_PROP = 3.0
W_DIR = 1.5
W_U = 0.05
W_PROGRESS = 0.12
W_TERMINAL = 0.8
VIDEO_FPS = 15
