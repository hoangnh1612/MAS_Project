import numpy as np

NUM_UAV = 5
DT = 0.15
MAX_STEPS = 140
NEIGHBOR_NUM = 2
ROBOT_RADIUS = 0.12
D_SAFE = 0.28
DREF = 0.55
VREF = 1.25
VMAX = 1.35
AMAX = 0.55
UREF = np.array([1.0, 0.0])
X_GOAL = 9.0
Y_MIN = -2.7
Y_MAX = 2.7
STARTS = np.array([[0, -1.0], [0, -0.5], [0, 0.0], [0, 0.5], [0, 1.0]], dtype=float)
INITIAL_VELOCITY = np.array([1.25, 0.0])
# Single large obstacle slightly below swarm center. Shortest bypass is above.
OBSTACLES = np.array([[4.60, -0.35, 0.88]], dtype=float)
OBS_MARGIN = 0.08
# PF
K_COHESION = 0.22
K_REPULSION = 0.72
K_FRICTION = 0.65
K_OBSTACLE = 0.28
OBSTACLE_INFLUENCE = 0.72
MAX_OBSTACLE_EFFECT = 2.5
BOUNDARY_INFLUENCE = 0.35
K_BOUNDARY = 0.15
# NMPC
HORIZON = 18
W_SEP = 2.0
W_PROP = 3.0
W_DIR = 1.6
W_U = 0.08
W_PROGRESS = 0.10
W_TERMINAL = 0.7
VIDEO_FPS = 15
