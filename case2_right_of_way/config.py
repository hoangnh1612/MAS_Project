import numpy as np

NUM_UAV = 2
DT = 0.15
MAX_STEPS = 160
ROBOT_RADIUS = 0.12
D_SAFE = 0.34
VREF = 0.70
VMAX = 0.90
AMAX = 1.10
Y_MIN = -2.2
Y_MAX = 2.2
GATE_X = 4.5
STARTS = np.array([[1.65, 0.0], [7.55, 0.0]], dtype=float)
GOALS = np.array([[8.6, 0.0], [0.4, 0.0]], dtype=float)
INITIAL_VELOCITY = np.array([[0.10, 0.0], [-0.10, 0.0]], dtype=float)
OBSTACLES = np.array(
    [
        [GATE_X, -1.55, 0.48],
        [GATE_X, -0.82, 0.48],
        [GATE_X, 0.82, 0.48],
        [GATE_X, 1.55, 0.48],
    ],
    dtype=float,
)
OBS_MARGIN = 0.06
# PF
K_AGENT_REP = 0.55
AGENT_INFLUENCE = 0.95
K_OBSTACLE = 0.08
OBSTACLE_INFLUENCE = 0.35
MAX_OBSTACLE_EFFECT = 1.7
BOUNDARY_INFLUENCE = 0.30
K_BOUNDARY = 0.16
# NMPC
HORIZON = 22
W_SPEED = 2.0
W_DIR = 2.5
W_PROGRESS = 0.22
W_U = 0.08
W_TERMINAL = 1.0
VIDEO_FPS = 15
