import time

import casadi as ca
import numpy as np

from config import (
    AMAX,
    BOUNDARY_INFLUENCE,
    D_SAFE,
    DREF,
    DT,
    HORIZON,
    K_BOUNDARY,
    K_COHESION,
    K_FRICTION,
    K_OBSTACLE,
    K_REPULSION,
    MAX_OBSTACLE_EFFECT,
    NEIGHBOR_NUM,
    NUM_UAV,
    OBS_MARGIN,
    OBSTACLE_INFLUENCE,
    OBSTACLES,
    ROBOT_RADIUS,
    UREF,
    VMAX,
    VREF,
    W_DIR,
    W_PROGRESS,
    W_PROP,
    W_SEP,
    W_TERMINAL,
    W_U,
    X_GOAL,
    Y_MAX,
    Y_MIN,
)


def clip_norm(v, vmax):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v if n < 1e-12 or n <= vmax else v / n * vmax


def nearest_neighbors(positions, i):
    d = np.linalg.norm(positions - positions[i], axis=1)
    return np.argsort(d)[1 : 1 + NEIGHBOR_NUM]


def neighbor_mask(positions):
    mask = np.zeros((NUM_UAV, NUM_UAV))
    for i in range(NUM_UAV):
        for j in nearest_neighbors(positions, i):
            mask[i, j] = 1.0
            mask[j, i] = 1.0
    return mask


class PotentialFieldController:
    def compute(self, states):
        t0 = time.perf_counter()
        p = states[:, :2]
        v = states[:, 2:]
        commands = np.zeros((NUM_UAV, 2))
        for i in range(NUM_UAV):
            neigh = nearest_neighbors(p, i)
            v_mig = VREF * UREF
            v_social = np.zeros(2)
            mean_neighbor_v = np.zeros(2)
            for j in neigh:
                dp = p[j] - p[i]
                dij = np.linalg.norm(dp) + 1e-9
                nij = dp / dij
                mean_neighbor_v += v[j]
                if dij > DREF:
                    v_social += K_COHESION * (dij - DREF) * nij
                else:
                    v_social += -K_REPULSION * (DREF - dij) / max(dij, 0.12) * nij
            mean_neighbor_v /= len(neigh)
            v_fric = K_FRICTION * (mean_neighbor_v - v[i])
            v_obs = np.zeros(2)
            for ox, oy, r in OBSTACLES:
                center = np.array([ox, oy])
                away = p[i] - center
                dc = np.linalg.norm(away) + 1e-9
                rho = dc - (r + ROBOT_RADIUS)
                if rho < OBSTACLE_INFLUENCE:
                    rho = max(rho, 0.08)
                    mag = K_OBSTACLE * (1.0 / rho - 1.0 / OBSTACLE_INFLUENCE) / (rho**2)
                    mag = min(mag, MAX_OBSTACLE_EFFECT)
                    v_obs += mag * away / dc
            v_bound = np.zeros(2)
            dist_low = p[i, 1] - (Y_MIN + ROBOT_RADIUS)
            if dist_low < BOUNDARY_INFLUENCE:
                d = max(dist_low, 0.05)
                v_bound[1] += K_BOUNDARY * (1.0 / d - 1.0 / BOUNDARY_INFLUENCE)
            dist_high = (Y_MAX - ROBOT_RADIUS) - p[i, 1]
            if dist_high < BOUNDARY_INFLUENCE:
                d = max(dist_high, 0.05)
                v_bound[1] -= K_BOUNDARY * (1.0 / d - 1.0 / BOUNDARY_INFLUENCE)
            v_des = clip_norm(v_mig + v_social + v_fric + v_obs + v_bound, VMAX)
            commands[i] = clip_norm((v_des - v[i]) / DT, AMAX)
        return commands, time.perf_counter() - t0


class CentralizedNMPC:
    def __init__(self):
        self.nx = 4
        self.nu = 2
        self.prev_X = None
        self.prev_U = None
        self._build()

    def xs(self, i):
        return slice(i * self.nx, (i + 1) * self.nx)

    def us(self, i):
        return slice(i * self.nu, (i + 1) * self.nu)

    def _build(self):
        o = ca.Opti()
        self.opti = o
        self.X = o.variable(NUM_UAV * self.nx, HORIZON + 1)
        self.U = o.variable(NUM_UAV * self.nu, HORIZON)
        self.X0 = o.parameter(NUM_UAV * self.nx)
        self.NMASK = o.parameter(NUM_UAV, NUM_UAV)
        o.subject_to(self.X[:, 0] == self.X0)
        J = 0.0
        eps = 1e-5
        for k in range(HORIZON):
            for i in range(NUM_UAV):
                x = self.X[self.xs(i), k]
                xn = self.X[self.xs(i), k + 1]
                u = self.U[self.us(i), k]
                pred = ca.vertcat(
                    x[0] + DT * x[2],
                    x[1] + DT * x[3],
                    x[2] + DT * u[0],
                    x[3] + DT * u[1],
                )
                o.subject_to(xn == pred)
                speed2 = xn[2] ** 2 + xn[3] ** 2
                speed = ca.sqrt(speed2 + eps)
                J += W_PROP * (speed2 - VREF**2) ** 2
                J += W_DIR * (1.0 - xn[2] / speed) ** 2
                J += W_U * (u[0] ** 2 + u[1] ** 2)
                J += W_PROGRESS * (X_GOAL - xn[0]) ** 2
                o.subject_to(speed2 <= VMAX**2)
                o.subject_to(u[0] ** 2 + u[1] ** 2 <= AMAX**2)
                o.subject_to(xn[1] >= Y_MIN + ROBOT_RADIUS)
                o.subject_to(xn[1] <= Y_MAX - ROBOT_RADIUS)
                for ox, oy, r in OBSTACLES:
                    sr = r + ROBOT_RADIUS + OBS_MARGIN
                    d2 = (xn[0] - ox) ** 2 + (xn[1] - oy) ** 2
                    o.subject_to(d2 >= sr**2)
            for i in range(NUM_UAV):
                xi = self.X[self.xs(i), k + 1]
                for j in range(i + 1, NUM_UAV):
                    xj = self.X[self.xs(j), k + 1]
                    d2 = (xi[0] - xj[0]) ** 2 + (xi[1] - xj[1]) ** 2
                    o.subject_to(d2 >= D_SAFE**2)
                    J += W_SEP * self.NMASK[i, j] * (d2 - DREF**2) ** 2
        for i in range(NUM_UAV):
            xt = self.X[self.xs(i), HORIZON]
            J += W_TERMINAL * (X_GOAL - xt[0]) ** 2
        o.minimize(J)
        o.solver(
            "ipopt",
            {"expand": True, "print_time": False},
            {
                "print_level": 0,
                "max_iter": 300,
                "tol": 1e-4,
                "acceptable_tol": 2e-3,
                "sb": "yes",
            },
        )

    def _initial_guess(self, states):
        Xg = np.zeros((NUM_UAV * self.nx, HORIZON + 1))
        Ug = np.zeros((NUM_UAV * self.nu, HORIZON))
        if self.prev_X is not None:
            Xg[:, :-1] = self.prev_X[:, 1:]
            Xg[:, -1] = self.prev_X[:, -1]
            Xg[:, 0] = states.reshape(-1)
            Ug[:, :-1] = self.prev_U[:, 1:]
            Ug[:, -1] = self.prev_U[:, -1]
            return Xg, Ug
        # Neutral first guess: continue straight. No gate-center hint.
        for k in range(HORIZON + 1):
            t = k * DT
            g = states.copy()
            g[:, 0] += VREF * t
            g[:, 2] = VREF
            g[:, 3] = 0.0
            Xg[:, k] = g.reshape(-1)
        return Xg, Ug

    def compute(self, states):
        t0 = time.perf_counter()
        self.opti.set_value(self.X0, states.reshape(-1))
        self.opti.set_value(self.NMASK, neighbor_mask(states[:, :2]))
        Xg, Ug = self._initial_guess(states)
        self.opti.set_initial(self.X, Xg)
        self.opti.set_initial(self.U, Ug)
        sol = self.opti.solve()
        self.prev_X = sol.value(self.X)
        self.prev_U = sol.value(self.U)
        return self.prev_U[:, 0].reshape(NUM_UAV, self.nu), time.perf_counter() - t0
