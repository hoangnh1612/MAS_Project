from pathlib import Path

import numpy as np

from config import (
    DT,
    GOALS,
    INITIAL_VELOCITY,
    MAX_STEPS,
    NUM_UAV,
    OBSTACLES,
    ROBOT_RADIUS,
    STARTS,
    VMAX,
)
from controllers import CentralizedNMPC, PotentialFieldController, clip_norm

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)


def init():
    return np.hstack([STARTS.copy(), INITIAL_VELOCITY.copy()])


def integ(s, u):
    n = s.copy()
    n[:, :2] = s[:, :2] + DT * s[:, 2:]
    n[:, 2:] = s[:, 2:] + DT * u
    for i in range(NUM_UAV):
        n[i, 2:] = clip_norm(n[i, 2:], VMAX)
    return n


def clear(s):
    return min(
        np.linalg.norm(pi - np.array([ox, oy])) - r - ROBOT_RADIUS
        for pi in s[:, :2]
        for ox, oy, r in OBSTACLES
    )


def pair(s):
    return np.linalg.norm(s[0, :2] - s[1, :2])


def reached(s):
    return np.all(np.linalg.norm(s[:, :2] - GOALS, axis=1) < 0.35)


def run(name, c):
    s = init()
    h = [s.copy()]
    us = []
    cts = []
    col = False
    fail = False
    for q in range(MAX_STEPS):
        try:
            u, ct = c.compute(s)
        except RuntimeError as e:
            print(name, "solver fail", q, e)
            fail = True
            break
        s = integ(s, u)
        h.append(s.copy())
        us.append(u.copy())
        cts.append(ct)
        if pair(s) < 2 * ROBOT_RADIUS or clear(s) < 0:
            print(name, "collision", q * DT)
            col = True
            break
        if reached(s):
            print(name, "mission", (q + 1) * DT)
            break
    np.savez(
        OUT / f"{name}.npz",
        states=np.asarray(h),
        controls=np.asarray(us),
        compute_times=np.asarray(cts),
        collision=int(col),
        solver_failed=int(fail),
    )


if __name__ == "__main__":
    print("PF")
    run("pf", PotentialFieldController())
    print("NMPC")
    run("nmpc", CentralizedNMPC())
