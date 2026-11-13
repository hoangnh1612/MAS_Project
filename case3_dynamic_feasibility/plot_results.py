import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation

from config import (
    AMAX,
    DREF,
    DT,
    NEIGHBOR_NUM,
    NUM_UAV,
    OBS_MARGIN,
    OBSTACLE_INFLUENCE,
    OBSTACLES,
    ROBOT_RADIUS,
    VIDEO_FPS,
    VREF,
    X_GOAL,
    Y_MAX,
    Y_MIN,
)

OUT = Path(__file__).resolve().parent / "results"


def load(n):
    return np.load(OUT / f"{n}.npz")


def topo(p):
    out = []
    for i in range(NUM_UAV):
        d = np.linalg.norm(p - p[i], axis=1)
        out.extend(d[np.argsort(d)[1 : 1 + NEIGHBOR_NUM]])
    return np.asarray(out)


def metrics(d):
    s = d["states"]
    u = d["controls"]
    raw = d["raw_accel"]
    c = d["compute_times"]
    t = np.arange(len(s)) * DT
    sp = np.linalg.norm(s[:, :, 2:], axis=2)
    actual = np.linalg.norm(u, axis=2)
    rawn = np.linalg.norm(raw, axis=2)
    clear = []
    cy = []
    serr = []
    minpair = np.inf
    for S in s:
        p = S[:, :2]
        ds = topo(p)
        serr.append(np.mean(np.abs(ds - DREF)))
        cy.append(np.mean(p[:, 1]))
        clear.append(
            min(
                np.linalg.norm(pi - np.array([ox, oy])) - r - ROBOT_RADIUS
                for pi in p
                for ox, oy, r in OBSTACLES
            )
        )
        for i in range(NUM_UAV):
            for j in range(i + 1, NUM_UAV):
                minpair = min(minpair, np.linalg.norm(p[i] - p[j]))
    return {
        "times": t,
        "speeds": sp,
        "actual_accel": actual,
        "raw_accel": rawn,
        "clearance": np.asarray(clear),
        "centroid_y": np.asarray(cy),
        "mission_time": float(t[-1]),
        "reached": bool(np.all(s[-1, :, 0] >= X_GOAL)),
        "collision": bool(d["collision"]),
        "solver_failed": bool(d["solver_failed"]),
        "mean_speed": float(sp.mean()),
        "speed_error": float(np.mean(np.abs(sp - VREF))),
        "spacing_error": float(np.mean(serr)),
        "min_clearance": float(np.min(clear)),
        "min_pair": float(minpair),
        "max_raw_accel": float(rawn.max()) if len(rawn) else np.nan,
        "max_actual_accel": float(actual.max()) if len(actual) else np.nan,
        "avg_compute_ms": float(c.mean() * 1000) if len(c) else np.nan,
        "max_compute_ms": float(c.max() * 1000) if len(c) else np.nan,
    }


def env(ax, infl=False):
    for ox, oy, r in OBSTACLES:
        ax.add_patch(plt.Circle((ox, oy), r, alpha=0.35))
        ax.add_patch(
            plt.Circle(
                (ox, oy), r + ROBOT_RADIUS + OBS_MARGIN, fill=False, ls="--", alpha=0.35
            )
        )
        if infl:
            ax.add_patch(
                plt.Circle(
                    (ox, oy),
                    r + ROBOT_RADIUS + OBSTACLE_INFLUENCE,
                    fill=False,
                    ls=":",
                    alpha=0.15,
                )
            )
    ax.axhline(Y_MIN, lw=1.2)
    ax.axhline(Y_MAX, lw=1.2)
    ax.axvline(X_GOAL, ls="--", alpha=0.6)
    ax.set_xlim(-0.4, X_GOAL + 0.7)
    ax.set_ylim(Y_MIN - 0.15, Y_MAX + 0.15)
    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")


def paths(pf, nm):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    for ax, S, title, inf in zip(
        axs, [pf, nm], ["Reactive Potential Field", "Centralized NMPC"], [True, False]
    ):
        env(ax, inf)
        for i in range(NUM_UAV):
            ax.plot(S[:, i, 0], S[:, i, 1], lw=1.6)
            ax.scatter(S[0, i, 0], S[0, i, 1], s=28)
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(OUT / "case3_path_comparison.png", dpi=220)
    plt.close(fig)


def speed(pm, nmm):
    fig, ax = plt.subplots(figsize=(8, 4))
    for m, label in [(pm, "PF"), (nmm, "NMPC")]:
        a = m["speeds"].mean(axis=1)
        lo = m["speeds"].min(axis=1)
        hi = m["speeds"].max(axis=1)
        ax.plot(m["times"], a, label=label)
        ax.fill_between(m["times"], lo, hi, alpha=0.14)
    ax.axhline(VREF, ls="--", label="v_ref")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Speed [m/s]")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "case3_speed.png", dpi=220)
    plt.close(fig)


def accel(pm, nmm):
    fig, ax = plt.subplots(figsize=(8, 4))
    # PF raw demand and actual clipped acceleration
    if len(pm["raw_accel"]):
        ax.plot(pm["times"][:-1], pm["raw_accel"].max(axis=1), label="PF raw demand")
        ax.plot(
            pm["times"][:-1],
            pm["actual_accel"].max(axis=1),
            label="PF actual (clipped)",
        )
    if len(nmm["actual_accel"]):
        ax.plot(nmm["times"][:-1], nmm["actual_accel"].max(axis=1), label="NMPC actual")
    ax.axhline(AMAX, ls="--", label="a_max")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Max acceleration norm [m/s²]")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "case3_acceleration.png", dpi=220)
    plt.close(fig)


def clearance(pm, nmm):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(pm["times"], pm["clearance"], label="PF")
    ax.plot(nmm["times"], nmm["clearance"], label="NMPC")
    ax.axhline(0, ls="--", label="Physical collision boundary")
    ax.axhline(OBS_MARGIN, ls=":", label="NMPC safety margin")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Minimum obstacle clearance [m]")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "case3_clearance.png", dpi=220)
    plt.close(fig)


def centroid(pm, nmm):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(pm["times"], pm["centroid_y"], label="PF")
    ax.plot(nmm["times"], nmm["centroid_y"], label="NMPC")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Swarm centroid y [m]")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "case3_centroid_y.png", dpi=220)
    plt.close(fig)


def summary(pf, nm, pm, nmm):
    fig = plt.figure(figsize=(12, 7))
    a1 = fig.add_axes([0.06, 0.53, 0.42, 0.40])
    a2 = fig.add_axes([0.53, 0.53, 0.42, 0.40])
    for ax, S, title, inf in [(a1, pf, "Reactive PF", True), (a2, nm, "NMPC", False)]:
        env(ax, inf)
        for i in range(NUM_UAV):
            ax.plot(S[:, i, 0], S[:, i, 1], lw=1.3)
        ax.set_title(title)
    a3 = fig.add_axes([0.08, 0.10, 0.38, 0.30])
    a3.plot(pm["times"][:-1], pm["raw_accel"].max(axis=1), label="PF raw demand")
    a3.plot(pm["times"][:-1], pm["actual_accel"].max(axis=1), label="PF clipped")
    a3.plot(nmm["times"][:-1], nmm["actual_accel"].max(axis=1), label="NMPC")
    a3.axhline(AMAX, ls="--", label="a_max")
    a3.set_title("Acceleration demand vs actuator limit")
    a3.set_xlabel("Time [s]")
    a3.set_ylabel("m/s²")
    a3.grid(True)
    a3.legend(fontsize=8)
    a4 = fig.add_axes([0.56, 0.10, 0.38, 0.30])
    a4.plot(pm["times"], pm["clearance"], label="PF")
    a4.plot(nmm["times"], nmm["clearance"], label="NMPC")
    a4.axhline(0, ls="--", label="collision")
    a4.axhline(OBS_MARGIN, ls=":", label="safety margin")
    a4.set_title("Minimum obstacle clearance")
    a4.set_xlabel("Time [s]")
    a4.set_ylabel("m")
    a4.grid(True)
    a4.legend(fontsize=8)
    fig.savefig(OUT / "case3_slide_summary.png", dpi=220)
    plt.close(fig)


def video(pf, nm):
    nf = max(len(pf), len(nm))
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    sc = []
    tr = []
    for ax, title, inf in zip(
        axs, ["Reactive Potential Field", "Centralized NMPC"], [True, False]
    ):
        env(ax, inf)
        ax.set_title(title)
        sc.append(ax.scatter([], [], s=80, zorder=5))
        tr.append([ax.plot([], [], lw=1.2, alpha=0.82)[0] for _ in range(NUM_UAV)])
    tt = fig.text(0.5, 0.02, "", ha="center")

    def upd(k):
        arts = [tt]
        for side, arr in enumerate([pf, nm]):
            kk = min(k, len(arr) - 1)
            S = arr[kk]
            sc[side].set_offsets(S[:, :2])
            arts.append(sc[side])
            for i in range(NUM_UAV):
                tr[side][i].set_data(arr[: kk + 1, i, 0], arr[: kk + 1, i, 1])
                arts.append(tr[side][i])
        tt.set_text(f"t = {k * DT:.2f} s")
        return arts

    a = FuncAnimation(
        fig, upd, frames=nf, interval=1000 / VIDEO_FPS, blit=False, repeat=False
    )
    a.save(
        OUT / "case3_pf_vs_nmpc.mp4", writer=FFMpegWriter(fps=VIDEO_FPS, bitrate=2500)
    )
    plt.close(fig)


if __name__ == "__main__":
    pd, nd = load("pf"), load("nmpc")
    pf, nm = pd["states"], nd["states"]
    pm, nmm = metrics(pd), metrics(nd)
    paths(pf, nm)
    speed(pm, nmm)
    accel(pm, nmm)
    clearance(pm, nmm)
    centroid(pm, nmm)
    summary(pf, nm, pm, nmm)
    video(pf, nm)
    keys = [
        "mission_time",
        "reached",
        "collision",
        "solver_failed",
        "mean_speed",
        "speed_error",
        "spacing_error",
        "min_clearance",
        "min_pair",
        "max_raw_accel",
        "max_actual_accel",
        "avg_compute_ms",
        "max_compute_ms",
    ]
    with open(OUT / "case3_metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["controller"] + keys)
        w.writerow(["PF"] + [pm[k] for k in keys])
        w.writerow(["NMPC"] + [nmm[k] for k in keys])
    print("PF", {k: pm[k] for k in keys})
    print("NMPC", {k: nmm[k] for k in keys})
