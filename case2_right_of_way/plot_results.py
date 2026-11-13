import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation

from config import (
    D_SAFE,
    DT,
    GATE_X,
    GOALS,
    NUM_UAV,
    OBS_MARGIN,
    OBSTACLE_INFLUENCE,
    OBSTACLES,
    ROBOT_RADIUS,
    STARTS,
    VIDEO_FPS,
    VREF,
    Y_MAX,
    Y_MIN,
)

OUT = Path(__file__).resolve().parent / "results"


def load(n):
    return np.load(OUT / f"{n}.npz")


def metrics(d):
    s = d["states"]
    c = d["compute_times"]
    t = np.arange(len(s)) * DT
    sp = np.linalg.norm(s[:, :, 2:], axis=2)
    pair = np.linalg.norm(s[:, 0, :2] - s[:, 1, :2], axis=1)
    progress = np.zeros((len(s), 2))
    total = np.linalg.norm(GOALS - STARTS, axis=1)
    for k, S in enumerate(s):
        progress[k] = 1.0 - np.linalg.norm(S[:, :2] - GOALS, axis=1) / total
    clearance = []
    for S in s:
        clearance.append(
            min(
                np.linalg.norm(pi - np.array([ox, oy])) - r - ROBOT_RADIUS
                for pi in S[:, :2]
                for ox, oy, r in OBSTACLES
            )
        )
    cross = []
    for i in range(NUM_UAV):
        if i == 0:
            idx = np.where(s[:, i, 0] >= GATE_X)[0]
        else:
            idx = np.where(s[:, i, 0] <= GATE_X)[0]
        cross.append(float(idx[0] * DT) if len(idx) else np.nan)
    reached = bool(np.all(np.linalg.norm(s[-1, :, :2] - GOALS, axis=1) < 0.35))
    return {
        "times": t,
        "speeds": sp,
        "pair": pair,
        "progress": progress,
        "clearance": np.asarray(clearance),
        "cross": np.asarray(cross),
        "mission_time": float(t[-1]),
        "reached": reached,
        "collision": bool(d["collision"]),
        "solver_failed": bool(d["solver_failed"]),
        "mean_speed": float(sp.mean()),
        "min_pair": float(pair.min()),
        "min_clearance": float(np.min(clearance)),
        "avg_compute_ms": float(c.mean() * 1000) if len(c) else np.nan,
        "max_compute_ms": float(c.max() * 1000) if len(c) else np.nan,
    }


def env(ax, influence=False):
    for ox, oy, r in OBSTACLES:
        ax.add_patch(plt.Circle((ox, oy), r, alpha=0.35))
        ax.add_patch(
            plt.Circle(
                (ox, oy), r + ROBOT_RADIUS + OBS_MARGIN, fill=False, ls="--", alpha=0.35
            )
        )
        if influence:
            ax.add_patch(
                plt.Circle(
                    (ox, oy),
                    r + ROBOT_RADIUS + OBSTACLE_INFLUENCE,
                    fill=False,
                    ls=":",
                    alpha=0.14,
                )
            )
    ax.axhline(Y_MIN, lw=1.2)
    ax.axhline(Y_MAX, lw=1.2)
    ax.axvline(GATE_X, ls="--", alpha=0.35)
    ax.set_xlim(0, 9)
    ax.set_ylim(Y_MIN - 0.1, Y_MAX + 0.1)
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
            ax.plot(S[:, i, 0], S[:, i, 1], lw=1.7)
            ax.scatter(S[0, i, 0], S[0, i, 1], s=34)
            ax.scatter(GOALS[i, 0], GOALS[i, 1], marker="x", s=55)
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(OUT / "case2_path_comparison.png", dpi=220)
    plt.close(fig)


def speed(pm, nm):
    fig, ax = plt.subplots(figsize=(8, 4))
    for m, label in [(pm, "PF"), (nm, "NMPC")]:
        avg = m["speeds"].mean(axis=1)
        lo = m["speeds"].min(axis=1)
        hi = m["speeds"].max(axis=1)
        ax.plot(m["times"], avg, label=label)
        ax.fill_between(m["times"], lo, hi, alpha=0.14)
    ax.axhline(VREF, ls="--", label="v_ref")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Speed [m/s]")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "case2_speed.png", dpi=220)
    plt.close(fig)


def progress(pm, nm):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for ax, m, title in zip(
        axs, [pm, nm], ["Reactive Potential Field", "Centralized NMPC"]
    ):
        for i in range(NUM_UAV):
            ax.plot(m["times"], 100 * m["progress"][:, i], label=f"Agent {i + 1}")
        ax.set_ylim(-5, 105)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Goal progress [%]")
        ax.grid(True)
        ax.legend()
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(OUT / "case2_progress.png", dpi=220)
    plt.close(fig)


def pairplot(pm, nm):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(pm["times"], pm["pair"], label="PF")
    ax.plot(nm["times"], nm["pair"], label="NMPC")
    ax.axhline(D_SAFE, ls=":", label="d_safe")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Inter-agent distance [m]")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "case2_pair_distance.png", dpi=220)
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
        sc.append(ax.scatter([], [], s=85, zorder=5))
        tr.append([ax.plot([], [], lw=1.3, alpha=0.82)[0] for _ in range(NUM_UAV)])
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
        OUT / "case2_pf_vs_nmpc.mp4", writer=FFMpegWriter(fps=VIDEO_FPS, bitrate=2500)
    )
    plt.close(fig)


def summary(pf, nm, pm, nmm):
    fig = plt.figure(figsize=(12, 7))
    ax1 = fig.add_axes([0.06, 0.53, 0.42, 0.40])
    ax2 = fig.add_axes([0.53, 0.53, 0.42, 0.40])
    for ax, S, title, inf in [(ax1, pf, "Reactive PF", True), (ax2, nm, "NMPC", False)]:
        env(ax, inf)
        for i in range(NUM_UAV):
            ax.plot(S[:, i, 0], S[:, i, 1], lw=1.5)
        ax.set_title(title)
    ax3 = fig.add_axes([0.08, 0.10, 0.38, 0.30])
    ax3.plot(pm["times"], pm["speeds"].mean(axis=1), label="PF")
    ax3.plot(nmm["times"], nmm["speeds"].mean(axis=1), label="NMPC")
    ax3.axhline(VREF, ls="--")
    ax3.set_title("Average agent speed")
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("m/s")
    ax3.grid(True)
    ax3.legend()
    ax4 = fig.add_axes([0.56, 0.10, 0.38, 0.30])
    ax4.plot(pm["times"], 100 * pm["progress"][:, 0], label="PF agent 1")
    ax4.plot(pm["times"], 100 * pm["progress"][:, 1], label="PF agent 2")
    ax4.plot(nmm["times"], 100 * nmm["progress"][:, 0], ls="--", label="NMPC agent 1")
    ax4.plot(nmm["times"], 100 * nmm["progress"][:, 1], ls="--", label="NMPC agent 2")
    ax4.set_title("Right-of-way resolution")
    ax4.set_xlabel("Time [s]")
    ax4.set_ylabel("Goal progress [%]")
    ax4.grid(True)
    ax4.legend(fontsize=8)
    fig.savefig(OUT / "case2_slide_summary.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    pd, nd = load("pf"), load("nmpc")
    pf, nm = pd["states"], nd["states"]
    pm, nmm = metrics(pd), metrics(nd)
    paths(pf, nm)
    speed(pm, nmm)
    progress(pm, nmm)
    pairplot(pm, nmm)
    summary(pf, nm, pm, nmm)
    video(pf, nm)
    keys = [
        "mission_time",
        "reached",
        "collision",
        "solver_failed",
        "mean_speed",
        "min_pair",
        "min_clearance",
        "avg_compute_ms",
        "max_compute_ms",
    ]
    with open(OUT / "case2_metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["controller"] + keys + ["gate_cross_agent1", "gate_cross_agent2"])
        w.writerow(["PF"] + [pm[k] for k in keys] + list(pm["cross"]))
        w.writerow(["NMPC"] + [nmm[k] for k in keys] + list(nmm["cross"]))
    print("PF", {k: pm[k] for k in keys}, "cross", pm["cross"])
    print("NMPC", {k: nmm[k] for k in keys}, "cross", nmm["cross"])
