import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation

from config import (
    D_SAFE,
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


def calc(d):
    s = d["states"]
    c = d["compute_times"]
    t = np.arange(len(s)) * DT
    speeds = np.linalg.norm(s[:, :, 2:], axis=2)
    sm = []
    slo = []
    shi = []
    serr = []
    order = []
    clear = []
    cy = []
    minpair = np.inf
    for S in s:
        p = S[:, :2]
        v = S[:, 2:]
        ds = topo(p)
        sm.append(ds.mean())
        slo.append(ds.min())
        shi.append(ds.max())
        serr.append(np.mean(np.abs(ds - DREF)))
        uv = np.array(
            [
                vi / (np.linalg.norm(vi) + 1e-12)
                if np.linalg.norm(vi) > 1e-8
                else np.zeros(2)
                for vi in v
            ]
        )
        order.append(np.linalg.norm(uv.mean(axis=0)))
        clear.append(
            min(
                np.linalg.norm(pi - np.array([ox, oy])) - r - ROBOT_RADIUS
                for pi in p
                for ox, oy, r in OBSTACLES
            )
        )
        cy.append(p[:, 1].mean())
        for i in range(NUM_UAV):
            for j in range(i + 1, NUM_UAV):
                minpair = min(minpair, np.linalg.norm(p[i] - p[j]))
    return {
        "times": t,
        "speeds": speeds,
        "spacing_mean": np.array(sm),
        "spacing_min": np.array(slo),
        "spacing_max": np.array(shi),
        "clearance": np.array(clear),
        "centroid_y": np.array(cy),
        "mission_time": float(t[-1]),
        "reached": bool(np.all(s[-1, :, 0] >= X_GOAL)),
        "collision": bool(d["collision"]),
        "solver_failed": bool(d["solver_failed"]),
        "mean_speed": float(speeds.mean()),
        "speed_error": float(np.mean(np.abs(speeds - VREF))),
        "spacing_error": float(np.mean(serr)),
        "mean_order": float(np.mean(order[1:])),
        "min_clearance": float(np.min(clear)),
        "min_pair_distance": float(minpair),
        "avg_compute_ms": float(np.mean(c) * 1000) if len(c) else np.nan,
        "max_compute_ms": float(np.max(c) * 1000) if len(c) else np.nan,
    }


def env(ax, infl=False):
    for ox, oy, r in OBSTACLES:
        ax.add_patch(plt.Circle((ox, oy), r, alpha=0.35))
        ax.add_patch(
            plt.Circle(
                (ox, oy),
                r + ROBOT_RADIUS + OBS_MARGIN,
                fill=False,
                linestyle="--",
                alpha=0.35,
            )
        )
        if infl:
            ax.add_patch(
                plt.Circle(
                    (ox, oy),
                    r + ROBOT_RADIUS + OBSTACLE_INFLUENCE,
                    fill=False,
                    linestyle=":",
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
    fig.savefig(OUT / "case1_path_comparison.png", dpi=220)
    plt.close(fig)


def lineplot(pfm, nmm, key, ylabel, file, href=None):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(pfm["times"], pfm[key], label="PF")
    ax.plot(nmm["times"], nmm[key], label="NMPC")
    if href is not None:
        ax.axhline(href, ls="--", label="reference")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / file, dpi=220)
    plt.close(fig)


def speed(pfm, nmm):
    fig, ax = plt.subplots(figsize=(8, 4))
    for m, label in [(pfm, "PF"), (nmm, "NMPC")]:
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
    fig.savefig(OUT / "case1_speed.png", dpi=220)
    plt.close(fig)


def spacing(pfm, nmm):
    fig, ax = plt.subplots(figsize=(8, 4))
    for m, label in [(pfm, "PF"), (nmm, "NMPC")]:
        ax.plot(m["times"], m["spacing_mean"], label=label)
        ax.fill_between(m["times"], m["spacing_min"], m["spacing_max"], alpha=0.14)
    ax.axhline(DREF, ls="--", label="d_ref")
    ax.axhline(D_SAFE, ls=":", label="d_safe")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Neighbor distance [m]")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "case1_spacing.png", dpi=220)
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
        OUT / "case1_pf_vs_nmpc.mp4", writer=FFMpegWriter(fps=VIDEO_FPS, bitrate=2500)
    )
    plt.close(fig)


def summary(pf, nm, pfm, nmm):
    fig = plt.figure(figsize=(12, 7))
    ax1 = fig.add_axes([0.06, 0.53, 0.42, 0.40])
    ax2 = fig.add_axes([0.53, 0.53, 0.42, 0.40])
    for ax, S, title, inf in [(ax1, pf, "Reactive PF", True), (ax2, nm, "NMPC", False)]:
        env(ax, inf)
        for i in range(NUM_UAV):
            ax.plot(S[:, i, 0], S[:, i, 1], lw=1.3)
        ax.set_title(title)
    ax3 = fig.add_axes([0.08, 0.10, 0.38, 0.30])
    ax3.plot(pfm["times"], pfm["speeds"].mean(axis=1), label="PF")
    ax3.plot(nmm["times"], nmm["speeds"].mean(axis=1), label="NMPC")
    ax3.axhline(VREF, ls="--")
    ax3.set_title("Average swarm speed")
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("m/s")
    ax3.grid(True)
    ax3.legend()
    ax4 = fig.add_axes([0.56, 0.10, 0.38, 0.30])
    ax4.plot(pfm["times"], pfm["centroid_y"], label="PF")
    ax4.plot(nmm["times"], nmm["centroid_y"], label="NMPC")
    ax4.axhline(0.50, ls="--", label="Gate center")
    ax4.set_title("Swarm lateral shift")
    ax4.set_xlabel("Time [s]")
    ax4.set_ylabel("Centroid y [m]")
    ax4.grid(True)
    ax4.legend()
    fig.savefig(OUT / "case1_slide_summary.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    pd = load("pf")
    nd = load("nmpc")
    pf = pd["states"]
    nm = nd["states"]
    pm = calc(pd)
    nmtr = calc(nd)
    paths(pf, nm)
    speed(pm, nmtr)
    spacing(pm, nmtr)
    lineplot(
        pm, nmtr, "centroid_y", "Swarm centroid y [m]", "case1_centroid_y.png", 0.50
    )
    lineplot(
        pm,
        nmtr,
        "clearance",
        "Minimum obstacle clearance [m]",
        "case1_clearance.png",
        0.0,
    )
    summary(pf, nm, pm, nmtr)
    video(pf, nm)
    keys = [
        "mission_time",
        "reached",
        "collision",
        "solver_failed",
        "mean_speed",
        "speed_error",
        "spacing_error",
        "mean_order",
        "min_clearance",
        "min_pair_distance",
        "avg_compute_ms",
        "max_compute_ms",
    ]
    with open(OUT / "case1_metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["controller"] + keys)
        w.writerow(["PF"] + [pm[k] for k in keys])
        w.writerow(["NMPC"] + [nmtr[k] for k in keys])
    print("PF", {k: pm[k] for k in keys})
    print("NMPC", {k: nmtr[k] for k in keys})
