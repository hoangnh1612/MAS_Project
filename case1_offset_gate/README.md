# Case 1 — Offset Gate: Reactive PF vs Centralized NMPC

Both controllers use the same 5-agent swarm, K=2 nearest neighbors, double-integrator dynamics, speed/acceleration limits, corridor and obstacle geometry.

- PF: migration + cohesion/repulsion + velocity damping + local obstacle repulsion.
- NMPC: centralized receding-horizon optimization with the same preferred speed/direction/spacing intentions, plus explicit dynamics and safety constraints.

The gate is vertically offset from the initial swarm centroid. The point of the experiment is to show that the reactive PF continues mostly straight until obstacle repulsion becomes significant, then shifts/compresses more abruptly. NMPC can distribute the lateral maneuver over its prediction horizon.

Run:
```bash
python3 -m pip install numpy matplotlib casadi
sudo apt install ffmpeg
python3 simulate.py
python3 plot_results.py
```
