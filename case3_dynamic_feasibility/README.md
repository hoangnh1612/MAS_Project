# Case 3 — High speed + limited acceleration

Five agents fly at high preferred speed toward a large obstacle. The PF controller first computes a desired velocity, then converts it to an acceleration request and clips that request to `AMAX`. Because obstacle repulsion becomes dominant late, the raw PF acceleration demand can become far larger than the robot can execute.

The NMPC model includes `AMAX` inside the prediction problem, so every candidate trajectory is dynamically executable by construction.

Run:
```bash
python3 simulate.py
python3 plot_results.py
```
