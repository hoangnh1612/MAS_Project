# Case 2 — Single-lane right-of-way

Two agents have opposite goals and must traverse the same single-lane gate.

Key control check: with the other agent removed, the same PF controller crosses the gate successfully (about 10.5–11 s). With both agents present, local reciprocal repulsion prevents either from resolving right-of-way in the simulation horizon. This makes the failure a multi-agent interaction issue, not simply a blocked gate.

Run:
```bash
python3 simulate.py
python3 plot_results.py
```
