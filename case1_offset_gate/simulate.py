from pathlib import Path
import numpy as np
from config import *
from controllers import PotentialFieldController,CentralizedNMPC,clip_norm
OUT=Path(__file__).resolve().parent/'results'; OUT.mkdir(exist_ok=True)

def initial_state(): return np.hstack([STARTS.copy(),np.tile(INITIAL_VELOCITY,(NUM_UAV,1))])
def integrate(s,u):
    n=s.copy(); n[:,:2]=s[:,:2]+DT*s[:,2:]; n[:,2:]=s[:,2:]+DT*u
    for i in range(NUM_UAV): n[i,2:]=clip_norm(n[i,2:],VMAX)
    return n

def min_pair(s):
    p=s[:,:2]; return min(np.linalg.norm(p[i]-p[j]) for i in range(NUM_UAV) for j in range(i+1,NUM_UAV))
def min_clear(s):
    p=s[:,:2]; return min(np.linalg.norm(pi-np.array([ox,oy]))-r-ROBOT_RADIUS for pi in p for ox,oy,r in OBSTACLES)

def run(name,ctrl):
    s=initial_state(); hist=[s.copy()]; comps=[]; ctrls=[]; collision=False; failed=False
    for step in range(MAX_STEPS):
        try: u,ct=ctrl.compute(s)
        except RuntimeError as e:
            failed=True; print(f'[{name}] solver failed at step {step}: {e}'); break
        s=integrate(s,u); hist.append(s.copy()); comps.append(ct); ctrls.append(u.copy())
        if min_pair(s)<2*ROBOT_RADIUS or min_clear(s)<0:
            collision=True; print(f'[{name}] collision at t={(step+1)*DT:.2f}s'); break
        if np.all(s[:,0]>=X_GOAL):
            print(f'[{name}] mission complete at t={(step+1)*DT:.2f}s'); break
    np.savez(OUT/f'{name}.npz',states=np.asarray(hist),controls=np.asarray(ctrls),compute_times=np.asarray(comps),collision=int(collision),solver_failed=int(failed))

if __name__=='__main__':
    print('=== Reactive PF ==='); run('pf',PotentialFieldController())
    print('\n=== Centralized NMPC ==='); run('nmpc',CentralizedNMPC())
