import numpy as np
from config import *
from controllers import clip_norm

def run_one(i):
    s=np.r_[STARTS[i], INITIAL_VELOCITY[i]].astype(float)
    for step in range(MAX_STEPS):
        p=s[:2]; v=s[2:]
        g=GOALS[i]-p; vgoal=VREF*g/(np.linalg.norm(g)+1e-9)
        vobs=np.zeros(2)
        for ox,oy,r in OBSTACLES:
            c=np.array([ox,oy]); away=p-c; dc=np.linalg.norm(away)+1e-9; rho=dc-(r+ROBOT_RADIUS)
            if rho < OBSTACLE_INFLUENCE:
                rho=max(rho,0.08)
                mag=K_OBSTACLE*(1/rho-1/OBSTACLE_INFLUENCE)/(rho**2)
                mag=min(mag,MAX_OBSTACLE_EFFECT)
                vobs += mag*away/dc
        vdes=clip_norm(vgoal+vobs,VMAX)
        a=clip_norm((vdes-v)/DT,AMAX)
        s[:2]=p+DT*v; s[2:]=clip_norm(v+DT*a,VMAX)
        if np.linalg.norm(s[:2]-GOALS[i]) < 0.35:
            return (step+1)*DT
    return None

if __name__=='__main__':
    for i in range(NUM_UAV):
        print(f'Agent {i+1} alone: {run_one(i)} s')
