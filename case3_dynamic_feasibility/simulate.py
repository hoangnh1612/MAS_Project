from pathlib import Path
import numpy as np
from config import *
from controllers import *
OUT=Path(__file__).resolve().parent/'results';OUT.mkdir(exist_ok=True)
def init():return np.hstack([STARTS.copy(),np.tile(INITIAL_VELOCITY,(NUM_UAV,1))])
def integ(s,u):
 n=s.copy();n[:,:2]=s[:,:2]+DT*s[:,2:];n[:,2:]=s[:,2:]+DT*u
 for i in range(NUM_UAV):n[i,2:]=clip_norm(n[i,2:],VMAX)
 return n
def pair(s):
 p=s[:,:2];return min(np.linalg.norm(p[i]-p[j]) for i in range(NUM_UAV) for j in range(i+1,NUM_UAV))
def clear(s):return min(np.linalg.norm(pi-np.array([ox,oy]))-r-ROBOT_RADIUS for pi in s[:,:2] for ox,oy,r in OBSTACLES)
def run(name,c):
 s=init();h=[s.copy()];us=[];raw=[];cts=[];col=False;fail=False
 for q in range(MAX_STEPS):
  try:u,ct=c.compute(s)
  except RuntimeError as e:print(name,'solver fail',q,e);fail=True;break
  if hasattr(c,'last_raw_accel'):raw.append(c.last_raw_accel.copy())
  else:raw.append(u.copy())
  s=integ(s,u);h.append(s.copy());us.append(u.copy());cts.append(ct)
  if pair(s)<2*ROBOT_RADIUS or clear(s)<0:print(name,'collision', (q+1)*DT,'clear',clear(s));col=True;break
  if np.all(s[:,0]>=X_GOAL):print(name,'mission',(q+1)*DT);break
 np.savez(OUT/f'{name}.npz',states=np.asarray(h),controls=np.asarray(us),raw_accel=np.asarray(raw),compute_times=np.asarray(cts),collision=int(col),solver_failed=int(fail))
if __name__=='__main__':print('PF');run('pf',PotentialFieldController());print('NMPC');run('nmpc',CentralizedNMPC())
