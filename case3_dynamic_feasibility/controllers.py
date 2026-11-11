import time, numpy as np, casadi as ca
from config import *

def clip_norm(v,m):
 n=np.linalg.norm(v);return np.asarray(v,float) if n<1e-12 or n<=m else np.asarray(v,float)/n*m

def nearest(p,i):
 d=np.linalg.norm(p-p[i],axis=1);return np.argsort(d)[1:1+NEIGHBOR_NUM]

def nmask(p):
 m=np.zeros((NUM_UAV,NUM_UAV))
 for i in range(NUM_UAV):
  for j in nearest(p,i):m[i,j]=m[j,i]=1
 return m

class PotentialFieldController:
 def __init__(self): self.last_raw_accel=np.zeros((NUM_UAV,2));self.last_vdes=np.zeros((NUM_UAV,2))
 def compute(self,s):
  t0=time.perf_counter();p=s[:,:2];v=s[:,2:];u=np.zeros((NUM_UAV,2));raws=np.zeros_like(u);vds=np.zeros_like(u)
  for i in range(NUM_UAV):
   neigh=nearest(p,i); vmig=VREF*UREF; vs=np.zeros(2);mn=np.zeros(2)
   for j in neigh:
    rel=p[j]-p[i];d=np.linalg.norm(rel)+1e-9;n=rel/d;mn+=v[j]
    if d>DREF:vs+=K_COHESION*(d-DREF)*n
    else:vs+=-K_REPULSION*(DREF-d)/max(d,.12)*n
   mn/=len(neigh);vf=K_FRICTION*(mn-v[i])
   vo=np.zeros(2)
   for ox,oy,r in OBSTACLES:
    c=np.array([ox,oy]);away=p[i]-c;dc=np.linalg.norm(away)+1e-9;rho=dc-(r+ROBOT_RADIUS)
    if rho<OBSTACLE_INFLUENCE:
     rho=max(rho,.06);mag=K_OBSTACLE*(1/rho-1/OBSTACLE_INFLUENCE)/(rho**2);mag=min(mag,MAX_OBSTACLE_EFFECT);vo+=mag*away/dc
   vb=np.zeros(2);dl=p[i,1]-(Y_MIN+ROBOT_RADIUS);dh=(Y_MAX-ROBOT_RADIUS)-p[i,1]
   if dl<BOUNDARY_INFLUENCE:
    d=max(dl,.05);vb[1]+=K_BOUNDARY*(1/d-1/BOUNDARY_INFLUENCE)
   if dh<BOUNDARY_INFLUENCE:
    d=max(dh,.05);vb[1]-=K_BOUNDARY*(1/d-1/BOUNDARY_INFLUENCE)
   vd=clip_norm(vmig+vs+vf+vo+vb,VMAX);raw=(vd-v[i])/DT;u[i]=clip_norm(raw,AMAX);raws[i]=raw;vds[i]=vd
  self.last_raw_accel=raws;self.last_vdes=vds
  return u,time.perf_counter()-t0

class CentralizedNMPC:
 def __init__(self):self.nx=4;self.nu=2;self.prev_X=None;self.prev_U=None;self._build()
 def xs(self,i):return slice(4*i,4*i+4)
 def us(self,i):return slice(2*i,2*i+2)
 def _build(self):
  o=ca.Opti();self.o=o;X=o.variable(NUM_UAV*4,HORIZON+1);U=o.variable(NUM_UAV*2,HORIZON);X0=o.parameter(NUM_UAV*4);M=o.parameter(NUM_UAV,NUM_UAV);self.X=X;self.U=U;self.X0=X0;self.M=M;o.subject_to(X[:,0]==X0);J=0.;eps=1e-6
  for k in range(HORIZON):
   for i in range(NUM_UAV):
    x=X[self.xs(i),k];xn=X[self.xs(i),k+1];u=U[self.us(i),k];o.subject_to(xn==ca.vertcat(x[0]+DT*x[2],x[1]+DT*x[3],x[2]+DT*u[0],x[3]+DT*u[1]))
    sp2=xn[2]**2+xn[3]**2;sp=ca.sqrt(sp2+eps);align=xn[2]/sp
    J+=W_PROP*(sp2-VREF**2)**2+W_DIR*(1-align)**2+W_U*(u[0]**2+u[1]**2)+W_PROGRESS*(X_GOAL-xn[0])**2
    o.subject_to(sp2<=VMAX**2);o.subject_to(u[0]**2+u[1]**2<=AMAX**2);o.subject_to(xn[1]>=Y_MIN+ROBOT_RADIUS);o.subject_to(xn[1]<=Y_MAX-ROBOT_RADIUS)
    for ox,oy,r in OBSTACLES:
     rs=r+ROBOT_RADIUS+OBS_MARGIN;o.subject_to((xn[0]-ox)**2+(xn[1]-oy)**2>=rs**2)
   for i in range(NUM_UAV):
    xi=X[self.xs(i),k+1]
    for j in range(i+1,NUM_UAV):
     xj=X[self.xs(j),k+1];d2=(xi[0]-xj[0])**2+(xi[1]-xj[1])**2;o.subject_to(d2>=D_SAFE**2);J+=W_SEP*M[i,j]*(d2-DREF**2)**2
  for i in range(NUM_UAV):J+=W_TERMINAL*(X_GOAL-X[self.xs(i),HORIZON][0])**2
  o.minimize(J);o.solver('ipopt',{'expand':True,'print_time':False},{'print_level':0,'max_iter':300,'tol':1e-4,'acceptable_tol':2e-3,'sb':'yes'})
 def guess(self,s):
  Xg=np.zeros((NUM_UAV*4,HORIZON+1));Ug=np.zeros((NUM_UAV*2,HORIZON))
  if self.prev_X is not None:
   Xg[:,:-1]=self.prev_X[:,1:];Xg[:,-1]=self.prev_X[:,-1];Xg[:,0]=s.reshape(-1);Ug[:,:-1]=self.prev_U[:,1:];Ug[:,-1]=self.prev_U[:,-1];return Xg,Ug
  # Numerical warm-start only: small smooth upward bend around the below-center obstacle.
  for k in range(HORIZON+1):
   t=k*DT;g=s.copy();g[:,0]+=VREF*t;alpha=1/(1+np.exp(-2.0*(g[:,0]-2.4)));g[:,1]+=0.75*alpha;g[:,2]=VREF;g[:,3]=0;Xg[:,k]=g.reshape(-1)
  return Xg,Ug
 def compute(self,s):
  t0=time.perf_counter();self.o.set_value(self.X0,s.reshape(-1));self.o.set_value(self.M,nmask(s[:,:2]));Xg,Ug=self.guess(s);self.o.set_initial(self.X,Xg);self.o.set_initial(self.U,Ug);sol=self.o.solve();self.prev_X=sol.value(self.X);self.prev_U=sol.value(self.U);return self.prev_U[:,0].reshape(NUM_UAV,2),time.perf_counter()-t0
