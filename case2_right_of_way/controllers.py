import time, numpy as np, casadi as ca
from config import *

def clip_norm(v,m):
    n=np.linalg.norm(v)
    return np.asarray(v,float) if n<1e-12 or n<=m else np.asarray(v,float)/n*m

class PotentialFieldController:
    def compute(self,s):
        t0=time.perf_counter(); p=s[:,:2]; v=s[:,2:]; u=np.zeros((NUM_UAV,2))
        for i in range(NUM_UAV):
            g=GOALS[i]-p[i]; gd=np.linalg.norm(g)+1e-9; vgoal=VREF*g/gd
            vag=np.zeros(2)
            for j in range(NUM_UAV):
                if j==i: continue
                rel=p[j]-p[i]; d=np.linalg.norm(rel)+1e-9; n=rel/d
                if d<AGENT_INFLUENCE:
                    mag=K_AGENT_REP*(1/d-1/AGENT_INFLUENCE)/max(d,0.12)
                    vag-=mag*n
            vobs=np.zeros(2)
            for ox,oy,r in OBSTACLES:
                c=np.array([ox,oy]); away=p[i]-c; dc=np.linalg.norm(away)+1e-9; rho=dc-(r+ROBOT_RADIUS)
                if rho<OBSTACLE_INFLUENCE:
                    rho=max(rho,0.08); mag=K_OBSTACLE*(1/rho-1/OBSTACLE_INFLUENCE)/(rho**2); mag=min(mag,MAX_OBSTACLE_EFFECT); vobs+=mag*away/dc
            vb=np.zeros(2); dl=p[i,1]-(Y_MIN+ROBOT_RADIUS); dh=(Y_MAX-ROBOT_RADIUS)-p[i,1]
            if dl<BOUNDARY_INFLUENCE:
                d=max(dl,.05); vb[1]+=K_BOUNDARY*(1/d-1/BOUNDARY_INFLUENCE)
            if dh<BOUNDARY_INFLUENCE:
                d=max(dh,.05); vb[1]-=K_BOUNDARY*(1/d-1/BOUNDARY_INFLUENCE)
            vdes=clip_norm(vgoal+vag+vobs+vb,VMAX); u[i]=clip_norm((vdes-v[i])/DT,AMAX)
        return u,time.perf_counter()-t0

class CentralizedNMPC:
    def __init__(self): self.nx=4; self.nu=2; self.prev_X=None; self.prev_U=None; self._build()
    def xs(self,i): return slice(4*i,4*i+4)
    def us(self,i): return slice(2*i,2*i+2)
    def _build(self):
        o=ca.Opti(); self.o=o; X=o.variable(8,HORIZON+1); U=o.variable(4,HORIZON); X0=o.parameter(8); self.X=X;self.U=U;self.X0=X0; o.subject_to(X[:,0]==X0); J=0.; eps=1e-6
        for k in range(HORIZON):
            for i in range(NUM_UAV):
                x=X[self.xs(i),k]; xn=X[self.xs(i),k+1]; u=U[self.us(i),k]
                o.subject_to(xn==ca.vertcat(x[0]+DT*x[2],x[1]+DT*x[3],x[2]+DT*u[0],x[3]+DT*u[1]))
                dx=GOALS[i,0]-xn[0]; dy=GOALS[i,1]-xn[1]; gd=ca.sqrt(dx*dx+dy*dy+eps); gx=dx/gd;gy=dy/gd
                sp2=xn[2]**2+xn[3]**2; sp=ca.sqrt(sp2+eps); align=(xn[2]*gx+xn[3]*gy)/sp
                J+=W_SPEED*(sp2-VREF**2)**2+W_DIR*(1-align)**2+W_PROGRESS*(dx*dx+dy*dy)+W_U*(u[0]**2+u[1]**2)
                o.subject_to(sp2<=VMAX**2); o.subject_to(u[0]**2+u[1]**2<=AMAX**2); o.subject_to(xn[1]>=Y_MIN+ROBOT_RADIUS); o.subject_to(xn[1]<=Y_MAX-ROBOT_RADIUS)
                for ox,oy,r in OBSTACLES:
                    rs=r+ROBOT_RADIUS+OBS_MARGIN; o.subject_to((xn[0]-ox)**2+(xn[1]-oy)**2>=rs**2)
            a=X[self.xs(0),k+1]; b=X[self.xs(1),k+1]; o.subject_to((a[0]-b[0])**2+(a[1]-b[1])**2>=D_SAFE**2)
        for i in range(NUM_UAV):
            xt=X[self.xs(i),HORIZON]; J+=W_TERMINAL*((xt[0]-GOALS[i,0])**2+(xt[1]-GOALS[i,1])**2)
        o.minimize(J); o.solver('ipopt',{'expand':True,'print_time':False},{'print_level':0,'max_iter':250,'tol':1e-4,'acceptable_tol':2e-3,'sb':'yes'})
    def _guess(self,s):
        Xg=np.zeros((8,HORIZON+1)); Ug=np.zeros((4,HORIZON))
        if self.prev_X is not None:
            Xg[:,:-1]=self.prev_X[:,1:];Xg[:,-1]=self.prev_X[:,-1];Xg[:,0]=s.reshape(-1);Ug[:,:-1]=self.prev_U[:,1:];Ug[:,-1]=self.prev_U[:,-1];return Xg,Ug
        # Feasible-ish warm start: left agent is initially closer, right agent waits before gate.
        for k in range(HORIZON+1):
            t=k*DT; g=s.copy(); g[0,0]+=VREF*t; g[0,2]=VREF
            # right moves slowly for the first guess only
            g[1,0]-=0.18*VREF*t; g[1,2]=-0.18*VREF
            Xg[:,k]=g.reshape(-1)
        return Xg,Ug
    def compute(self,s):
        t0=time.perf_counter(); self.o.set_value(self.X0,s.reshape(-1)); Xg,Ug=self._guess(s); self.o.set_initial(self.X,Xg); self.o.set_initial(self.U,Ug); sol=self.o.solve(); self.prev_X=sol.value(self.X);self.prev_U=sol.value(self.U); return self.prev_U[:,0].reshape(2,2),time.perf_counter()-t0
