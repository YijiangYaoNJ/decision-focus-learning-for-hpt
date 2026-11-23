"""Timed single-snapshot solvers on the native, fixed-tap six-HPT AC equations."""
from time import perf_counter

import numpy as np
import osqp
from scipy.optimize import minimize
from scipy.sparse import csc_matrix, eye, vstack



class Snapshot:
    def __init__(self, model, slot, taps, pv):
        self.model,self.slot,self.taps,self.pv = model,slot,taps,pv
        self.keep = np.ones(model.original_n,bool)
        self.keep[model.secondary[0]] = False
        self.last_x = self.last_y = self.last_v = None
        self.jx = self.jac = None
        self.calls = 0
        self.rejected_flows = 0

    def evaluate(self,x,start=None):
        self.calls+=1
        m=self.model
        s=m.solve(self.slot,x,"high",self.taps,start=start,generation_kw=self.pv,
                  tolerance=1e-8,max_iterations=250)
        v=np.r_[s["v"],0j]
        lv=v[m.secondary[0]]
        pu=np.r_[s["pu"][:m.original_n][self.keep],abs(lv-np.roll(lv,-1))/480]
        main=[]
        for j,name in enumerate(m.site_names):
            ids,mats,_=m.parts[m.sites[name][0]]
            u=v[ids]
            if j==0:
                hv=(u[:3]-np.roll(u[:3],-1))*(.48/4.16)
                sec=u[4:7]-np.roll(u[4:7],-1)
                current=(hv-sec)/(m.cfg["main_z_pu"][j]*480**2/(1000*m.ratings[j,0]))
                powers=np.r_[abs(hv*current),abs(sec*current)]
            else:
                powers=abs(u*(mats[0]@u))
            main.extend(powers/(1000*m.ratings[j,0]))
        disk=(x.reshape(18,3)[:,:2]**2).sum(1)
        margins=np.r_[pu-.95,1.05-pu,1-np.square(main),1-s["ports"].ravel()**2,1-disk]
        return np.r_[s["loss_kw"],margins],s

    def at(self,x):
        if self.last_x is None or not np.array_equal(x,self.last_x):
            try:
                self.last_y,s=self.evaluate(x,self.last_v)
                self.last_v=s["v"]
            except ArithmeticError:
                if self.last_y is None: raise
                self.rejected_flows+=1
                self.last_y=np.r_[1e7,np.full(len(self.last_y)-1,-1.)]
            self.last_x=x.copy()
        return self.last_y

    def derivative(self,x):
        if self.jx is None or not np.array_equal(x,self.jx):
            y=self.at(x)
            # One-sided physical perturbations, inward at scalar bounds.
            h=np.where(x>.999,-1e-3,1e-3)
            columns=[]
            for i,step in enumerate(h):
                trial=x.copy()
                trial[i]+=step
                try:
                    value,_=self.evaluate(trial,self.last_v)
                except ArithmeticError:
                    trial[i]-=2*step
                    if abs(trial[i])>1: raise
                    step=-step
                    value,_=self.evaluate(trial,self.last_v)
                columns.append((value-y)/step)
            self.jac=np.stack(columns,axis=1)
            self.jx=x.copy()
        return self.jac


def qp(x,y,J,radius,restore=False):
    # Linearized constraints with a small common numerical guard. Voltage and
    # utilization safety is independently verified on the nonlinear network.
    guard=2e-5
    A=vstack([csc_matrix(J[1:]),eye(54)],format="csc")
    lo=np.r_[guard-y[1:],np.maximum(-radius,-1-x)]
    hi=np.r_[np.full(len(y)-1,np.inf),np.minimum(radius,1-x)]
    prob=osqp.OSQP()
    prob.setup(P=eye(54,format="csc")*10,q=np.zeros(54) if restore else J[0],A=A,l=lo,u=hi,verbose=False,
               eps_abs=2e-6,eps_rel=2e-6,max_iter=5000,polishing=True)
    r=prob.solve()
    if r.info.status_val not in (1,2): return None,r.info.status
    return np.clip(x+r.x,-1,1),r.info.status


def solve(model,slot,taps,pv,old,method):
    t0=perf_counter()
    f=Snapshot(model,slot,taps,pv)
    events=[]
    visited=set()
    def inspect(x):
        y=f.at(x)
        key=x.tobytes()
        if key not in visited:
            visited.add(key)
            if min(y[1:])>=-1e-7 and (not events or y[0]<events[-1][2]-1e-7):
                checked,_=f.evaluate(x,start=None)
                if min(checked[1:])>=0 and (not events or checked[0]<events[-1][2]-1e-7):
                    events.append((perf_counter()-t0,x.copy(),checked[0]))
        return y
    x=old.copy()
    y=inspect(x)
    initial_ok=bool(events)
    status=""; nit=0
    if method=="AC-OPF":
        # Reduced-space nonconvex AC OPF. Every trial resolves the AC equations.
        if not initial_ok:
            restored,_=qp(x,y,f.derivative(x),1.,restore=True)
            if restored is not None and min(inspect(restored)[1:])>min(y[1:]):
                x=restored
        result=minimize(lambda a:inspect(a)[0],x,method="SLSQP",
            jac=lambda a:f.derivative(a)[0],bounds=[(-1,1)]*54,
            constraints=[dict(type="ineq",fun=lambda a:f.at(a)[1:]-2e-5,
                              jac=lambda a:f.derivative(a)[1:])],
            options=dict(ftol=1e-4,maxiter=45,disp=False))
        inspect(result.x)
        status=result.message
        nit=result.nit
    else:
        for nit in range(1,(2 if method=="QP" else 11)):
            y=inspect(x)
            xn,status=qp(x,y,f.derivative(x),1.0,restore=min(y[1:])<0)
            if xn is None: break
            # Exact AC line search, common to one-step and sequential QP.
            accepted=False
            for alpha in (1,.5,.25,.125,.0625):
                trial=x+alpha*(xn-x)
                new=inspect(trial)
                merit=lambda z:z[0]+1e6*max(0,2e-5-min(z[1:]))
                if merit(new)<merit(y)-1e-5:
                    x=trial
                    accepted=True
                    break
            if not accepted or np.max(abs(xn-f.jx))<2e-4: break
    finish=perf_counter()-t0
    return events,dict(elapsed_s=finish,first_s=events[0][0] if events else np.nan,
        first_new_s=next((t for t,a,_ in events if np.max(abs(a-old))>1e-8),np.nan),
        initial_ok=initial_ok,ac_calls=f.calls,iterations=int(nit),status=str(status),
        rejected_flows=f.rejected_flows)
