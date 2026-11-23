"""Multiple right-hand sides for the existing fixed-tap AC equilibrium.

Batch either weather scenarios or finite-difference action perturbations. The
single-case reference and independent release auditor use the native solver.
"""
import numpy as np
from ieee123_six import LOAD,EFF
from ieee123_ramp_dispatch import Snapshot


class BatchAC:
    def __init__(self,model,slot,taps):
        self.m,self.slot,self.taps=model,slot,taps
        self.lu,self.mats=model.factor(tuple(taps))
        self.src=model.parts[model.source][0][:3]
        self.ys=self.mats[model.source][:3,:3]
        self.source=self.ys@model.nominal[self.src]
        self.nodes=model.scenarios["high"][0]
        self.keep=np.ones(model.original_n,bool)
        self.keep[model.secondary[0]]=False

    def evaluate(self,action,generation,start=None,*,max_iterations=250,relaxation=1.0):
        m=self.m
        A=np.asarray(action).reshape(-1,54)
        G=np.asarray(generation).reshape(-1,18)
        k=max(len(A),len(G))
        a=np.broadcast_to(A,(k,54)).T.reshape(6,3,3,k)
        gen=np.broadcast_to(G,(k,18)).T
        c=a[:,:,0]+1j*a[:,:,1]
        c=m.cfg["series_limit_pu"]*c/np.maximum(1,abs(c))
        q=a[:,:,2]*m.port[:,:,None]
        init=m.nominal if start is None else start
        v=np.zeros((m.n+1,k),complex)
        v[:-1]=np.broadcast_to(np.asarray(init).reshape(m.n,-1),(m.n,k))
        for _ in range(max_iterations):
            rhs=np.zeros_like(v)
            rhs[self.src]=self.source[:,None]
            vl=v[m.lp]-v[m.lm]
            u=abs(vl)/m.lbase[:,None]
            b=np.clip(u,.85,1.2)
            demand=m.series[self.slot,:,None]*(.3*b*b+.3*b+.4)*(u/b)**2
            il=np.conj(1000*demand/vl)
            np.add.at(rhs,m.lp,-il)
            np.add.at(rhs,m.lm,il)
            u=abs(v[self.nodes])/m.bases[self.nodes,None]
            actual=gen*(u/np.clip(u,.9,1.1))**2
            np.add.at(rhs,self.nodes,np.conj(1000*actual/v[self.nodes]))
            ref=v[m.primary].copy()
            ref[0]-=ref[0].mean(axis=0)
            e=c*ref
            z=m.z[:,:,None]
            current=(v[m.primary]+e-v[m.secondary])/z
            se=e*current.conj()/1000
            sh=se.real-1j*q
            ish=np.conj(1000*sh/v[m.secondary])
            lv=v[m.secondary[0]]
            branch=np.conj(1000*sh[0]/(lv-np.roll(lv,-1,axis=0)))
            ish[0]=branch-np.roll(branch,1,axis=0)
            np.add.at(rhs,m.primary,-e/z)
            np.add.at(rhs,m.secondary,e/z-ish)
            vn=self.lu.solve(rhs[:-1])
            err=np.max(abs(vn-v[:-1])/m.bases[:,None])
            if relaxation==1.0 or err<1e-8:
                v[:-1]=vn
            else:
                v[:-1]+=relaxation*(vn-v[:-1])
            if err<1e-8: break
        else:
            raise ArithmeticError("Batch AC did not settle")
        grid=(v[self.src]*(self.source[:,None]-self.ys@v[self.src]).conj()).real.sum(0)/1000
        network=grid-demand.real.sum(0)+actual.sum(0)
        ports=np.stack((abs(se),abs(sh)))/m.port[None,:,:,None]
        heat=(np.where(ports<=1,np.interp(ports,LOAD,LOAD*(1/EFF-1)),ports*(1/EFF[-1]-1))
              *m.port[None,:,:,None]).sum(axis=(0,1,2))
        lv=v[m.secondary[0]]
        pu=np.vstack(((abs(v[:m.original_n])/m.bases[:m.original_n,None])[self.keep],
                      abs(lv-np.roll(lv,-1,axis=0))/480))
        main=[]
        for j,name in enumerate(m.site_names):
            ids,mats,_=m.parts[m.sites[name][0]]
            term=v[ids]
            if j==0:
                hv=(term[:3]-np.roll(term[:3],-1,axis=0))*(.48/4.16)
                low=term[4:7]-np.roll(term[4:7],-1,axis=0)
                iw=(hv-low)/(m.cfg["main_z_pu"][j]*480**2/(1000*m.ratings[j,0]))
                powers=np.vstack((abs(hv*iw),abs(low*iw)))
            else:
                powers=abs(term*(mats[0]@term))
            main.append(powers/(1000*m.ratings[j,0]))
        disk=(a[:,:,:2]**2).sum(2).reshape(18,k)
        return np.vstack((network+heat,pu-.95,1.05-pu,1-np.vstack(main)**2,
                          1-ports.reshape(36,k)**2,1-disk)).T

    def jac(self,x,pv,v):
        step=np.where(x>.999,-1e-3,1e-3)
        cmd=np.vstack((x,x+np.diag(step)))
        try:
            y=self.evaluate(cmd,pv,v)
            return ((y[1:]-y[0])/step[:,None]).T
        except ArithmeticError:
            return Snapshot(self.m,self.slot,self.taps,pv).derivative(x)
