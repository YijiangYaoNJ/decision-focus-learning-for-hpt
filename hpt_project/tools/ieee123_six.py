"""Unbalanced nodal AC equilibrium for the six-HPT IEEE123. Volt, ampere, kW/kVA."""
from functools import lru_cache
import csv

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu

from ieee123_six_source import ROOT, load_case

LOAD = np.array([0, .02, .05, .1, .2, .4, .6, .8, 1.])
EFF = np.array([1., .96, .985, .986, .982, .976, .970, .964, .957])
ZERO = np.zeros(54)


class SixHPT:
    action_dim = 54

    @classmethod
    def commissioned(cls):
        with (ROOT/"runs/ieee123_six_v1/sites.csv").open() as f:
            rows = list(csv.DictReader(f))
        model = cls(ratings=[float(r["winding_per_phase_kva"]) for r in rows])
        if model.site_names != [r["site"] for r in rows]:
            raise ValueError("Commissioned sites differ from the configured lines")
        return model

    def __init__(self, case=None, ratings=None):
        self.cfg, self.names, bases, parts, loads, self.scenarios = case or load_case()
        self.original_n = len(bases)
        self.site_names = (self.cfg["site_elements"] if "site_elements" in self.cfg else
                           ["line." + s for s in self.cfg["site_lines"]])
        self.delta_sites = self.cfg.get("delta_site_indices",[])
        self.ratings = None if ratings is None else np.asarray(ratings, float).reshape(6, 1)
        self.parts, self.regs, self.sites = [], [], {}
        self.nodes = list(self.names)
        base = list(bases)
        for name, ids, mats, ctrl in parts:
            ids = np.array(ids)
            if name in self.site_names:
                first = len(ids)//2
                downstream = ids[first:first+3].copy()
                if min(downstream)<0 or [self.nodes[i].split(".")[1] for i in downstream] != ["1","2","3"]:
                    raise ValueError(f"Not a complete ABC circuit: {name}")
                internal = downstream
                if self.ratings is not None:
                    internal = np.arange(len(base), len(base)+3)
                    for i in downstream:
                        self.nodes.append(f"{name}_hpt.{self.nodes[i][-1]}")
                        base.append(base[i])
                    ids[first:first+3] = internal
                self.sites[name] = (len(self.parts), internal, downstream)
            if ctrl:
                self.regs.append((len(self.parts), ctrl))
            if name.startswith("vsource."):
                self.source = len(self.parts)
            self.parts.append((ids, mats, ctrl))
        self.bases = np.array(base)
        self.n = len(base)
        self.primary = np.array([self.sites[s][1] for s in self.site_names])
        self.secondary = np.array([self.sites[s][2] for s in self.site_names])
        self.phasors = np.exp(-2j*np.pi*np.arange(3)/3)
        self.nominal = self.bases * self.phasors[[int(s.split(".")[-1])-1 for s in self.nodes]]
        self.lp, self.lm, self.lbase, series = zip(*loads)
        self.lp, self.lm, self.lbase = map(np.asarray, (self.lp,self.lm,self.lbase))
        self.series = np.array(series).T
        if self.ratings is not None:
            if np.any(self.ratings <= 0):
                raise ValueError("Winding ratings must be positive")
            self.port = self.ratings*self.cfg["port_fraction"]
            z_pu = (np.asarray(self.cfg["winding_r_pu"]).reshape(-1,1)
                    +1j*np.asarray(self.cfg["winding_x_pu"]).reshape(-1,1))
            self.z = z_pu*self.bases[self.primary]**2/(1000*self.ratings)
        self.factor = lru_cache(maxsize=32)(self._factor)

    def _factor(self, taps):
        rows, cols, vals = [], [], []
        matrices = []
        it = iter(taps)
        for ids, mats, ctrl in self.parts:
            mat = mats[next(it)+16] if ctrl else mats[0]
            matrices.append(mat)
            r,c = np.nonzero(mat)
            keep = (ids[r]>=0) & (ids[c]>=0)
            rows.extend(ids[r[keep]])
            cols.extend(ids[c[keep]])
            vals.extend(mat[r[keep],c[keep]])
        if self.ratings is not None:
            for a,b,sign in ((self.primary,self.primary,1),(self.secondary,self.secondary,1),
                             (self.primary,self.secondary,-1),(self.secondary,self.primary,-1)):
                rows.extend(a.ravel())
                cols.extend(b.ravel())
                vals.extend((sign/self.z).ravel())
        y = coo_matrix((vals,(rows,cols)),shape=(self.n,self.n)).tocsc()
        return splu(y), matrices

    def solve(self, slot, action=ZERO, scenario="native", taps=None, start=None,
              demand_kva=None, generation_kw=None, tolerance=3e-10, max_iterations=100):
        if not isinstance(slot,(int,np.integer)) or not 0<=slot<35040:
            raise ValueError("slot must be a zero-based integer in the QSTS year")
        taps = tuple([0]*len(self.regs) if taps is None else taps)
        if len(taps)!=len(self.regs) or any(not isinstance(x,(int,np.integer)) or abs(x)>16 for x in taps):
            raise ValueError("Expected one integer tap in [-16,16] per regulator")
        a = np.asarray(action, float)
        if a.shape not in {(54,), (6,3,3)} or not np.isfinite(a).all() or abs(a).max()>1+1e-12:
            raise ValueError("Expected 54 normalized coordinates in [-1,1]")
        if self.ratings is None and a.any():
            raise ValueError("No HPT installed")
        a = a.reshape(6,3,3)
        c = a[:,:,0]+1j*a[:,:,1]
        c = self.cfg["series_limit_pu"]*c/np.maximum(1,abs(c))
        lu, mats = self.factor(taps)
        v = np.r_[self.nominal if start is None else start, 0j]
        src = self.parts[self.source][0][:3]
        ys = mats[self.source][:3,:3]
        source_i = ys @ self.nominal[src]
        pv_node,pv = self.scenarios[scenario]
        gen = pv[slot] if generation_kw is None else np.asarray(generation_kw,float)
        nominal_load = self.series[slot] if demand_kva is None else np.asarray(demand_kva,complex)
        if gen.shape != pv.shape[1:] or not np.isfinite(gen).all() or (gen<0).any():
            raise ValueError("Expected nonnegative available PV kW per connected phase")
        if nominal_load.shape != self.series.shape[1:] or not np.isfinite(nominal_load).all():
            raise ValueError("Expected nominal complex load kVA per connection")
        for iteration in range(max_iterations):
            rhs = np.zeros(self.n+1,complex)
            rhs[src] = source_i
            vl = v[self.lp]-v[self.lm]
            u = abs(vl)/self.lbase
            # DSS model 8 uses the nominal admittance beyond the normal voltage range.
            zip_factor = .3*u*u+.3*u+.4
            for boundary, mask in ((.85,u<.85),(1.2,u>1.2)):
                zip_factor[mask] = (.3*boundary**2+.3*boundary+.4)*(u[mask]/boundary)**2
            demand = nominal_load*zip_factor
            il = np.conj(1000*demand/vl)
            np.add.at(rhs,self.lp,-il)
            np.add.at(rhs,self.lm,il)
            pv_pu = abs(v[pv_node])/self.bases[pv_node]
            actual_gen = gen*(pv_pu/np.clip(pv_pu,.9,1.1))**2
            np.add.at(rhs,pv_node,np.conj(1000*actual_gen/v[pv_node]))
            if self.ratings is not None:
                ref_v = v[self.primary].copy()
                for j in self.delta_sites:
                    ref_v[j] -= ref_v[j].mean()
                e = c*ref_v
                current = (v[self.primary]+e-v[self.secondary])/self.z
                se = e*current.conj()/1000
                q = a[:,:,2]*self.port
                np.add.at(rhs,self.primary,-e/self.z)
                sh_current = np.conj(1000*(se.real-1j*q)/v[self.secondary])
                for j in self.delta_sites:
                    u = v[self.secondary[j]]
                    # A-B, B-C and C-A shunt branches; no connection to ground.
                    branch = np.conj(1000*(se[j].real-1j*q[j])/(u-np.roll(u,-1)))
                    sh_current[j] = branch-np.roll(branch,1)
                np.add.at(rhs,self.secondary,e/self.z-sh_current)
            vn = lu.solve(rhs[:-1])
            error = np.max(abs(vn-v[:-1])/self.bases)
            v[:-1] = vn
            if error<tolerance:
                break
        else:
            raise ArithmeticError(f"AC iteration failed at slot {slot}, error {error}")
        grid = np.sum(v[src]*np.conj(source_i-ys@v[src]))/1000
        result = dict(v=v[:-1].copy(), pu=abs(v[:-1])/self.bases, grid=grid,
                      load_kw=demand.real.sum(), pv_kw=actual_gen.sum(), taps=taps,
                      network_kw=grid.real-demand.real.sum()+actual_gen.sum(), iterations=iteration+1)
        flow = []
        for site in self.site_names:
            i,_,_ = self.sites[site]
            ids = self.parts[i][0]
            currents = mats[i]@v[ids]
            terminal = slice(len(ids)//2,len(ids)//2+3)
            flow.append(-v[ids[terminal]]*currents[terminal].conj()/1000)
        result["site_kva"] = np.array(flow)
        result["heat_kw"] = 0.
        if self.ratings is not None:
            main = np.maximum(abs(v[self.primary]),abs(v[self.secondary]))*abs(current)/(1000*self.ratings)
            ports = np.array([abs(se),abs(se.real-1j*q)])/self.port
            curve = LOAD*(1/EFF-1)
            heat = np.where(ports<=1,np.interp(ports,LOAD,curve),ports*(1/EFF[-1]-1))
            result.update(main=main, ports=ports, series_kva=se, shunt_kva=se.real-1j*q,
                          heat_kw=float((heat*self.port).sum()), winding_kw=float((abs(current)**2*self.z.real).sum()/1000))
        result["loss_kw"] = result["network_kw"] + result["heat_kw"]
        return result

    def feedback_jacobian(self, slot, action, scenario, taps, eps=.0001, **injections):
        """Fixed-tap derivative of original-node voltage magnitudes and total kW wrt 54 commands."""
        x = np.asarray(action,float).reshape(54)
        if eps<=0 or np.any(abs(x)+eps>1):
            raise ValueError("Centered feedback requires an interior action and positive step")
        def metrics(a):
            s = self.solve(slot,a,scenario,taps,**injections)
            return np.r_[s["pu"][:self.original_n],s["loss_kw"]]
        return np.stack([(metrics(x+eps*e)-metrics(x-eps*e))/(2*eps) for e in np.eye(54)],axis=1)

    def step(self, slot, action=ZERO, scenario="native", taps=None, **injections):
        """Settled 15-minute regulator equilibrium; tap state carries into the next slot."""
        taps = list(taps if taps is not None else [0]*len(self.regs))
        start = None
        for _ in range(160):
            res = self.solve(slot,action,scenario,taps,start,**injections)
            start = res["v"]
            v = np.r_[start,0j]
            _,mats = self.factor(tuple(taps))
            previous = list(taps)
            for k,(idx,ctrl) in enumerate(self.regs):
                ids = self.parts[idx][0]
                w = len(ids)//2
                i = (mats[idx]@v[ids])[w]
                vc = v[ids[w]]/ctrl["pt"] + complex(ctrl["r"],ctrl["x"])*i/ctrl["ct"]
                delta = ctrl["target"]-abs(vc)
                if abs(delta)>ctrl["band"]/2:
                    steps = round(delta*ctrl["pt"]/self.bases[ids[w]]/.00625)
                    if steps:
                        move = np.sign(steps)*max(1,int(.7*abs(steps)))
                        taps[k] = int(np.clip(taps[k]+move,-16,16))
            if previous == taps:
                return res
        raise ArithmeticError(f"Regulators did not settle at slot {slot}")
