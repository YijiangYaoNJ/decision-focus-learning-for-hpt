"""Six MV/LV service interfaces with matched passive transformers and causal controls."""
import json

import numpy as np
import pandas as pd

from ieee123_six import SixHPT, ZERO
from ieee123_six_source import ROOT, CONFIG, circuit, import_case, load_case

OUT = ROOT/"runs/ieee123_service_areas_v2"
SPEC = json.loads((ROOT/"configs/ieee123_service_areas.json").read_text())
SITES = SPEC["sites"]
DAYS = [14,104,195,287]  # Jan/Apr/Jul/Oct 15 in the QSTS non-leap year (zero-based).


def build():
    d = circuit()
    mv_to_lv = {s["mv"]:s["lv"] for s in SITES}
    d.Solution.Solve()
    original_names = d.Transformers.AllNames()
    d.Transformers.Name("xfm1")
    assert d.CktElement.BusNames()==["61s","610"]
    cfg = json.loads(CONFIG.read_text())
    spec = SPEC["extensions"]
    ratings, rows, moves, tertiary = [], [], [], []
    for s in SITES:
        if not s["existing"]:
            d.Text.Command(f"new transformer.{s['element']} phases=3 windings=2 "
                           f"buses=[{s['mv']} {s['lv']}] conns=[wye wye] "
                           f"kvs=[{spec['kv_high']} {spec['kv_low']}] kvas=[{spec['kva_3phase']} {spec['kva_3phase']}] "
                           f"%rs=[{spec['r_pct_per_winding']} {spec['r_pct_per_winding']}] xhl={spec['xhl_pct']}")
        d.Transformers.Name(s["element"])
        d.Transformers.Wdg(1)
        r1 = d.Transformers.R()
        d.Transformers.Wdg(2)
        main = d.Transformers.kVA()
        r2,x = d.Transformers.R(),d.Transformers.Xhl()
        x_series = SPEC["tertiary_x_fraction_of_main_impedance"]*np.hypot(r1+r2,x)
        tertiary.append(x_series/100)
        ratings.append(main/3)
        rows.append(dict(**{k:s[k] for k in ("element","mv","lv","existing")},kva=main,
                         connection="delta/delta" if s["existing"] else "grounded-wye/grounded-wye",
                         r1_percent=r1,r2_percent=r2,xhl_percent=x,extra_series_x_percent=x_series,
                         main_source="OEDI IEEE123 XFM1" if s["existing"] else "IEEE13 XFM-1"))
        for name in s["loads"]:
            d.Loads.Name(name)
            if d.Loads.Name()!=name:
                raise ValueError(name)
            before=d.CktElement.BusNames()[0]
            _,sep,phases=before.partition(".")
            dest=s["lv"]+sep+phases
            kv=.48 if d.Loads.IsDelta() or d.CktElement.NumPhases()==3 else .48/np.sqrt(3)
            if s["existing"] and not d.Loads.IsDelta():
                raise ValueError("A floating delta secondary requires line-line service")
            moves.append(dict(kind="load",name=name,old_bus=before,new_bus=dest,shape=d.Loads.Yearly()))
            d.Text.Command(f"edit load.{name} bus1={dest} kv={kv:.13g}")
    for name in d.PVsystems.AllNames():
        d.PVsystems.Name(name)
        before=d.CktElement.BusNames()[0]
        bus,sep,phases=before.partition(".")
        if bus in mv_to_lv:
            dest=mv_to_lv[bus]+sep+phases
            kv=.48 if d.CktElement.NumPhases()==3 else .48/np.sqrt(3)
            d.Text.Command(f"edit pvsystem.{name} bus1={dest} kv={kv:.13g}")
            moves.append(dict(kind="native_pv",name=name,old_bus=before,new_bus=dest,shape=f"pvshape_{name}"))
    for panel in cfg["high_pv"]:
        old=panel["bus"]
        panel["bus"]=mv_to_lv.get(old,old)
        moves.append(dict(kind="high_pv",name=old,old_bus=old,new_bus=panel["bus"],shape="pvshape_"+panel["profile"]))
    d.Text.Command("calcvoltagebases")
    d.Solution.DblHour(0.)
    cfg.update(site_elements=["transformer."+s["element"] for s in SITES],delta_site_indices=[0],
               port_fraction=SPEC["port_fraction"],winding_r_pu=0.,winding_x_pu=tertiary,
               main_z_pu=[complex(r["r1_percent"]+r["r2_percent"],r["xhl_percent"])/100 for r in rows])
    case = import_case(source_circuit=d,config=cfg)
    ordinary=SixHPT(case)
    upgraded=SixHPT(case,ratings=ratings)
    # Verify every nominal load/PV channel for all 35,040 timestamps, irrespective of relocation.
    ref=load_case()
    assert np.array_equal(ordinary.series,np.asarray([x[3] for x in ref[4]]).T)
    for scenario,(_,values) in ordinary.scenarios.items():
        assert np.array_equal(values,ref[5][scenario][1])
    assert len(set(x["name"] for x in moves if x["kind"]=="load"))==sum(len(s["loads"]) for s in SITES)
    return ordinary,upgraded,d,rows,moves,original_names


def metrics(model,state,slot,scenario,arm,ratings):
    v=state["v"]
    keep=np.ones(model.original_n,bool)
    keep[model.secondary[0]]=False
    delta=v[model.secondary[0]]
    ll=abs(delta-np.roll(delta,-1))/480
    u=np.r_[state["pu"][:model.original_n][keep],ll]
    main,main_loss=[],0.
    for j,name in enumerate(model.site_names):
        i=model.sites[name][0]
        ids,mats,_=model.parts[i]
        terminals=np.r_[v,0j][ids]
        current=mats[0]@terminals
        main_loss+=(terminals@current.conj()).real/1000
        if j==0:
            # Delta/delta branch windings carry delta currents rather than line currents.
            hv=terminals[:3]-np.roll(terminals[:3],-1)
            lv=terminals[4:7]-np.roll(terminals[4:7],-1)
            z=model.cfg["main_z_pu"][j]*480**2/(ratings[j]*1000)
            iw=(hv*(.48/4.16)-lv)/z
            main.append(np.max(np.maximum(abs(hv)*(.48/4.16),abs(lv))*abs(iw))/(1000*ratings[j]))
        else:
            main.append(np.max(abs(terminals*current.conj()))/(1000*ratings[j]))
    port=state["ports"].max() if "ports" in state else 0.
    return dict(scenario=scenario,arm=arm,slot=slot,day=slot//96+1,vmin=u.min(),vmax=u.max(),
                excursion=max(0,.95-u.min(),u.max()-1.05),violation=bool(u.min()<.95 or u.max()>1.05),
                main_util=max(main),peak_main_site=SITES[int(np.argmax(main))]["element"],
                port_util=port,overload=bool(max(max(main),port)>1+1e-7),
                grid_kw=state["grid"].real,load_kw=state["load_kw"],pv_kw=state["pv_kw"],
                main_kw=main_loss,network_kw=state["network_kw"],heat_kw=state["heat_kw"],loss_kw=state["loss_kw"])


def local(state,model,old):
    if state is None:
        return ZERO.copy()
    v=state["v"][model.secondary].copy()
    v[0]-=v[0].mean()
    mag=abs(v)/model.bases[model.secondary]
    # Estimate uncompensated voltage from the last executed series command.
    uncompensated=mag/(1+.1*old.reshape(6,3,3)[:,:,0])
    err=1-uncompensated
    err=np.sign(err)*np.maximum(abs(err)-.02,0)
    a=np.zeros((6,3,3))
    a[:,:,0]=np.clip(6*err,-1,1)
    a[:,:,2]=np.clip(2*err,-.25,.25)
    return a.ravel()


def check_upgraded(d,hpt):
    """Independent explicit DSS voltage sources and shunt loads, including a three-wire secondary."""
    for j,s in enumerate(SITES):
        d.Text.Command(f"edit transformer.{s['element']} wdg=2 bus=inside{j}")
        for k in range(3):
            z=hpt.z[j,k]
            end=f"{s['lv']}.{k+1}"
            d.Text.Command(f"new vsource.h{j}_{k} phases=1 bus1={end} bus2=inside{j}.{k+1} "
                           f"basekv={.48/np.sqrt(3)} pu=0 r1=0 x1={z.imag} r0=0 x0={z.imag}")
            pair=end+(f".{(k+1)%3+1}" if s["existing"] else ".0")
            kv=.48 if s["existing"] else .48/np.sqrt(3)
            d.Text.Command(f"new load.h{j}_{k} phases=1 bus1={pair} kv={kv} model=1 "
                           "kw=0 kvar=0 status=fixed vminpu=.5 vmaxpu=1.5")
    rng=np.random.default_rng(123)
    errors=[]
    for scenario in ("native","high"):
        if scenario=="high":
            for name in d.PVsystems.AllNames():
                d.Text.Command(f"disable pvsystem.{name}")
            for panel in hpt.cfg["high_pv"]:
                bus=panel["bus"]
                kv=.48 if bus in [s["lv"] for s in SITES] else 4.16
                d.Text.Command(f"new pvsystem.check{bus} phases=3 bus1={bus} kv={kv} "
                               f"kva={panel['kva']} pmpp={panel['kva']} pf=1 yearly=pvshape_{panel['profile']}")
        for slot,action in ((0,ZERO),(195*96+48,rng.uniform(-.16,.16,54))):
            a=action.reshape(6,3,3)
            state=hpt.step(slot,a,scenario)
            d.Solution.DblHour((slot+1)/4)
            for (_,ctrl),tap in zip(hpt.regs,state["taps"]):
                d.RegControls.Name(ctrl["name"])
                d.RegControls.TapNumber(tap)
            d.Solution.SolveNoControl()
            index={s.lower():k for k,s in enumerate(d.Circuit.AllNodeNames())}
            inside=np.array([[index[f"inside{j}.{k+1}"] for k in range(3)] for j in range(6)])
            p=np.zeros((6,3))
            for _ in range(60):
                old=np.asarray(d.Circuit.AllBusVolts()).view(complex)[inside]
                ref=old.copy()
                ref[0]-=ref[0].mean()
                e=.1*(a[:,:,0]+1j*a[:,:,1])*ref
                for j,k in np.ndindex(6,3):
                    d.Vsources.Name(f"h{j}_{k}")
                    d.Vsources.PU(abs(e[j,k])/(480/np.sqrt(3)))
                    d.Vsources.AngleDeg(float(np.angle(e[j,k],deg=True)))
                    d.Loads.Name(f"h{j}_{k}")
                    d.Loads.kW(p[j,k])
                    d.Loads.kvar(-a[j,k,2]*hpt.port[j,0])
                d.Solution.SolveNoControl()
                assert d.Solution.Converged()
                si=np.empty((6,3),complex)
                for j,k in np.ndindex(6,3):
                    d.Vsources.Name(f"h{j}_{k}")
                    si[j,k]=-e[j,k]*complex(*d.CktElement.Currents()[:2]).conjugate()/1000
                volts=np.asarray(d.Circuit.AllBusVolts()).view(complex)
                change=volts[inside]-old
                change[0]-=change[0].mean()
                if abs(si.real-p).max()<1e-7 and abs(change).max()<1e-5:
                    break
                p=si.real.copy()
            else:
                raise ArithmeticError("Reference converter ports failed to settle")
            diff=state["v"][:hpt.original_n]-volts[[index[n] for n in hpt.names]]
            diff[hpt.secondary[0]]-=diff[hpt.secondary[0]].mean()
            errors.append(dict(scenario=scenario,slot=slot,v_pu=(abs(diff)/hpt.bases[:hpt.original_n]).max(),
                               network_kw=abs(d.Circuit.Losses()[0]/1000-state["network_kw"]),
                               series_kva=abs(si-state["series_kva"]).max()))
    df=pd.DataFrame(errors)
    df.to_csv(OUT/"upgraded_reference.csv",index=False)
    maxima=df[["v_pu","network_kw","series_kva"]].max()
    if not (maxima < [1e-4,.05,.01]).all():
        raise ArithmeticError(maxima)
    return maxima.to_dict()


def main():
    OUT.mkdir(exist_ok=True,parents=True)
    ordinary,hpt,d,sites,moves,inventory=build()
    ratings=np.array([s["kva"]/3 for s in sites])
    pd.DataFrame(sites).to_csv(OUT/"sites.csv",index=False)
    pd.DataFrame(moves).to_csv(OUT/"mapping.csv",index=False)
    # Independent direct OpenDSS replay, same ordinary transformers and ZIP/PV profiles.
    errors=[]
    for slot in (0,47,96*195+48,96*287+70):
        d.Solution.DblHour((slot+1)/4)
        d.Text.Command("set tolerance=.000000001 maxiterations=100")
        d.Solution.SolveSnap()
        assert d.Solution.Converged() and d.Solution.ControlActionsDone()
        taps={d.RegControls.Name():d.RegControls.TapNumber() for _ in d.RegControls}
        fixed=[taps[c["name"]] for _,c in ordinary.regs]
        res=ordinary.solve(slot,taps=fixed)
        vr=dict(zip(d.Circuit.AllNodeNames(),np.asarray(d.Circuit.AllBusVolts()).view(complex)))
        ref=np.array([vr[n] for n in ordinary.names])
        diff=res["v"]-ref
        diff[ordinary.secondary[0]]-=diff[ordinary.secondary[0]].mean()
        errors.append((float((abs(diff)/ordinary.bases).max()),abs(res["network_kw"]-d.Circuit.Losses()[0]/1000)))
    checks=dict(original_transformer_elements=inventory,annual_nominal_load_and_pv_exact=True,
                reference_max_v_pu=max(x[0] for x in errors),reference_max_loss_kw=max(x[1] for x in errors))
    if checks["reference_max_v_pu"]>1e-4 or checks["reference_max_loss_kw"]>.05:
        raise ArithmeticError(checks)
    checks["upgraded_reference_max_errors"]=check_upgraded(d,hpt)
    rows=[]
    for scenario in ordinary.scenarios:
        for day in DAYS:
            for arm,model in (("ordinary",ordinary),("ports_off",hpt),("controlled",hpt)):
                prev,taps,a=None,None,ZERO
                for slot in range((day-1)*96,(day+1)*96):
                    a=local(prev,model,a) if arm=="controlled" else ZERO
                    prev=model.step(slot,a,scenario,taps)
                    taps=prev["taps"]
                    if slot>=day*96:
                        rows.append(metrics(model,prev,slot,scenario,arm,ratings))
    df=pd.DataFrame(rows)
    df.to_csv(OUT/"slots.csv.gz",index=False)
    daily=df.groupby(["scenario","arm","day"]).agg(
        slots=("slot","count"),violations=("violation","sum"),overloads=("overload","sum"),
        worst_pu=("excursion","max"),peak_main=("main_util","max"),peak_port=("port_util","max"),
        **{k+"_kwh":(k+"_kw",lambda x:x.sum()*.25) for k in ("loss","network","main","heat","load","pv")}).reset_index()
    daily.to_csv(OUT/"daily.csv",index=False)
    summary=daily.groupby(["scenario","arm"]).agg(slots=("slots","sum"),violations=("violations","sum"),
        overloads=("overloads","sum"),worst_pu=("worst_pu","max"),peak_main=("peak_main","max"),
        peak_port=("peak_port","max"),**{k+"_kwh_day":(k+"_kwh","mean") for k in ("loss","network","main","heat","load","pv")})
    summary.to_csv(OUT/"summary.csv")
    assert len(df)==2*3*384 and not df.isna().any().any()
    (OUT/"audit.json").write_text(json.dumps(checks,indent=2))
    print(summary.to_string(),flush=True)

if __name__=="__main__":
    main()
