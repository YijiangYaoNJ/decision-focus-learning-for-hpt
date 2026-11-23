"""Action-side IEEE123 comparison with a common frozen forecast and AC release rule."""
import argparse
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from torch import nn

import ieee123_chen_actors as prior
from ieee123_chen_projection import optimize


OUT = prior.OUT.parents[1] / "ieee123_matched_inputs_v1"
MODES = ("aggregate", "connections")


@dataclass
class DecisionInput:
    plant: object
    old: np.ndarray
    fixed: np.ndarray
    past: np.ndarray
    forecast: np.ndarray

    def qp(self):
        return optimize(self.plant, self.old, self.fixed, self.past, self.forecast)

    def network(self):
        plant = self.plant
        # Same connection ordering, including line-to-line loads, as BatchAC.evaluate.
        load = plant.m.series[plant.slot]
        loads = np.column_stack((load.real, load.imag)).ravel() / 100
        x = prior.features(plant.m, plant.slot, plant.taps, self.past)
        local = prior.local_features(plant, self.old, self.past)
        return tuple(prior.joint.tensor(a[None]) for a in
                     (x, self.past, self.forecast, local, loads, self.fixed))

    def score(self, proposals, truth):
        weather = prior.joint.phase_power(np.vstack((self.past[-1], self.forecast)))
        return prior.outcomes(self.plant, self.old, self.fixed, proposals, weather, truth)


class MatchedActor(prior.DispatchActor):
    def __init__(self, seed, base, n_loads):
        super().__init__(seed, base)
        # Keep exact inherited coefficients, including the last preference column.
        previous = self.base.body[0]
        extra = n_loads * 2 + 54
        layer = nn.Linear(previous.in_features + extra, previous.out_features)
        with torch.no_grad():
            layer.weight.zero_()
            layer.weight[:, :previous.in_features-1].copy_(previous.weight[:, :-1])
            layer.weight[:, -1].copy_(previous.weight[:, -1])
            layer.bias.copy_(previous.bias)
        self.base.body[0] = layer
        self.base.mean = torch.cat((self.base.mean, torch.zeros(extra)))
        self.base.scale = torch.cat((self.base.scale, torch.ones(extra)))

    def forward(self, x, past, path, local, loads, fixed, *, mode):
        if mode == "aggregate":
            loads = torch.zeros_like(loads)
        elif mode != "connections":
            raise ValueError(mode)
        return super().forward(torch.cat((x, loads, fixed), dim=1), past, path, local)


def decision(data, i, old):
    return DecisionInput(data.plant(i), old, data.fixed, data.d["past"][i], data.ref[i])


def schema(data):
    m = data.model
    def terminal(index):
        return m.nodes[index] if 0 <= index < m.original_n else "ground"
    return [dict(column=j, positive=terminal(a), negative=terminal(b))
            for j, (a, b) in enumerate(zip(m.lp, m.lm))]


def audit(data):
    sample_slots=set()
    for i in data.val:
        obs=decision(data,i,data.d["old"][i])
        x,past,path,local,loads,fixed=obs.network()
        expected=obs.plant.m.series[obs.plant.slot]
        np.testing.assert_allclose(loads.numpy().reshape(-1,2)*100,
                                   np.column_stack((expected.real,expected.imag)),rtol=2e-7,atol=1e-6)
        np.testing.assert_allclose(x[0,18:20].numpy()*1000,[expected.real.sum(),expected.imag.sum()],rtol=2e-7)
        np.testing.assert_array_equal(path[0].numpy(),obs.forecast.astype("float32"))
        np.testing.assert_array_equal(past[0,-1].numpy(),obs.past[-1].astype("float32"))
        np.testing.assert_array_equal(local[0,:,:9].numpy().ravel(),obs.old.astype("float32"))
        np.testing.assert_array_equal(fixed[0].numpy(),obs.fixed.astype("float32"))
        np.testing.assert_allclose(x[0,20:-2].numpy()*16,obs.plant.taps)
        sample_slots.add(int(obs.plant.slot))
    info=dict(development_states=len(data.val),load_slots=len(sample_slots),load_connections=schema(data),
              forecast_sha256=prior.joint.sha(prior.joint.OUT/"init_11.pt"),
              load_source="BatchAC.m.series[slot]; alternating nominal kW and kvar / 100",
              forecast="Frozen seed-11 MSE for both proposers and release.",
              scope="Same exogenous information. Network uses causal derived features; QP queries the physical model.")
    (OUT/"input_audit.json").write_text(json.dumps(info,indent=2))


def validate(models,data,*,with_qp=False):
    rows=[]
    with torch.no_grad():
        for i in data.val:
            obs=decision(data,i,data.d["old"][i])
            args=obs.network()
            actions={mode:prior.disk(model(*args,mode=mode).numpy())[0] for mode,model in models.items()}
            if with_qp:
                actions["qp"]=obs.qp()
            truth=prior.joint.phase_power(data.d["truth"][i])
            _,cost,risks,won=obs.score(np.stack(list(actions.values())),truth)
            for j,mode in enumerate(actions):
                rows.append([i,int(data.d["date"][i]),mode,cost[j],*risks[j],bool(won[j])])
    return pd.DataFrame(rows,columns="case date mode loss_kw voltage_fraction max_voltage_gap overload_fraction max_overload_gap proposal_won".split())


def train_pair(data,seed):
    from copy import deepcopy
    torch.manual_seed(seed)
    _,base=prior.joint.load_models(seed,prior.joint.OUT/f"init_{seed}.pt")
    initial=MatchedActor(seed,base.state_dict(),len(data.model.lp))
    models={mode:deepcopy(initial) for mode in MODES}
    opts={m:torch.optim.Adam(net.parameters(),lr=.0005) for m,net in models.items()}
    rng=np.random.default_rng(seed)
    best,picked,history={}, {}, []
    for step in range(prior.STEPS+1):
        if step%prior.EVERY==0:
            val=validate(models,data).groupby("mode").mean(numeric_only=True)
            cols=["voltage_fraction","max_voltage_gap","overload_fraction","max_overload_gap"]
            if step==0:
                caps=val.loc["aggregate",cols].to_numpy()
                np.testing.assert_allclose(val.loc["aggregate"],val.loc["connections"],atol=1e-9,rtol=0)
            for mode in MODES:
                row=val.loc[mode]
                excess=np.maximum(0.,row[cols].to_numpy()-caps-1e-8) @ [1,50,2,20]
                key=(float(excess),float(row.loss_kw))
                if mode not in best or key<best[mode]:
                    best[mode],picked[mode]=key,step
                    torch.save(models[mode].state_dict(),OUT/f"{mode}_{seed}.pt")
                history.append(dict(mode=mode,step=step,risk_excess=excess,**row[["loss_kw",*cols]].to_dict()))
            if step==prior.STEPS or step-max(picked.values())>=prior.PATIENCE:
                break
        if step%prior.FRAGMENT==0:
            starts=rng.choice(data.starts,prior.STREAMS,replace=False)+rng.integers(0,121-prior.FRAGMENT,prior.STREAMS)
            carried={m:[data.d["old"][s].copy() for s in starts] for m in MODES}
        terms={mode:[] for mode in MODES}
        for j,start in enumerate(starts):
            i=int(start+step%prior.FRAGMENT)
            directions=rng.choice([-1.,1.],(prior.DIRECTIONS,54))
            truth=prior.joint.phase_power(data.d["truth"][i])
            for mode,model in models.items():
                obs=decision(data,i,carried[mode][j])
                cmd=model(*obs.network(),mode=mode)
                raw=prior.disk(cmd.detach().numpy())[0]
                delta=prior.EPSILON*directions
                probes=prior.disk(np.vstack((raw,raw+delta,raw-delta)))
                chosen,kw,risk,_=obs.score(probes,truth)
                values=prior.op_values(kw,risk)
                slope=(values[1:1+prior.DIRECTIONS]-values[1+prior.DIRECTIONS:])/(2*prior.EPSILON)
                grad=prior.joint.tensor(slope@directions/prior.DIRECTIONS)
                terms[mode].append((grad*cmd[0]).sum())
                carried[mode][j]=chosen[0].copy()
        for mode in MODES:
            opts[mode].zero_grad()
            torch.stack(terms[mode]).mean().backward()
            nn.utils.clip_grad_norm_(models[mode].parameters(),3.,error_if_nonfinite=True)
            opts[mode].step()
    for mode,model in models.items():
        model.load_state_dict(torch.load(OUT/f"{mode}_{seed}.pt",weights_only=True))
    result=validate(models,data,with_qp=True)
    result.to_csv(OUT/f"development_{seed}.csv",index=False)
    pd.DataFrame(history).to_csv(OUT/f"training_{seed}.csv",index=False)
    print(seed,step,picked,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("audit","train"))
    parser.add_argument("--seeds",nargs="+",type=int,default=[11,23,37])
    args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(1)
    with threadpool_limits(1):
        data=prior.April()
        audit(data)
        if args.stage=="train":
            for seed in args.seeds:
                train_pair(data,seed)
