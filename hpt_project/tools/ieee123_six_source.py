"""Import passive IEEE123 component primitives and the published QSTS profiles once."""
import json
import hashlib
from pathlib import Path
import sys

import numpy as np
ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/ieee123_six.json"
SOURCE = ROOT.parent / "networks/oedi_ieee123"
CACHE = ROOT / "runs/ieee123_six_v1/passive_network.npz"


def fingerprint():
    digest = hashlib.sha256(CONFIG.read_bytes())
    for path in sorted((SOURCE/"qsts").glob("*.dss")):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def circuit():
    sys.path.insert(0, str(ROOT/"runs/trans_scale_dependencies"))
    import opendssdirect as od
    d = od.NewContext()
    d.Basic.AllowChangeDir(False)
    d.Text.Command(f"compile [{SOURCE/'qsts/master.dss'}]")
    d.Text.Command("set mode=yearly stepsize=15m number=1")
    return d


def import_case(save=False, *, source_circuit=None, config=None):
    cfg = json.loads(CONFIG.read_text()) if config is None else config
    d = circuit() if source_circuit is None else source_circuit
    d.Solution.Solve()
    names = [s.lower() for s in d.Circuit.YNodeOrder()]
    index = {name: k for k, name in enumerate(names)}
    bases = []
    for name in names:
        d.Circuit.SetActiveBus(name.split(".")[0])
        bases.append(d.Bus.kVBase() * 1000)

    def terminals():
        order = np.asarray(d.CktElement.NodeOrder()).reshape(d.CktElement.NumTerminals(), -1)
        return [index[f"{bus.split('.')[0].lower()}.{phase}"] if phase else -1
                for bus, phases in zip(d.CktElement.BusNames(), order) for phase in phases]

    def primitive():
        n = d.CktElement.NumConductors() * d.CktElement.NumTerminals()
        return np.asarray(d.CktElement.YPrim()).view(complex).reshape(n, n).copy()

    regulators = {}
    for ctrl in d.RegControls.AllNames():
        d.RegControls.Name(ctrl)
        regulators[d.RegControls.Transformer().lower()] = dict(
            name=ctrl, target=d.RegControls.ForwardVreg(), band=d.RegControls.ForwardBand(),
            pt=d.RegControls.PTRatio(), ct=d.RegControls.CTPrimary(),
            r=d.RegControls.ForwardR(), x=d.RegControls.ForwardX())

    parts = []
    for name in d.Circuit.AllElementNames():
        kind, key = name.lower().split(".", 1)
        if kind not in {"line", "transformer", "capacitor", "vsource"}:
            continue
        d.Circuit.SetActiveElement(name)
        nodes = terminals()
        mats = [primitive()]
        control = regulators.get(key) if kind == "transformer" else None
        if control:
            mats = []
            d.Transformers.Name(key)
            d.Transformers.Wdg(2)
            original = d.Transformers.Tap()
            for tap in range(-16, 17):
                d.Transformers.Tap(1 + tap * .00625)
                d.Solution.BuildYMatrix(1, 1)
                d.Circuit.SetActiveElement(name)
                mats.append(primitive())
            d.Transformers.Tap(original)
            d.Solution.BuildYMatrix(1, 1)
        parts.append((name.lower(), nodes, np.array(mats), control))

    load_specs = []
    for name in d.Loads.AllNames():
        d.Loads.Name(name)
        n = d.CktElement.NumPhases()
        nodes = terminals()
        delta = d.Loads.IsDelta()
        if delta and n != 1:
            raise ValueError(f"Unsupported load connection: {name}")
        if d.Loads.Model() != 8 or not np.allclose(d.Loads.ZipV(), [.3,.3,.4,.3,.3,.4,0]):
            raise ValueError(f"Unsupported ZIP: {name}")
        for k in range(n):
            base = d.Loads.kV()*1000 / (np.sqrt(3) if n == 3 else 1)
            load_specs.append((nodes[k], nodes[1] if delta else -1, base,
                               d.Loads.kW()/n, d.Loads.kvar()/n, d.Loads.Yearly()))

    native = []
    for name in d.PVsystems.AllNames():
        d.PVsystems.Name(name)
        phases = d.CktElement.NumPhases()
        native.append((terminals()[:phases], d.PVsystems.Pmpp(), name))
    high = [([index[f"{p['bus']}.{phase}"] for phase in (1, 2, 3)], p["kva"], p["profile"])
            for p in cfg["high_pv"]]
    stations = {"native":native, "high":high}
    if save:
        meta = dict(hash=fingerprint(), names=names, bases=bases, loads=load_specs, stations=stations,
                    parts=[(name,nodes,ctrl) for name,nodes,_,ctrl in parts])
        np.savez_compressed(CACHE, meta=json.dumps(meta), **{name:m for name,_,m,_ in parts})
    loads,scenarios = profiles(load_specs,stations)
    return cfg, names, np.asarray(bases), parts, loads, scenarios


def profiles(load_specs, station_specs):
    curves = {}
    def curve(kind, name):
        key = kind, name
        if key not in curves:
            path = SOURCE/"profiles"/f"{kind}_profiles"/f"{name}.csv"
            values = np.loadtxt(path,delimiter=",") if name else np.ones(35040)
            if values.shape!=(35040,) or not np.isfinite(values).all():
                raise ValueError(f"Invalid QSTS curve: {key}")
            curves[key] = values
        return curves[key]
    loads = [(a,b,v,(p+1j*q)*curve("load",shape)) for a,b,v,p,q,shape in load_specs]
    scenarios = {}
    for scenario, stations in station_specs.items():
        nodes, series = [], []
        for phases, kw, shape in stations:
            profile = np.minimum(1, curve("pv", f"pvshape_{shape}"))
            # OpenDSS PVSystem defaults: inverter cut-in and cut-out at 20%.
            profile = np.where(profile >= .2, profile, 0)
            for phase in phases:
                nodes.append(phase)
                series.append(kw * profile / len(phases))
        scenarios[scenario] = np.array(nodes), np.asarray(series).T
    return loads, scenarios


def load_case():
    """Production path reads arrays and public curves; OpenDSS is only a build/verify dependency."""
    with np.load(CACHE, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        if meta["hash"] != fingerprint():
            raise ValueError("Feeder inputs changed; rerun ieee123-six build")
        parts = [(name,nodes,data[name],ctrl) for name,nodes,ctrl in meta["parts"]]
    loads,scenarios = profiles(meta["loads"],meta["stations"])
    return json.loads(CONFIG.read_text()),meta["names"],np.array(meta["bases"]),parts,loads,scenarios
