"""Locked IEEE123 feeder, sparse AC equilibrium and converter losses."""

import gzip
import json
import pickle
from functools import lru_cache

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu

from hpt.config import DATA, sha

LOAD = np.array([0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
EFF = np.array([1.0, 0.96, 0.985, 0.986, 0.982, 0.976, 0.97, 0.964, 0.957])
ZERO = np.zeros(54)


class SixHPT:
    action_dim = 54

    def _factor(self, taps):
        rows, cols, vals = [], [], []
        matrices = []
        it = iter(taps)
        for ids, mats, ctrl in self.parts:
            mat = mats[next(it) + 16] if ctrl else mats[0]
            matrices.append(mat)
            r, c = np.nonzero(mat)
            keep = (ids[r] >= 0) & (ids[c] >= 0)
            rows.extend(ids[r[keep]])
            cols.extend(ids[c[keep]])
            vals.extend(mat[r[keep], c[keep]])
        if self.ratings is not None:
            for a, b, sign in (
                (self.primary, self.primary, 1),
                (self.secondary, self.secondary, 1),
                (self.primary, self.secondary, -1),
                (self.secondary, self.primary, -1),
            ):
                rows.extend(a.ravel())
                cols.extend(b.ravel())
                vals.extend((sign / self.z).ravel())
        y = coo_matrix((vals, (rows, cols)), shape=(self.n, self.n)).tocsc()
        return splu(y), matrices

    def solve(
        self,
        slot,
        action=ZERO,
        scenario="native",
        taps=None,
        start=None,
        demand_kva=None,
        generation_kw=None,
        tolerance=3e-10,
        max_iterations=100,
    ):
        if not isinstance(slot, (int, np.integer)) or not 0 <= slot < 35040:
            raise ValueError("slot must be a zero-based integer in the QSTS year")
        taps = tuple([0] * len(self.regs) if taps is None else taps)
        if len(taps) != len(self.regs) or any(
            not isinstance(x, (int, np.integer)) or abs(x) > 16 for x in taps
        ):
            raise ValueError("Expected one integer tap in [-16,16] per regulator")
        a = np.asarray(action, float)
        if (
            a.shape not in {(54,), (6, 3, 3)}
            or not np.isfinite(a).all()
            or abs(a).max() > 1 + 1e-12
        ):
            raise ValueError("Expected 54 normalized coordinates in [-1,1]")
        if self.ratings is None and a.any():
            raise ValueError("No HPT installed")
        a = a.reshape(6, 3, 3)
        c = a[:, :, 0] + 1j * a[:, :, 1]
        c = self.cfg["series_limit_pu"] * c / np.maximum(1, abs(c))
        lu, mats = self.factor(taps)
        v = np.r_[self.nominal if start is None else start, 0j]
        src = self.parts[self.source][0][:3]
        ys = mats[self.source][:3, :3]
        source_i = ys @ self.nominal[src]
        pv_node, pv = self.scenarios[scenario]
        gen = pv[slot] if generation_kw is None else np.asarray(generation_kw, float)
        nominal_load = (
            self.series[slot] if demand_kva is None else np.asarray(demand_kva, complex)
        )
        if gen.shape != pv.shape[1:] or not np.isfinite(gen).all() or (gen < 0).any():
            raise ValueError("Expected nonnegative available PV kW per connected phase")
        if (
            nominal_load.shape != self.series.shape[1:]
            or not np.isfinite(nominal_load).all()
        ):
            raise ValueError("Expected nominal complex load kVA per connection")
        for iteration in range(max_iterations):
            rhs = np.zeros(self.n + 1, complex)
            rhs[src] = source_i
            vl = v[self.lp] - v[self.lm]
            u = abs(vl) / self.lbase
            # DSS model 8 uses the nominal admittance beyond the normal voltage range.
            zip_factor = 0.3 * u * u + 0.3 * u + 0.4
            for boundary, mask in ((0.85, u < 0.85), (1.2, u > 1.2)):
                zip_factor[mask] = (0.3 * boundary**2 + 0.3 * boundary + 0.4) * (
                    u[mask] / boundary
                ) ** 2
            demand = nominal_load * zip_factor
            il = np.conj(1000 * demand / vl)
            np.add.at(rhs, self.lp, -il)
            np.add.at(rhs, self.lm, il)
            pv_pu = abs(v[pv_node]) / self.bases[pv_node]
            actual_gen = gen * (pv_pu / np.clip(pv_pu, 0.9, 1.1)) ** 2
            np.add.at(rhs, pv_node, np.conj(1000 * actual_gen / v[pv_node]))
            if self.ratings is not None:
                ref_v = v[self.primary].copy()
                for j in self.delta_sites:
                    ref_v[j] -= ref_v[j].mean()
                e = c * ref_v
                current = (v[self.primary] + e - v[self.secondary]) / self.z
                se = e * current.conj() / 1000
                q = a[:, :, 2] * self.port
                np.add.at(rhs, self.primary, -e / self.z)
                sh_current = np.conj(1000 * (se.real - 1j * q) / v[self.secondary])
                for j in self.delta_sites:
                    u = v[self.secondary[j]]
                    # A-B, B-C and C-A shunt branches; no connection to ground.
                    branch = np.conj(
                        1000 * (se[j].real - 1j * q[j]) / (u - np.roll(u, -1))
                    )
                    sh_current[j] = branch - np.roll(branch, 1)
                np.add.at(rhs, self.secondary, e / self.z - sh_current)
            vn = lu.solve(rhs[:-1])
            error = np.max(abs(vn - v[:-1]) / self.bases)
            v[:-1] = vn
            if error < tolerance:
                break
        else:
            raise ArithmeticError(f"AC iteration failed at slot {slot}, error {error}")
        grid = np.sum(v[src] * np.conj(source_i - ys @ v[src])) / 1000
        result = dict(
            v=v[:-1].copy(),
            pu=abs(v[:-1]) / self.bases,
            grid=grid,
            load_kw=demand.real.sum(),
            pv_kw=actual_gen.sum(),
            taps=taps,
            network_kw=grid.real - demand.real.sum() + actual_gen.sum(),
            iterations=iteration + 1,
        )
        flow = []
        for site in self.site_names:
            i, _, _ = self.sites[site]
            ids = self.parts[i][0]
            currents = mats[i] @ v[ids]
            terminal = slice(len(ids) // 2, len(ids) // 2 + 3)
            flow.append(-v[ids[terminal]] * currents[terminal].conj() / 1000)
        result["site_kva"] = np.array(flow)
        result["heat_kw"] = 0.0
        if self.ratings is not None:
            main = (
                np.maximum(abs(v[self.primary]), abs(v[self.secondary]))
                * abs(current)
                / (1000 * self.ratings)
            )
            ports = np.array([abs(se), abs(se.real - 1j * q)]) / self.port
            curve = LOAD * (1 / EFF - 1)
            heat = np.where(
                ports <= 1, np.interp(ports, LOAD, curve), ports * (1 / EFF[-1] - 1)
            )
            result.update(
                main=main,
                ports=ports,
                series_kva=se,
                shunt_kva=se.real - 1j * q,
                heat_kw=float((heat * self.port).sum()),
                winding_kw=float((abs(current) ** 2 * self.z.real).sum() / 1000),
            )
        result["loss_kw"] = result["network_kw"] + result["heat_kw"]
        return result


def plant_model():
    lock = json.loads(((DATA / "feeder") / "lock.json").read_text())
    file = (DATA / "feeder") / "feeder.pkl.gz"
    assert sha(file) == lock["feeder"]
    with gzip.open(file, "rb") as src:
        model = SixHPT.__new__(SixHPT)
        model.__dict__.update(pickle.load(src))
    model.factor = lru_cache(maxsize=32)(model._factor)
    return model
