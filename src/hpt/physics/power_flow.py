"""Scalar and batched AC responses on one fixed-tap operating point."""

import numpy as np

from hpt.physics.ieee123 import EFF, LOAD


class Snapshot:
    def __init__(self, model, slot, taps, pv):
        self.model, self.slot, self.taps, self.pv = model, slot, taps, pv
        self.keep = np.ones(model.original_n, bool)
        self.keep[model.secondary[0]] = False
        self.last_x = self.last_y = self.last_v = None
        self.jx = self.jac = None
        self.calls = 0
        self.rejected_flows = 0

    def evaluate(self, x, start=None):
        self.calls += 1
        m = self.model
        s = m.solve(
            self.slot,
            x,
            "high",
            self.taps,
            start=start,
            generation_kw=self.pv,
            tolerance=1e-8,
            max_iterations=250,
        )
        v = np.r_[s["v"], 0j]
        lv = v[m.secondary[0]]
        pu = np.r_[s["pu"][: m.original_n][self.keep], abs(lv - np.roll(lv, -1)) / 480]
        main = []
        for j, name in enumerate(m.site_names):
            ids, mats, _ = m.parts[m.sites[name][0]]
            u = v[ids]
            if j == 0:
                hv = (u[:3] - np.roll(u[:3], -1)) * (0.48 / 4.16)
                sec = u[4:7] - np.roll(u[4:7], -1)
                current = (hv - sec) / (
                    m.cfg["main_z_pu"][j] * 480**2 / (1000 * m.ratings[j, 0])
                )
                powers = np.r_[abs(hv * current), abs(sec * current)]
            else:
                powers = abs(u * (mats[0] @ u))
            main.extend(powers / (1000 * m.ratings[j, 0]))
        disk = (x.reshape(18, 3)[:, :2] ** 2).sum(1)
        margins = np.r_[
            pu - 0.95,
            1.05 - pu,
            1 - np.square(main),
            1 - s["ports"].ravel() ** 2,
            1 - disk,
        ]
        return np.r_[s["loss_kw"], margins], s

    def at(self, x):
        if self.last_x is None or not np.array_equal(x, self.last_x):
            try:
                self.last_y, s = self.evaluate(x, self.last_v)
                self.last_v = s["v"]
            except ArithmeticError:
                if self.last_y is None:
                    raise
                self.rejected_flows += 1
                self.last_y = np.r_[1e7, np.full(len(self.last_y) - 1, -1.0)]
            self.last_x = x.copy()
        return self.last_y

    def derivative(self, x):
        if self.jx is None or not np.array_equal(x, self.jx):
            y = self.at(x)
            # One-sided physical perturbations, inward at scalar bounds.
            h = np.where(x > 0.999, -1e-3, 1e-3)
            columns = []
            for i, step in enumerate(h):
                trial = x.copy()
                trial[i] += step
                try:
                    value, _ = self.evaluate(trial, self.last_v)
                except ArithmeticError:
                    trial[i] -= 2 * step
                    if abs(trial[i]) > 1:
                        raise
                    step = -step
                    value, _ = self.evaluate(trial, self.last_v)
                columns.append((value - y) / step)
            self.jac = np.stack(columns, axis=1)
            self.jx = x.copy()
        return self.jac


class BatchAC:
    def __init__(self, model, slot, taps):
        self.m, self.slot, self.taps = model, slot, taps
        self.lu, self.mats = model.factor(tuple(taps))
        self.src = model.parts[model.source][0][:3]
        self.ys = self.mats[model.source][:3, :3]
        self.source = self.ys @ model.nominal[self.src]
        self.nodes = model.scenarios["high"][0]
        self.keep = np.ones(model.original_n, bool)
        self.keep[model.secondary[0]] = False

    def evaluate(
        self, action, generation, start=None, *, max_iterations=250, relaxation=1.0
    ):
        m = self.m
        A = np.asarray(action).reshape(-1, 54)
        G = np.asarray(generation).reshape(-1, 18)
        k = max(len(A), len(G))
        a = np.broadcast_to(A, (k, 54)).T.reshape(6, 3, 3, k)
        gen = np.broadcast_to(G, (k, 18)).T
        c = a[:, :, 0] + 1j * a[:, :, 1]
        c = m.cfg["series_limit_pu"] * c / np.maximum(1, abs(c))
        q = a[:, :, 2] * m.port[:, :, None]
        init = m.nominal if start is None else start
        v = np.zeros((m.n + 1, k), complex)
        v[:-1] = np.broadcast_to(np.asarray(init).reshape(m.n, -1), (m.n, k))
        for _ in range(max_iterations):
            rhs = np.zeros_like(v)
            rhs[self.src] = self.source[:, None]
            vl = v[m.lp] - v[m.lm]
            u = abs(vl) / m.lbase[:, None]
            b = np.clip(u, 0.85, 1.2)
            demand = (
                m.series[self.slot, :, None]
                * (0.3 * b * b + 0.3 * b + 0.4)
                * (u / b) ** 2
            )
            il = np.conj(1000 * demand / vl)
            np.add.at(rhs, m.lp, -il)
            np.add.at(rhs, m.lm, il)
            u = abs(v[self.nodes]) / m.bases[self.nodes, None]
            actual = gen * (u / np.clip(u, 0.9, 1.1)) ** 2
            np.add.at(rhs, self.nodes, np.conj(1000 * actual / v[self.nodes]))
            ref = v[m.primary].copy()
            ref[0] -= ref[0].mean(axis=0)
            e = c * ref
            z = m.z[:, :, None]
            current = (v[m.primary] + e - v[m.secondary]) / z
            se = e * current.conj() / 1000
            sh = se.real - 1j * q
            ish = np.conj(1000 * sh / v[m.secondary])
            lv = v[m.secondary[0]]
            branch = np.conj(1000 * sh[0] / (lv - np.roll(lv, -1, axis=0)))
            ish[0] = branch - np.roll(branch, 1, axis=0)
            np.add.at(rhs, m.primary, -e / z)
            np.add.at(rhs, m.secondary, e / z - ish)
            vn = self.lu.solve(rhs[:-1])
            err = np.max(abs(vn - v[:-1]) / m.bases[:, None])
            if relaxation == 1.0 or err < 1e-8:
                v[:-1] = vn
            else:
                v[:-1] += relaxation * (vn - v[:-1])
            if err < 1e-8:
                break
        else:
            raise ArithmeticError("Batch AC did not settle")
        grid = (
            v[self.src] * (self.source[:, None] - self.ys @ v[self.src]).conj()
        ).real.sum(0) / 1000
        network = grid - demand.real.sum(0) + actual.sum(0)
        ports = np.stack((abs(se), abs(sh))) / m.port[None, :, :, None]
        heat = (
            np.where(
                ports <= 1,
                np.interp(ports, LOAD, LOAD * (1 / EFF - 1)),
                ports * (1 / EFF[-1] - 1),
            )
            * m.port[None, :, :, None]
        ).sum(axis=(0, 1, 2))
        lv = v[m.secondary[0]]
        pu = np.vstack(
            (
                (abs(v[: m.original_n]) / m.bases[: m.original_n, None])[self.keep],
                abs(lv - np.roll(lv, -1, axis=0)) / 480,
            )
        )
        main = []
        for j, name in enumerate(m.site_names):
            ids, mats, _ = m.parts[m.sites[name][0]]
            term = v[ids]
            if j == 0:
                hv = (term[:3] - np.roll(term[:3], -1, axis=0)) * (0.48 / 4.16)
                low = term[4:7] - np.roll(term[4:7], -1, axis=0)
                iw = (hv - low) / (
                    m.cfg["main_z_pu"][j] * 480**2 / (1000 * m.ratings[j, 0])
                )
                powers = np.vstack((abs(hv * iw), abs(low * iw)))
            else:
                powers = abs(term * (mats[0] @ term))
            main.append(powers / (1000 * m.ratings[j, 0]))
        disk = (a[:, :, :2] ** 2).sum(2).reshape(18, k)
        return np.vstack(
            (
                network + heat,
                pu - 0.95,
                1.05 - pu,
                1 - np.vstack(main) ** 2,
                1 - ports.reshape(36, k) ** 2,
                1 - disk,
            )
        ).T

    def jac(self, x, pv, v):
        step = np.where(x > 0.999, -1e-3, 1e-3)
        cmd = np.vstack((x, x + np.diag(step)))
        try:
            y = self.evaluate(cmd, pv, v)
            return ((y[1:] - y[0]) / step[:, None]).T
        except ArithmeticError:
            return Snapshot(self.m, self.slot, self.taps, pv).derivative(x)


class ProposalPowerFlow:
    """Retry difficult, unexecuted proposals using the same AC residual tolerance."""

    def __init__(self, plant):
        self.plant = plant
        self.m = plant.m

    def evaluate(self, action, power, *, max_iterations=250):
        try:
            return self.plant.evaluate(action, power, max_iterations=max_iterations)
        except ArithmeticError:
            if max_iterations < 1500:
                raise
        for relaxation in (0.5, 0.25, 0.1):
            try:
                return self.plant.evaluate(
                    action, power, max_iterations=1500, relaxation=relaxation
                )
            except ArithmeticError:
                continue
        raise ArithmeticError("Proposal AC residual did not reach 1e-8 with damping")
