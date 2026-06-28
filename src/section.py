import numpy as np
from typing import List
from .materials import Concrete, Steel


class Fiber:
    def __init__(self, y: float, area: float, material):
        self.y = y
        self.area = area
        self.material = material

    def stress(self, eps: float) -> float:
        return self.material.stress(eps)

    def tangent(self, eps: float) -> float:
        return self.material.tangent(eps)


class RCSection:
    def __init__(self, b: float, h: float, cover_top: float, cover_bot: float,
                 As_top: float, As_bot: float,
                 fc: float, fy: float, Es: float = 200e3,
                 conc_ecu: float = 0.0035, conc_ec0: float = 0.002,
                 n_fiber_concrete: int = 100):
        self.b = b
        self.h = h
        self.d_top = cover_top
        self.d_bot = cover_bot
        self.As_top = As_top
        self.As_bot = As_bot
        self.d = h - cover_bot

        self.concrete = Concrete(fc, conc_ecu, conc_ec0)
        self.steel = Steel(fy, Es)

        self.fibers: List[Fiber] = []
        self._build_fibers(n_fiber_concrete)

    def _build_fibers(self, n: int):
        self.fibers.clear()
        dy = self.h / n
        for i in range(n):
            y = dy * i + dy / 2
            area = self.b * dy
            self.fibers.append(Fiber(y, area, self.concrete))
        if self.As_top > 0:
            self.fibers.append(Fiber(self.d_top, self.As_top, self.steel))
        if self.As_bot > 0:
            self.fibers.append(Fiber(self.h - self.d_bot, self.As_bot, self.steel))

    def section_force(self, c: float, phi: float) -> tuple:
        """Compute axial force and moment for given neutral axis depth and curvature.
        c: distance from top fiber to neutral axis (mm).
        phi: curvature (1/mm). Positive => tension at bottom.

        eps(y) = phi * (y - c)

        N: compression positive.
        M: positive when bottom in tension (sagging). Moment about mid-height.
        """
        N = 0.0
        M = 0.0
        y_ref = self.h / 2
        for f in self.fibers:
            eps = phi * (f.y - c)
            sig = f.stress(eps)
            N -= sig * f.area
            M -= sig * f.area * (y_ref - f.y)
        return N, M

    def section_tangent_c(self, c: float, phi: float) -> float:
        """dN/dc at given c, phi."""
        dNdc = 0.0
        for f in self.fibers:
            eps = phi * (f.y - c)
            Et = f.tangent(eps)
            dNdc += Et * f.area * phi
        return dNdc

    def moment_curvature(self, N: float = 0.0, n_steps: int = 200,
                         max_phi: float = 0.5) -> tuple:
        curvatures = []
        moments = []
        nas = []
        top_strains = []
        mid_strains = []

        EA = (self.concrete.Ec * self.b * self.h
              + self.steel.Es * (self.As_top + self.As_bot))
        phi_min = max(1e-12, abs(N) / max(1.0, EA * self.h * 0.5))
        phi_vals = np.logspace(np.log10(phi_min), np.log10(max_phi), n_steps)
        c_guess = self.h / 2
        had_converged = False
        M_last = -1e30

        for phi in phi_vals:
            c, converged = self._find_c_for_N(N, phi, c_guess, max_iter=80)
            if not converged:
                if had_converged:
                    break
                continue
            had_converged = True
            N_calc, M = self.section_force(c, phi)
            eps_top = phi * (0 - c)
            eps_mid = phi * (self.h / 2 - c)
            if self._failed(c, phi):
                if M > M_last:
                    curvatures.append(phi)
                    moments.append(M)
                    nas.append(c)
                    top_strains.append(eps_top)
                    mid_strains.append(eps_mid)
                break
            if M > M_last:
                curvatures.append(phi)
                moments.append(M)
                nas.append(c)
                top_strains.append(eps_top)
                mid_strains.append(eps_mid)
                M_last = M
            c_guess = c

        return (np.array(curvatures), np.array(moments), np.array(nas),
                np.array(top_strains), np.array(mid_strains))

    def _find_c_for_N(self, N_target: float, phi: float, c_guess: float,
                      tol: float = 500.0, max_iter: int = 80) -> tuple:
        if phi < 1e-15:
            return c_guess, False

        EA = self.concrete.Ec * self.b * self.h + self.steel.Es * (self.As_top + self.As_bot)
        N_possible = phi * EA * self.h
        if abs(N_target) > 0 and abs(N_target) > N_possible * 1.5:
            return c_guess, False

        c = np.clip(c_guess, -self.h * 5, self.h * 5)
        c_max = self.h * 10
        c_min = -self.h * 5

        for iteration in range(max_iter):
            N_calc, _ = self.section_force(c, phi)
            residual = N_calc - N_target
            if abs(residual) < tol:
                return c, True

            if N_calc < N_target:
                c_min = max(c_min, c)
            else:
                c_max = min(c_max, c)

            dNdc = self.section_tangent_c(c, phi)
            if abs(dNdc) < 1.0:
                c = (c_min + c_max) / 2
            else:
                dc = residual / dNdc
                if abs(dc) > 0.5 * (c_max - c_min):
                    dc = np.sign(dc) * 0.5 * (c_max - c_min)
                c_new = c + dc
                if c_new <= c_min or c_new >= c_max:
                    c_new = (c_min + c_max) / 2
                c = c_new

            if abs(c_max - c_min) < 1e-6:
                return c, False

        return c, False

    def _failed(self, c: float, phi: float) -> bool:
        for f in self.fibers:
            eps = phi * (f.y - c)
            if isinstance(f.material, Concrete):
                if eps < -self.concrete.ecu:
                    return True
            elif isinstance(f.material, Steel):
                if abs(eps) > self.steel.epsu:
                    return True
        return False

    def pm_interaction(self, n_points: int = 30) -> tuple:
        """Compute P-M interaction diagram."""
        N_max_comp = (self.concrete.fc * self.b * self.h +
                      self.steel.fy * (self.As_top + self.As_bot))
        N_max_tens = self.steel.fy * (self.As_top + self.As_bot)
        N_vals = np.linspace(-0.9 * N_max_tens, 0.85 * N_max_comp, n_points)
        moments = []
        P_vals = []
        for N_try in N_vals:
            phis, Ms, _, _, _ = self.moment_curvature(N_try)
            if len(Ms) > 0:
                idx = np.argmax(np.abs(Ms))
                moments.append(Ms[idx])
                P_vals.append(N_try)
        return np.array(P_vals), np.array(moments)
