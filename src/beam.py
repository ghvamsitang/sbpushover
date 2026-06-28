import numpy as np
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d
from typing import Tuple
from .section import RCSection
from dataclasses import dataclass


@dataclass
class PushoverResult:
    displacements: np.ndarray
    axial_forces: np.ndarray
    max_moments: np.ndarray
    midspan_deflections: np.ndarray
    failure_mode: str
    failure_index: int
    converged: bool
    moment_profiles: list = None
    strain_profiles: list = None
    x_positions: np.ndarray = None


class Beam:
    def __init__(self, section: RCSection, span_mm: float, w_Nmm: float = 0.0):
        self.section = section
        self.L = span_mm
        self.w = w_Nmm

    def initial_deflection(self, n_seg: int = 100) -> np.ndarray:
        x = np.linspace(0, self.L, n_seg + 1)
        dx = self.L / n_seg
        M = self.w * x * (self.L - x) / 2
        Ec = self.section.concrete.Ec
        Ig = self.section.b * self.section.h ** 3 / 12
        EI = 0.35 * Ec * Ig
        phi = M / EI
        v = self._integrate_curvature(phi, dx)
        return v

    def _integrate_curvature(self, phi: np.ndarray, dx: float) -> np.ndarray:
        """Integrate curvature twice to get deflection.
        v'' = -φ. Simply supported: v(0) = v(L) = 0.
        """
        n = len(phi)
        theta = np.zeros(n)
        for i in range(1, n):
            theta[i] = theta[i - 1] - 0.5 * (phi[i - 1] + phi[i]) * dx

        v_raw = np.zeros(n)
        for i in range(1, n):
            v_raw[i] = v_raw[i - 1] + 0.5 * (theta[i - 1] + theta[i]) * dx

        L = dx * (n - 1)
        C1 = -v_raw[-1] / L
        v = v_raw + C1 * np.linspace(0, L, n)
        return v


class PushoverAnalysis:
    def __init__(self, beam: Beam, n_seg: int = 50, n_steps: int = 100):
        self.beam = beam
        self.n_seg = n_seg
        self.n_steps = n_steps

    def run(self) -> PushoverResult:
        sec = self.beam.section
        L = self.beam.L
        w = self.beam.w
        n = self.n_seg
        dx = L / n
        x = np.linspace(0, L, n + 1)

        P_tensile_cap = (sec.steel.fy * (sec.As_top + sec.As_bot) +
                         sec.concrete.ft * sec.b * sec.h)
        dP_step = P_tensile_cap / self.n_steps

        P_vals = []
        disp_vals = []
        M_max_vals = []
        mid_v_vals = []
        M_profile_vals = []
        eps0_profile_vals = []
        failure_mode = "None"
        fail_idx = -1

        P = 0.0

        mc_cache = {}
        M_sw = w * x * (L - x) / 2

        # Initial state at P=0 using the M-φ curve (accounts for cracking
        # from the distributed load alone)
        phi_c0, M_c0, _, _, eps0_c0 = sec.moment_curvature(0.0)
        if len(phi_c0) == 0:
            failure_mode = "Initial section analysis failed"
            fail_idx = 0
            return PushoverResult(
                displacements=np.array([]), axial_forces=np.array([]),
                max_moments=np.array([]), midspan_deflections=np.array([]),
                failure_mode=failure_mode, failure_index=fail_idx,
                converged=False, moment_profiles=[], strain_profiles=[], x_positions=x)
        mc_cache[0.0] = (phi_c0, M_c0, eps0_c0)

        interp0 = interp1d(M_c0, phi_c0, kind='linear',
                           bounds_error=False, fill_value=(phi_c0[0], phi_c0[-1]))
        phi_init = np.maximum(interp0(M_sw), 0)
        v = self.beam._integrate_curvature(phi_init, dx)

        interp0_eps = interp1d(M_c0, eps0_c0, kind='linear',
                               bounds_error=False,
                               fill_value=(eps0_c0[0], eps0_c0[-1]))
        eps0_init = interp0_eps(M_sw)
        axial_disp_init = trapezoid(eps0_init, x)

        # Record initial step at P=0
        P_vals.append(0.0)
        disp_vals.append(axial_disp_init)
        M_max_vals.append(np.max(np.abs(M_sw)))
        mid_v_vals.append(v[n // 2])
        M_profile_vals.append(M_sw.copy())
        eps0_profile_vals.append(eps0_init.copy())

        for step in range(1, self.n_steps + 1):
            P = -step * dP_step
            if P < -0.85 * P_tensile_cap:
                P = -0.85 * P_tensile_cap

            if P not in mc_cache:
                phi_c, M_c, _, _, eps0_c = sec.moment_curvature(P)
                if len(phi_c) == 0:
                    failure_mode = "Maximum axial tension capacity reached"
                    fail_idx = len(P_vals) - 1
                    break
                mc_cache[P] = (phi_c, M_c, eps0_c)
            else:
                phi_c, M_c, eps0_c = mc_cache[P]
            M_ult_capacity = np.max(np.abs(M_c)) if len(M_c) > 0 else 0

            if len(M_c) < 2:
                failure_mode = "Maximum axial tension capacity reached"
                fail_idx = len(P_vals) - 1
                break

            try:
                interp_M_phi = interp1d(M_c, phi_c, kind='linear',
                                        bounds_error=False,
                                        fill_value=(phi_c[0], phi_c[-1]))
                interp_M_eps0 = interp1d(M_c, eps0_c, kind='linear',
                                         bounds_error=False,
                                         fill_value=(eps0_c[0], eps0_c[-1]))
            except Exception:
                failure_mode = "Interpolation failed"
                fail_idx = len(P_vals)
                break

            v_new = v.copy()
            pd_converged = False
            max_M_seen = 0.0
            for iteration in range(50):
                M_total = M_sw + P * v_new
                max_M_seen = max(max_M_seen, np.max(np.abs(M_total)))

                if max_M_seen > M_ult_capacity * 1.05:
                    failure_mode = "Section flexural failure (P-M interaction)"
                    fail_idx = len(P_vals)
                    break

                phi_at_x = interp_M_phi(M_total)
                phi_at_x = np.maximum(phi_at_x, 0)

                v_new2 = self.beam._integrate_curvature(phi_at_x, dx)
                dv = np.max(np.abs(v_new2 - v_new))
                v_new = 0.7 * v_new + 0.3 * v_new2
                if dv < 1e-5:
                    pd_converged = True
                    break

            if fail_idx >= 0:
                break

            if not pd_converged:
                if max_M_seen > M_ult_capacity * 0.95:
                    failure_mode = "Section flexural failure (near capacity)"
                else:
                    failure_mode = "P-Delta iteration did not converge"
                fail_idx = len(P_vals)
                break

            eps0_at_x = interp_M_eps0(M_total)
            axial_disp = trapezoid(eps0_at_x, x)

            v = v_new
            P_vals.append(P)
            disp_vals.append(axial_disp)
            M_max_vals.append(np.max(np.abs(M_total)))
            mid_v_vals.append(v[n // 2])
            M_profile_vals.append(M_total.copy())
            eps0_profile_vals.append(eps0_at_x.copy())

            if np.max(np.abs(M_total)) > M_ult_capacity * 1.05:
                failure_mode = "Section flexural failure"
                fail_idx = len(P_vals) - 1
                break

        return PushoverResult(
            displacements=np.array(disp_vals),
            axial_forces=np.array(P_vals),
            max_moments=np.array(M_max_vals),
            midspan_deflections=np.array(mid_v_vals),
            failure_mode=failure_mode,
            failure_index=fail_idx,
            converged=(fail_idx < 0 or fail_idx == len(P_vals) - 1),
            moment_profiles=M_profile_vals,
            strain_profiles=eps0_profile_vals,
            x_positions=x,
        )
