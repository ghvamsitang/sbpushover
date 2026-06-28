import numpy as np


class Concrete:
    def __init__(self, fc: float, ecu: float = 0.0035, ec0: float = 0.002,
                 ft: float = None):
        self.fc = fc
        self.ecu = ecu
        self.ec0 = ec0
        self.Ec = 4700 * fc ** 0.5
        self.ft = ft if ft is not None else 0.33 * fc ** 0.5
        self.epst0 = self.ft / self.Ec
        self.epstu = 10 * self.epst0

    def stress(self, eps: float) -> float:
        """Stress (MPa). Compression: eps < 0 -> stress < 0. Tension: eps > 0 -> stress > 0."""
        if eps > 0:
            if eps <= self.epst0:
                return self.Ec * eps
            elif eps <= self.epstu:
                return self.ft * (self.epstu - eps) / (self.epstu - self.epst0)
            return 0.0
        eps_c = -eps
        if eps_c <= self.ec0:
            return -self.fc * (2 * eps_c / self.ec0 - (eps_c / self.ec0) ** 2)
        elif eps_c <= self.ecu:
            return -self.fc
        return 0.0

    def tangent(self, eps: float) -> float:
        """d(stress)/d(strain). Always >= 0 for numerical stability."""
        if eps > 0:
            if eps < self.epst0:
                return self.Ec
            return 0.0
        eps_c = -eps
        if eps_c < self.ec0:
            return self.fc * (2 / self.ec0 - 2 * eps_c / self.ec0 ** 2)
        return 0.0


class Steel:
    def __init__(self, fy: float, Es: float = 200e3, epsu: float = 0.05):
        self.fy = fy
        self.Es = Es
        self.epsu = epsu
        self.epsy = fy / Es

    def stress(self, eps: float) -> float:
        if abs(eps) <= self.epsy:
            return self.Es * eps
        if eps > 0:
            return self.fy
        return -self.fy

    def tangent(self, eps: float) -> float:
        if abs(eps) <= self.epsy:
            return self.Es
        return 0.0
