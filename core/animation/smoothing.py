from __future__ import annotations

import math
import time


class OneEuroFilter:
    def __init__(
        self,
        min_cutoff: float = 1.5,
        beta: float = 0.3,
        d_cutoff: float = 1.0,
    ) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x_prev: float | None = None
        self._dx_prev: float = 0.0
        self._t_prev: float | None = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: float, t: float | None = None) -> float:
        t = time.perf_counter() if t is None else t
        if self._t_prev is None or self._x_prev is None:
            self._t_prev = t
            self._x_prev = x
            return x
        dt = max(t - self._t_prev, 1e-6)
        dx = (x - self._x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self._dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self._x_prev
        self._t_prev = t
        self._x_prev = x_hat
        self._dx_prev = dx_hat
        return x_hat

    def reset(self) -> None:
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None


class VectorSmoother:
    def __init__(self, dim: int, **kwargs: float) -> None:
        self.filters = [OneEuroFilter(**kwargs) for _ in range(dim)]

    def __call__(self, values: list[float], t: float | None = None) -> list[float]:
        return [f(v, t) for f, v in zip(self.filters, values)]

    def reset(self) -> None:
        for f in self.filters:
            f.reset()


class QuaternionSmoother:
    def __init__(self, beta: float = 0.2) -> None:
        self.beta = beta
        self._prev: list[float] | None = None

    def __call__(self, q: list[float]) -> list[float]:
        if self._prev is None:
            self._prev = list(q)
            return list(q)
        dot = sum(a * b for a, b in zip(q, self._prev))
        sign = 1.0 if dot >= 0 else -1.0
        out = [
            self.beta * b + (1.0 - self.beta) * (sign * a)
            for a, b in zip(self._prev, q)
        ]
        norm = math.sqrt(sum(c * c for c in out)) or 1.0
        out = [c / norm for c in out]
        self._prev = out
        return out

    def reset(self) -> None:
        self._prev = None
