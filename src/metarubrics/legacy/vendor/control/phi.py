#!/usr/bin/env python3
"""Outer parameters and their mapping to criterion rewards.

    reward_i = sign_i * anchor(class_i) * exp(tau[class_i, label_i]) + delta_i

The outer state contains only tau and delta. Sign, class, and anchor remain fixed.
"""
from __future__ import annotations
import math, os

# The anchor table is dataset-specific. A compressed table can be supplied
# through ACRE_ANCHOR_TABLE when rubric scores occupy a narrow numeric range.
_DEFAULT_ANCHORS = (1.0, 2.0, 4.0, 8.0)
_env = os.environ.get("ACRE_ANCHOR_TABLE", "").strip()
_a = tuple(float(x) for x in _env.split(",")) if _env else _DEFAULT_ANCHORS
if len(_a) != 4 or any(_a[i] >= _a[i + 1] for i in range(3)) or _a[0] <= 0:
    raise ValueError(f"ACRE_ANCHOR_TABLE must contain four increasing positive numbers: {_env!r}")
ANCHOR = {"not_applicable": 0.0, "nice_to_have": _a[0], "should_have": _a[1],
          "must_have": _a[2], "contraindication": _a[3]}
CLASSES = ["not_applicable", "nice_to_have", "should_have", "must_have", "contraindication"]
LABELS = ["INVARIANT", "TARGET_CHANGE", "WEIGHT_CHANGE", "DROPPED", "ADDED"]
EPS_DELTA = 0.4          # Less than half the minimum default anchor interval.
# Bound tau by the smallest adjacent anchor ratio. This permits an effective
# magnitude to meet, but not pass, the next severity anchor. Deriving the cap
# from the active table also preserves this interpretation for compressed
# anchor tables; a fixed ln(2) cap would be too wide for such tables.
TAU_CAP = math.log(min(_a[i + 1] / _a[i] for i in range(3)))


class Phi:
    def __init__(self, tau=None, delta=None):
        self.tau = dict(tau or {})            # (class, label) -> float
        self.delta = dict(delta or {})        # criterion key -> float

    def get_tau(self, cls, label):
        return self.tau.get((cls, label), 0.0)

    def points(self, sign, cls, label, key=None):
        base = math.copysign(1.0, sign) * ANCHOR[cls] * math.exp(self.get_tau(cls, label))
        d = max(-EPS_DELTA, min(EPS_DELTA, self.delta.get(key, 0.0))) if key else 0.0
        return base + math.copysign(1.0, sign) * d

    # Enforce severity ordering and a fixed channel total.
    def project(self):
        """Apply isotonic projection and preserve the channel total.

        Within each label, anchor(k) * exp(tau) must increase with severity.
        The zero-mean constraint allows redistribution across cells without
        globally rescaling rewards, which would act like a learning-rate change.
        """
        for _ in range(8):        # Alternating constraints converge approximately.
            self._isotonic()
            for k in self.tau:
                self.tau[k] = max(-TAU_CAP, min(TAU_CAP, self.tau[k]))
            if self.tau:
                m = sum(self.tau.values()) / len(self.tau)
                if abs(m) < 1e-12:
                    break
                for k in self.tau:
                    self.tau[k] -= m
        for k in self.tau:
            self.tau[k] = max(-TAU_CAP, min(TAU_CAP, self.tau[k]))
        for k in self.delta:
            self.delta[k] = max(-EPS_DELTA, min(EPS_DELTA, self.delta[k]))
        return self

    def _isotonic(self):
        for label in LABELS:
            prev = None
            for cls in CLASSES:
                a = ANCHOR[cls]
                if a <= 0:
                    continue
                key = (cls, label)
                if key not in self.tau:
                    continue
                v = a * math.exp(self.tau[key])
                if prev is not None and v <= prev:
                    self.tau[key] = math.log((prev * 1.0001) / a)
                prev = max(v, prev * 1.0001 if prev is not None else v)

    def copy(self):
        return Phi(dict(self.tau), dict(self.delta))

    def cells(self):
        return sorted(self.tau)

    def __repr__(self):
        body = ", ".join(f"{c}/{l}={v:+.3f}" for (c, l), v in sorted(self.tau.items()))
        return f"Phi({body})"
