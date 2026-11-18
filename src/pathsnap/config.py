"""Central configuration for the PathSnap core pipeline.

Every tunable number lives here so experiments can be logged as plain JSON.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class CandidateConfig:
    radius: float = 50.0        # metres: arcs closer than this become candidates
    k: int = 8                  # max candidates per GPS observation
    sample_step: float = 10.0   # metres between KD-tree samples along each arc
    fallback_k: int = 3         # if nothing is within `radius`, take this many nearest arcs


@dataclass
class FeatureConfig:
    heading_window: int = 2     # GPS heading from point i-w to i+w
    context_window: int = 3     # neighbour points used for context-distance features
    long_window: int = 10       # long baseline for direction features (robust to GPS jitter)
    min_heading_disp: float = 3.0  # metres moved before the heading is trusted
    label_max_dist: float = 50.0   # obs farther than this from its aligned route arc = unreliable label
    trip_max_median_label_dist: float = 30.0  # whole trip excluded from training above this


@dataclass
class HMMConfig:
    # emissions
    w_cls: float = 1.0          # weight on log classifier probability
    w_dist: float = 0.0         # weight on Gaussian distance term when a classifier is used
    sigma: float = 10.0         # GPS noise std (m) for the Gaussian distance term
    # transitions (Newson & Krumm style: |route_dist - gps_dist| ~ Exponential(beta))
    beta: float = 5.0
    gap_beta_scale: float = 0.2  # beta grows with time gap: beta * (1 + scale*(dt-1))
    back_tol: float = 30.0      # metres of backward movement on the same arc tolerated (GPS jitter)
    back_penalty: float = 1.0   # backward jitter costs (1 + penalty) x its length in route distance
    vmax: float = 50.0          # m/s, faster implied route speeds are infeasible
    route_factor: float = 3.0   # Dijkstra search limit = factor * gps_dist + slack
    route_slack: float = 200.0
    cache_limit: float = 1500.0  # shortest-path rows computed with this limit are cached
    eps: float = 1e-6


@dataclass
class Config:
    cand: CandidateConfig = field(default_factory=CandidateConfig)
    feat: FeatureConfig = field(default_factory=FeatureConfig)
    hmm: HMMConfig = field(default_factory=HMMConfig)
    undirected: bool = False    # treat arcs as two-way (set after checking the data!)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        return cls(
            cand=CandidateConfig(**d.get("cand", {})),
            feat=FeatureConfig(**d.get("feat", {})),
            hmm=HMMConfig(**d.get("hmm", {})),
            undirected=d.get("undirected", False),
        )

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        return cls.from_dict(json.loads(Path(path).read_text()))
