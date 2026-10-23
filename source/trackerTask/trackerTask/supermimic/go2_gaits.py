"""Reproducible synthetic gait specifications; all quantities use SI units."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Gait:
    name: str
    speed: float
    period: float = 0.5
    duty_factor: float = 0.6
    clearance: float = 0.05
    root_height: float = 0.30
    duration: float = 20.0
    lateral_foot_position: float = 0.165
    forward_foot_offset: float = 0.02


GAITS = (
    Gait("trot_035", 0.35),
    Gait("trot_050", 0.50),
    Gait("trot_065", 0.65),
    Gait("trot_080", 0.80),
)
