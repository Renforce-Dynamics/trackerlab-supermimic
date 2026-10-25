"""Compatibility import for the original G1 robot module."""

from .tracking.humanoid.robots.g1.robot import *
from .tracking.humanoid.robots.g1 import robot as _implementation


def __getattr__(name):
    return getattr(_implementation, name)
