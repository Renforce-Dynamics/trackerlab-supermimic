"""Backward-compatible G1 kinematics entry point."""

from trackerTask.supermimic.robots import get_robot
from .robot import RobotKinematics


class G1Kinematics(RobotKinematics):
    def __init__(self, device, tracked=None):
        super().__init__(device, get_robot("g1"), tracked)
