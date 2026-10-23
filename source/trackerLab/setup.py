"""Install the focused TrackerLab core; use the root project for both packages."""

from setuptools import find_packages, setup

setup(
    name="trackerLab",
    version="0.2.0",
    packages=find_packages(),
    package_data={"trackerLab.algorithms.supermimic": ["LICENSE.upstream"]},
    python_requires=">=3.11,<3.12",
)
