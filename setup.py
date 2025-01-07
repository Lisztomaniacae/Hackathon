from setuptools import find_packages, setup

with open("requirements.txt", "r") as fh:
    requirements = fh.readlines()

setup(
    name="ProKI Hackathon 2024 Solution",
    version="4.2.0",
    author="whoami",
    description="Our solution for described problem",
    install_requires=[req for req in requirements if req[:2] != "# "],
    packages=find_packages(),
)
