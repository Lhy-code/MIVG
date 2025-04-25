from setuptools import setup, find_packages

setup(
    name="mivg",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "roboticstoolbox-python",
        "swift-simulator",
        "spatialmath-python",
        "spatialgeometry",
        "numpy",
        "matplotlib",
        "qpsolvers",
    ],
    author="Hangyu Lin",
    author_email="202130120126@scut.edu.cn",
    description="Mode-Isolated Velocity-Guide Algorithm for Robot Obstacle Avoidance",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Lhy-code/MIVG",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)