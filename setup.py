from setuptools import setup, find_packages

reqs = [
    "numpy>=1.18.2",
    "networkx>=3.0",
    "matplotlib>=1.4.3",
    "seaborn>=0.9.0",
    "pandas>=1.0.3",
    "IPython>=7.19.0",
]

setup(
    name="gnar",
    version="1.0.0",
    description="Genaralised Network Autoregressive Processes",
    author="Henry Antonio Palasciano",
    author_email="h.palasciano17@imperial.ac.uk",
    packages=find_packages(),
    install_requires=reqs,
)