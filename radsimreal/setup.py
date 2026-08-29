from setuptools import setup, find_packages
setup(
    name="radsimreal",
    version="0.1.0",
    description="RadSimReal: PSF-convolution radar simulation (Bialer & Haitman, CVPR 2024) - unofficial implementation",
    packages=find_packages(),
    install_requires=["numpy>=1.21", "scipy>=1.7", "matplotlib>=3.4"],
    python_requires=">=3.9",
    entry_points={"console_scripts": ["radsimreal=radsimreal.cli.main:main"]},
)
