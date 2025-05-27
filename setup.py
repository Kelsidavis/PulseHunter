#!/usr/bin/env python3
"""
PulseHunter: Optical SETI & Exoplanet Detection Network
Setup script for package installation
"""

from setuptools import find_packages, setup

# Read the README file for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip() for line in fh if line.strip() and not line.startswith("#")
    ]

setup(
    name="pulsehunter",
    version="0.1.0",
    author="Kelsi Davis",
    author_email="pulsehunter@geekastro.dev",
    description="Optical SETI & Exoplanet Detection Network for Amateur Astronomers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Kelsidavis/PulseHunter",
    project_urls={
        "Bug Tracker": "https://github.com/Kelsidavis/PulseHunter/issues",
        "Documentation": "https://kelsidavis.github.io/PulseHunter/",
        "Source Code": "https://github.com/Kelsidavis/PulseHunter",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "flake8>=5.0.0",
            "black>=22.0.0",
            "mypy>=0.991",
        ],
    },
    entry_points={
        "console_scripts": [
            "pulsehunter=pulse_gui:main",
        ],
    },
    include_package_data=True,
    package_data={
        "pulsehunter": ["*.ui", "icons/*", "docs/*"],
    },
    keywords="astronomy seti exoplanet detection citizen-science astrophotography",
    zip_safe=False,
)
