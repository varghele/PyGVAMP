"""Setup script for MD Visualizer package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "md_visualizer" / "requirements.txt"
with open(requirements_file, "r", encoding="utf-8") as f:
    requirements = [
        line.strip() for line in f
        if line.strip() and not line.startswith("#") and not line.startswith("--")
    ]

setup(
    name="md-visualizer",
    version="0.1.0",
    author="PyGVAMP Contributors",
    author_email="",
    description="Interactive 3D visualization toolkit for molecular dynamics trajectory analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/md-visualizer",
    packages=find_packages(),
    package_data={
        "md_visualizer": [
            "templates/*.html",
            "templates/assets/*.css",
            "templates/assets/*.js",
        ]
    },
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Visualization",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "jinja2>=3.0.0",
    ],
    extras_require={
        "mdtraj": ["mdtraj>=1.9.0"],
        "mdanalysis": ["MDAnalysis>=2.0.0"],
        "graph": ["networkx>=2.6.0"],
        "all": [
            "mdtraj>=1.9.0",
            "networkx>=2.6.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "md-visualizer-demo=md_visualizer.examples.generate_mock_data:main",
        ],
    },
)
