from setuptools import setup, find_packages

setup(
    name="pygv",
    version="0.9.0",
    description="Graph-based VAMPNet for MD trajectory analysis with PyTorch Geometric",
    packages=find_packages(),
    package_data={
        "pygv.visualization": [
            "templates/*.html",
            "templates/assets/*.css",
            "templates/assets/*.js",
        ]
    },
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "pandas",
        "scikit-learn",
        "tqdm",
        "joblib",
        "jinja2>=3.0.0",
        "mdtraj",
        "gensim",
        "pyyaml",
        "sympy",
    ],
    extras_require={
        "umap": ["umap-learn"],
        "test": ["pytest"],
    },
    entry_points={
        "console_scripts": [
            "pygvamp=pygv.pipe.master_pipeline:main",
        ],
    },
)