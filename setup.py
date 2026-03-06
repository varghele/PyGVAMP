from setuptools import setup, find_packages

setup(
    name="pygv",
    version="0.1",
    packages=find_packages(),
    package_data={
        "pygv.visualization": [
            "templates/*.html",
            "templates/assets/*.css",
            "templates/assets/*.js",
        ]
    },
    include_package_data=True,
    install_requires=[
        "jinja2>=3.0.0",
    ],
)
