from setuptools import setup, find_namespace_packages


APP_NAME = "mlflow_vega"
VERSION = "1.0.0"
LICENSE = "MIT"
AUTHOR = "James Hibbard, Andrew Bauman"
DESCRIPTION = (
    "MLflow model flavour for mlflow_vega visualization grammar"
)

setup(
    name=APP_NAME,
    version=VERSION,
    license=LICENSE,
    author=AUTHOR,
    description=DESCRIPTION,
    install_requires=[
        'mlflow==1.10.0',
        'PyYAML==5.3.1',
        'altair==4.1.0',
        'altair_viewer',
    ],
    extras_require={
    },
    packages=find_namespace_packages("src"),
    package_dir={"": "src"},
    package_data={"": ["*.yaml"], },
    entry_points="""
    [console_scripts]
    """,
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
    ],
)
