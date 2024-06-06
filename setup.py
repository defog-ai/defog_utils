from setuptools import setup, find_packages

setup(
    name="defog_utils",
    version="0.1.0",
    description="Various helper functions used by Defog projects",
    author="defog",
    author_email="support@defog.ai",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "sqlglot",
        "sqlparse",
    ],
)
