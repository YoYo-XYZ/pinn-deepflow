from setuptools import setup, find_packages

with open("README.md", "r") as f:
    description = f.read()

setup(
    name="deepflow",
    version="0.0.1",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    long_description=description,
    long_description_content_type="text/markdown"
)