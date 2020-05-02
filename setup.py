from setuptools import find_packages, setup


setup(
    name="Prostate Cancer Grade Assessment",
    author="Peter Todorov, Denis Diackov",
    version="1.0",
    description="Detecting prostate cancer grade",
    install_requires=list(open("requirements.txt").readlines()),
    packages=find_packages(),
)
