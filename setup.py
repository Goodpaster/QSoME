import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="QSoME",
    version='0.6.0',
    author="Daniel Graham",
    author_email="graha682@umn.edu",
    description="Quantum Solid state Method Embedding",
    long_description=long_description,
    url="https://github.com/Goodpaster/QSoME",
    packages=setuptools.find_packages(),
    install_requires=[
        'pyscf',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
