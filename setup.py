import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="trisbm",
    version="0.1.0",
    author="Filippo Valle",
    author_email="filippo.valle@unito.it",
    description="Package to run stochastic block model on tri-partite networks.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fvalle1/trisbm",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires = ["pandas", "numpy", "graph-tool", "cloudpickle", "matplotlib"],
    python_requires='>=3.7',
)
