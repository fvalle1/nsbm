import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="trisbm",
    version="0.0.1",
    author="Filippo Valle",
    author_email="filippo.valle@unito.it",
    description="Package to run topic models with keywords.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fvalle1/trisbm",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires = ["numpy", "pandas", "cloudpickle", "matplotlib"],
    python_requires='>=3.7',
)
