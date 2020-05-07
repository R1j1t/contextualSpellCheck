import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="contextualSpellCheck",
    version="0.0.1",
    author="R1j1t",
    author_email="r1j1t@protonmail.com",
    description="Contextual spell correction using BERT (bidirectional representations)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/R1j1t/contextualSpellCheck",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
