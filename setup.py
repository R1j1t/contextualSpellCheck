import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="contextualSpellCheck",
    version="0.3.0",
    author="R1j1t",
    author_email="r1j1t@protonmail.com",
    description="Contextual spell correction using BERT (bidirectional representations)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/R1j1t/contextualSpellCheck",
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Development Status :: 4 - Beta",
    ],
    python_requires=">=3.6",
    install_requires=["torch", "editdistance", "transformers", "spacy"],
)
