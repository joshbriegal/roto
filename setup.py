import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="roto",
    version="0.0.1",
    author="Josh Briegal & Ed Gillen",
    author_email="jtb34@cam.ac.uk",
    description="One stop rotation finder tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/joshbriegal/roto",
    project_urls={
        "Bug Tracker": "https://github.com/joshbriegal/roto",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    install_requires=[
        "astropy",
        "gacf==1.0.0",
        "numpy",
        "peakutils==1.3.3",
        "pandas",
        "pymc3",
        "celerite2",
        "pymc3_ext",
        "corner",
        "matplotlib",
    ],
)
