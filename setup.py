import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="stroll-jiskattema",
    version="0.0.1",
    author="Jisk Attema",
    author_email="j.attema@esciencecenter.nl",
    description="Graph based semantic role labeler",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Filter-Bubble/stroll",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)