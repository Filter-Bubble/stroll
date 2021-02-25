import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

setuptools.setup(
    name="stroll-srl",
    version="0.5.1",
    author="Jisk Attema",
    author_email="j.attema@esciencecenter.nl",
    description="Graph based semantic role labeler",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Filter-Bubble/stroll",
    packages=setuptools.find_packages(),
    install_requires=required,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
