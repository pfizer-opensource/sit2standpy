import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pysit2stand",
    version="2019.09",
    author="Lukas Adamowicz",
    author_email="lukas.adamowicz@pfizer.com",
    description="Sit-to-stand detection using a single lumbar-mounted accelerometer.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PfizerRD/pysit2stand",
    include_pacakge_data=True,
    packages=setuptools.find_packages(),
    license='MIT',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
    ],
)
