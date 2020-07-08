import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

fid = open('sit2standpy/version.py')
vers = fid.readlines()[-1].split()[-1].strip("\"'")
fid.close()

setuptools.setup(
    name="sit2standpy",
    version=vers,
    author="Lukas Adamowicz",
    author_email="lukas.adamowicz@pfizer.com",
    description="Sit-to-stand detection using a single lumbar-mounted accelerometer.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PfizerRD/sit2standpy",
    download_url="https://pypi.org/project/sit2standpy/",
    project_urls={
        "Documentation": "https://sit2standpy.readthedocs.io/en/latest/"
    },
    include_pacakge_data=True,
    package_data={'sit2standpy': ['data/**']},
    packages=setuptools.find_packages(),
    license='MIT',
    python_requires='>=3.7',
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'pywavelets',
        'udatetime'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Programming Language :: Python :: 3.7"
    ],
)
