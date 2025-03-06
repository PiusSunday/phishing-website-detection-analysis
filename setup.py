from typing import List

from setuptools import find_packages, setup

HYPHEN_E_DOT = "-e ."


def get_requirements(file_path: str) -> List[str]:
    """
    this function will return the list of requirements
    """
    requirements = []

    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements


setup(
    # name="phishing-website-detection-analysis",
    # version="0.0.1",
    # author="Sunnythesage",
    # author_email="Sundaypius2000gmail.com",
    # install_requires=get_requirements("requirements.txt"),
    packages=find_packages(),
)

# To install the project as a package in the current environment, run:
# pip install .

# If you need to build the package for distribution, run:
# pip install build
# python -m build

# Then, to install the built package, run:
# pip install dist/*.whl
