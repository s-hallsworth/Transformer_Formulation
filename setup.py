from setuptools import setup, find_packages

# Function to parse requirements.txt
def parse_requirements(filename):
    with open(filename, "r") as file:
        return [line.strip() for line in file if line.strip() and not line.startswith("#")]

setup(
    name="MINLP_tnn",
    version="0.1",
    description="A MINLP optimisation-based formulation of a trained transformer neural network",
    author="S Hallsworth",
    packages=find_packages(),
    install_requires=parse_requirements("requirements_tnn.txt"),
    python_requires=">=3.10",
)
