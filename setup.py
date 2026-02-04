from setuptools import setup, find_packages

def get_requirements(filepath):
    with open(filepath) as req:
        requirements = req.readlines()
        return [requires.strip() for requires in requirements]

# print(get_requirements("requirements.txt"))

setup(
    name="Predictive Modeling for Cancer Risk Assessment Using Machine Learning",
    author="Abhishek Kumar",
    version="0.1",
    packages=find_packages(),
    install_requires=get_requirements()
)
