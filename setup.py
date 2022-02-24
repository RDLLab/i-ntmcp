"""I-NTMCP package install info """
from setuptools import setup, find_packages


extras = {
    'tree_vis': [
        'pygraphviz>=1.7'
    ],
}

extras['all'] = [item for group in extras.values() for item in group]


setup(
    name='intmcp',
    version='0.0.1',
    url="https://github.com/RDLLab/i-ntmcp",
    description="Interactive Nested Tree Monte Carlo Planning.",
    long_description_content_type='text/x-rst',
    author="Jonathon Schwartz",
    author_email="Jonathon.Schwartz@anu.edu.au",
    license="MIT",
    packages=[
        package for package in find_packages()
        if package.startswith('intmcp')
    ],
    install_requires=[
        'numpy>=1.20',
        'matplotlib>=3.4',
        'networkx>=2.5',
        'pandas>=1.2',
        'pyyaml>=5.4',
        'prettytable>=2.2'
    ],
    extras_require=extras,
    python_requires='>=3.9',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
    ],
    zip_safe=False
)
