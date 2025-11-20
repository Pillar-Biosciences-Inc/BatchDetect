from setuptools import setup, find_packages

setup(
    name='batchdetect',
    version='0.1.0',
    description='Detecting Batch Heterogeneity',
    author='Austin Talbot',
    author_email='talbota@pillarbiosci.com',
    url='https://github.com/Pillar-Biosciences-Inc/BatchDetect',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'scipy',
        'tqdm',
        'cloudpickle',
        'matplotlib',
        'pytest',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
    ],
    python_requires='>=3.10',
)

