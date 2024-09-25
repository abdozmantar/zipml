from setuptools import setup, find_packages

setup(
    name='zipml',
    version='0.2.0',
    description='A simple AutoML tool for small datasets with useful helper functions',
    author='Abdullah OZMANTAR',
    author_email='abdullahozmntr@gmail.com',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown', 
    url='https://github.com/abdozmantar/zipml',
    packages=find_packages(),
    install_requires=[
        'pandas>=1.0.0', 
        'scikit-learn>=0.22', 
        'matplotlib>=3.0.0', 
        'seaborn>=0.9.0'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Topic :: Software Development :: Libraries',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6', 
    entry_points={
        'console_scripts': [
            'zipml=zipml.zipml:main',
        ],
    },
)
