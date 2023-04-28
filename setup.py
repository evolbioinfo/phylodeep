import os
from setuptools import setup, find_packages

setup(
    name='phylodeep',
    packages=find_packages(),
    include_package_data=True,
    package_data={'phylodeep': [os.path.join('pca_a_priori', 'models', '*.pkl'),
                                os.path.join('pca_a_priori', 'scalers', '*.pkl'),
                                os.path.join('pca_a_priori', 'simulated_data', '*.csv'),
                                os.path.join('pretrained_models', 'models', '*.json'),
                                os.path.join('pretrained_models', 'scalers', '*.pkl'),
                                os.path.join('pretrained_models', 'weights', '*.h5'),
                                'README_PYPI.md']},
    long_description=open('README_PYPI.md').read(),
    long_description_content_type='text/markdown',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3 :: Only',
    ],
    version='0.3.1',
    description='Phylodynamic paramater and model inference using pretrained deep neural networks.',
    author='Jakub Voznica',
    author_email='jakub.voznica@pasteur.fr',
    maintainer='Anna Zhukova',
    maintainer_email='anna.zhukova@pasteur.fr',
    url='https://github.com/evolbioinfo/phylodeep',
    keywords=['phylodynamics', 'molecular epidemiology', 'phylogeny', 'model selection',
              'paramdeep', 'phylodeep', 'deep learning', 'convolutional networks'],
    python_requires='>=3.8',
    install_requires=['ete3>=3.1.1', 'pandas>=1.0.0', 'numpy>=1.22', 'scipy>=1.5.0',
                      'scikit-learn>=1.1.3', 'tensorflow>=2.10.0', 'h5py>=3.0.0', 'Keras>=2.11.0', 'matplotlib>=3.6.0',
                      'phylodeep-data-BD-small', 'phylodeep-data-BD-large>=0.0.2',
                      'phylodeep-data-BDEI-small', 'phylodeep-data-BDEI-large',
                      'phylodeep-data-BDSS-large'],
    entry_points={
            'console_scripts': [
                'checkdeep = phylodeep.checkdeep:main',
                'modeldeep = phylodeep.modeldeep:main',
                'paramdeep = phylodeep.paramdeep:main',
                'subtree_picker = phylodeep.tree_utilities:subtree_picker_main'
            ]
    },
)
