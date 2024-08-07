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
                                os.path.join('..', 'README_PYPI.md'), os.path.join('..', 'README.md'),
                                os.path.join('..', 'LICENCE')]},
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
        'Programming Language :: Python :: 3 :: Only',
    ],
    version='0.6',
    description='Phylodynamic paramater and model inference using pretrained deep neural networks.',
    author='Jakub Voznica',
    author_email='jakub.voznica@pasteur.fr',
    maintainer='Anna Zhukova',
    maintainer_email='anna.zhukova@pasteur.fr',
    url='https://github.com/evolbioinfo/phylodeep',
    keywords=['phylodynamics', 'molecular epidemiology', 'phylogeny', 'model selection',
              'paramdeep', 'phylodeep', 'deep learning', 'convolutional networks'],
    python_requires='>=3.8',
    install_requires=['scikit-learn>=1.1.3,<1.3.0',
                      'tensorflow==2.5.3', # this tensorflow is compatible with the keras version below
                      'ete3>=3.1.1,<=3.1.3', 'pandas>=1.0.0,<1.4.0', 'matplotlib>=3.6.0,<3.7.0',
                      'keras==2.2.4', # keras must be this version as the models are encoded with it
                      'phylodeep_data_bd>=0.6', 'phylodeep_data_bdei>=0.4', 'phylodeep_data_bdss>=0.4',
                      # 'scipy>=1.5.0,<1.10.0', 'numpy~=1.19.2' # these are already installed via other libraries
                      ],
    entry_points={
            'console_scripts': [
                'checkdeep = phylodeep.checkdeep:main',
                'modeldeep = phylodeep.modeldeep:main',
                'paramdeep = phylodeep.paramdeep:main',
                'subtree_picker = phylodeep.tree_utilities:subtree_picker_main'
            ]
    },
)
