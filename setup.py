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
                                'README.md']},
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Programming Language :: Python :: 3'
    ],
    version='0.2.59',
    description='Phylodynamic paramater and model inference using pretrained deep neural networks.',
    author='Jakub Voznica',
    author_email='jakub.voznica@pasteur.fr',
    maintainer='Anna Zhukova',
    maintainer_email='anna.zhukova@pasteur.fr',
    url='https://github.com/evolbioinfo/pastml',
    keywords=['phylodynamics', 'molecular epidemiology', 'phylogeny', 'model selection',
              'paramdeep', 'phylodeep', 'deep learning', 'convolutional networks'],
    install_requires=['ete3', 'pandas', 'numpy', 'scipy==1.1.0', 'scikit-learn==0.19.1',
                      'tensorflow==1.13.1', 'joblib==0.13.2', 'h5py==2.10.0', 'Keras==2.4.3', 'matplotlib==3.1.3',
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
