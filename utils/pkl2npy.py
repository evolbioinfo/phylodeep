import os
import pickle as pk
import numpy as np
from sklearn.externals import joblib

PRETRAINED_MODELS_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'pretrained_models')
PRETRAINED_PCA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'pca_a_priori')

for model in ('BDEI', 'BD', 'BDSS'):
    for tree_size in ('LARGE', 'SMALL'):
        if 'BDSS' == model and 'SMALL' == tree_size:
            continue
        pattern = os.path.join(PRETRAINED_PCA_DIR, 'models', model + '_' + tree_size + '_PCA')
        with open(pattern + '.pkl', 'rb') as f:
            pca_reload = pk.load(f)
        np.save(pattern + '_components', pca_reload.components_, allow_pickle=False)
        np.save(pattern + '_variance', pca_reload.explained_variance_, allow_pickle=False)
        np.save(pattern + '_variance_ratio', pca_reload.explained_variance_ratio_, allow_pickle=False)
        np.save(pattern + '_mean', pca_reload.mean_, allow_pickle=False)
        np.save(pattern + '_singular_values', pca_reload.singular_values_, allow_pickle=False)

        pattern = os.path.join(PRETRAINED_PCA_DIR, 'scalers', model + '_' + tree_size + '_PCA')
        scaler = joblib.load(pattern + '.pkl')
        np.save(pattern + '_mean', scaler.mean_, allow_pickle=False)
        np.save(pattern + '_scale', scaler.scale_, allow_pickle=False)
        np.save(pattern + '_var', scaler.var_, allow_pickle=False)

        pattern = os.path.join(PRETRAINED_MODELS_DIR, 'scalers', model + '_' + tree_size + '_FFNN')
        scaler = joblib.load(pattern +'.pkl')
        np.save(pattern + '_mean', scaler.mean_, allow_pickle=False)
        np.save(pattern + '_scale', scaler.scale_, allow_pickle=False)
        np.save(pattern + '_var', scaler.var_, allow_pickle=False)

for model in ('BD_vs_BDEI_vs_BDSS_LARGE', 'BD_vs_BDEI_SMALL'):
    pattern = os.path.join(PRETRAINED_MODELS_DIR, 'scalers', model + '_' + 'FFNN')
    scaler = joblib.load(pattern +'.pkl')
    np.save(pattern + '_mean', scaler.mean_, allow_pickle=False)
    np.save(pattern + '_scale', scaler.scale_, allow_pickle=False)
    np.save(pattern + '_var', scaler.var_, allow_pickle=False)
