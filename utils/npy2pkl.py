
import os
import pickle as pk

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

PRETRAINED_MODELS_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'pretrained_models')
PRETRAINED_PCA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'pca_a_priori')


def load_scaler(tree_size, model, dir, suffix='PCA', n_samples_seen=4000000):
    scaler = StandardScaler()
    scaler.mean_ = np.load(os.path.join(dir, 'scalers', '{}_{}_{}_{}.npy'.format(model, tree_size, suffix, 'mean')))
    scaler.scale_ = np.load(os.path.join(dir, 'scalers', '{}_{}_{}_{}.npy'.format(model, tree_size, suffix, 'scale')))
    scaler.var_ = np.load(os.path.join(dir, 'scalers', '{}_{}_{}_{}.npy'.format(model, tree_size, suffix, 'var')))
    scaler.n_samples_seen_ = n_samples_seen
    return scaler


for model in ('BDEI', 'BD', 'BDSS'):
    for tree_size in ('LARGE', 'SMALL'):
        if 'BDSS' == model and 'SMALL' == tree_size:
            continue
        pattern = os.path.join(PRETRAINED_PCA_DIR, 'models', model + '_' + tree_size + '_PCA')

        comps = np.load(pattern + '_components.npy')
        pca = PCA(n_components=len(comps))
        pca.components_ = comps
        pca.explained_variance_ = np.load(pattern + '_variance.npy')
        pca.explained_variance_ratio_ = np.load(pattern + '_variance_ratio.npy')
        pca.mean_ = np.load(pattern + '_mean.npy')
        pca.singular_values_ = np.load(pattern + '_singular_values.npy')

        with open(pattern + '.pkl', 'wb+') as f:
            pk.dump(pca, f)

        pattern = os.path.join(PRETRAINED_PCA_DIR, 'scalers', model + '_' + tree_size + '_PCA')
        scaler = load_scaler(tree_size, model, dir=PRETRAINED_PCA_DIR, suffix='PCA', n_samples_seen=10000)
        with open(pattern + '.pkl', 'wb+') as f:
            pk.dump(scaler, f)

        pattern = os.path.join(PRETRAINED_MODELS_DIR, 'scalers', model + '_' + tree_size + '_FFNN')
        scaler = load_scaler(tree_size, model, dir=PRETRAINED_MODELS_DIR, suffix='FFNN', n_samples_seen=4000000)
        with open(pattern + '.pkl', 'wb+') as f:
            pk.dump(scaler, f)


for model in ('BD_vs_BDEI_vs_BDSS_LARGE', 'BD_vs_BDEI_SMALL'):
    pattern = os.path.join(PRETRAINED_MODELS_DIR, 'scalers', model + '_' + 'FFNN')
    scaler = load_scaler(model.split('_')[-1], model[:model.rfind('_')], dir=PRETRAINED_MODELS_DIR,
                         suffix='FFNN', n_samples_seen=12000000)
    scaler.with_std = True
    with open(pattern + '.pkl', 'wb+') as f:
        pk.dump(scaler, f)

