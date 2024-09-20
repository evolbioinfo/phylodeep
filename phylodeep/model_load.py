import os
import pickle as pk
import warnings

import numpy as np
from tensorflow import keras

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

PRETRAINED_MODELS_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'pretrained_models')
PRETRAINED_PCA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'pca_a_priori')


def pca_scale_load(tree_size, model):
    with open(os.path.join(PRETRAINED_PCA_DIR, 'models', '{}_{}_PCA.pkl'.format(model, tree_size)), 'rb') as f:
        pca_reload = pk.load(f)

    with open(os.path.join(PRETRAINED_PCA_DIR, 'scalers', '{}_{}_PCA.pkl'.format(model, tree_size)), 'rb') as f:
        scaler = pk.load(f)

    return pca_reload, scaler


def pca_data_load(tree_size, model):
    return np.genfromtxt(os.path.join(PRETRAINED_PCA_DIR, 'simulated_data', '{}_{}_PCA.csv'.format(model, tree_size)),
                         delimiter=',')


def model_scale_load_ffnn(tree_size, model):
    pred_method = 'FFNN'
    # with open(os.path.join(PRETRAINED_MODELS_DIR, 'models', '{}_{}_{}.json'.format(model, tree_size, pred_method)), 'r') \
    #         as json_file:
    #     loaded_model = json_file.read()
    #
    # model_ffnn = keras.models.model_from_json(loaded_model)
    # model_ffnn.load_weights(os.path.join(PRETRAINED_MODELS_DIR, 'weights',
    #                                      '{}_{}_{}.h5'.format(model, tree_size, pred_method)))
    model_ffnn = keras.models.load_model(
        os.path.join(PRETRAINED_MODELS_DIR, 'models', '{}_{}_{}.h5'.format(model, tree_size, pred_method)),
        compile=False)

    with open(os.path.join(PRETRAINED_MODELS_DIR, 'scalers', '{}_{}_{}.pkl'.format(model, tree_size, pred_method)),
              'rb') as f:
        scaler = pk.load(f)

    return model_ffnn, scaler


def model_load_cnn(tree_size, model):
    pred_method = 'CNN'
    # with open(os.path.join(PRETRAINED_MODELS_DIR, 'models', '{}_{}_{}.json'.format(model, tree_size, pred_method)),
    #           'r') as json_file:
    #     loaded_model = json_file.read()

    # model_cnn = keras.models.model_from_json(loaded_model)
    # model_cnn.load_weights(os.path.join(PRETRAINED_MODELS_DIR, 'weights',
    #                                     '{}_{}_{}.h5'.format(model, tree_size, pred_method)))
    model_cnn = keras.models.load_model(
        os.path.join(PRETRAINED_MODELS_DIR, 'models', '{}_{}_{}.h5'.format(model, tree_size, pred_method)),
        compile=False)

    return model_cnn
