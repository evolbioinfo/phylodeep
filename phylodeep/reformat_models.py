import os.path

import keras

prefix = '/home/azhukova/projects/phylodeep/phylodeep/pretrained_models/'
sizes = ['SMALL', 'LARGE']
models = ['BD', 'BDEI', 'BDSS', 'BD_vs_BDEI', 'BD_vs_BDEI_vs_BDSS']
archs = ['CNN', 'FFNN']

for model in models:
    for size in sizes:
        for arch in archs:
            inpath = prefix + 'models/' + model + '_' + size + '_' + arch + '.json'
            if os.path.exists(inpath):
                model_cnn = keras.models.model_from_json(
                    open(inpath, 'r').read())
                model_cnn.load_weights(prefix + 'weights/' + model + '_' + size + '_' + arch + '.h5')
                model_cnn.save(prefix + 'models/' + model + '_' + size + '_' + arch + '.h5', save_format='h5')
