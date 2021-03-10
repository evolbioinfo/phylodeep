
import pandas as pd

from phylodeep import FULL, SUMSTATS, BD, BDEI, BDSS
from phylodeep.encoding import encode_into_summary_statistics, encode_into_most_recent
from phylodeep.model_load import model_scale_load_ffnn, model_load_cnn
from phylodeep.ci_comput import ci_comp
from phylodeep.tree_utilities import *

import warnings
warnings.filterwarnings("ignore")

prediction_method_options = [FULL, SUMSTATS]
param_model_options = [BD, BDEI, BDSS]


def paramdeep(tree_file, proba_sampling, model=BD, vector_representation=FULL, ci_computation=False, **kvargs):
    """
    Provides model selection between selected models for given tree.
    For more information on the covered parameter subspaces, we refer you to the following paper: ...
    :param tree_file: path to a file with a dated tree in newick format (must be rooted, without polytomies and of size
    between 50 adn 500 tips).
    :type tree_file: str
    :param proba_sampling: presumed sampling probability for all input trees, value between 0.01 and 1
    :type proba_sampling: float
    :param model: option to choose, for a tree of size between 50 and 199 tips, you can choose either 'BD' (basic
    birth-death model with incomplete sampling BD), 'BDEI' (BD with exposed class); for a tree of size between 200 and
    500 tips, you can choose between 'BD', 'BDEI' and  'BDSS' (BD with superspreading).
    :type model: str
    :param vector_representation: option to choose between 'FFNN_SUMSTATS' to select a network trained on summary
    statistics or 'CNN_FULL_TREE' to select a network trained on full tree representation, by default, we use
    'CNN_FULL_TREE'
    :type vector_representation: str
    :param ci_computation: (optional, default is False) By default (ci_computation=False), paramdeep outputs point
    estimate for each parameter. With ci_computation=True, paramdeep computes and outputs 95% confidence intervals
    (2.5% and 97.5%) for estimated value using approximated parametric bootstrap.
    :type ci_computation: bool
    :return: pd.df, predicted parameter values (and 95% CIs if option chosen)
    """
    # check options
    if proba_sampling > 1 or proba_sampling < 0.01:
        raise ValueError('Incorrect value of \'sampling probability\' parameter')
    if model not in param_model_options:
        raise ValueError('Incorrect value of \'model\' option.')
    if vector_representation not in prediction_method_options:
        raise ValueError('Incorrect value of \'prediction_method\' option.')

    # read trees
    tree = read_tree_file(tree_file)

    # check tree size
    tree_size = check_tree_size(tree)

    # only inference on short trees available for BDSS
    if tree_size == "SMALL" and model == "BDSS":
        raise ValueError('Parameter inference under BDSS is available only for trees of size between 200 and 500 tips')

    # encode the trees
    if vector_representation == SUMSTATS:
        encoded_tree, rescale_factor = encode_into_summary_statistics(tree, proba_sampling)
    elif vector_representation == FULL:
        encoded_tree, rescale_factor = encode_into_most_recent(tree, proba_sampling)

    # load model
    if vector_representation == SUMSTATS:
        loaded_model, scaler = model_scale_load_ffnn(tree_size, model)
    elif vector_representation == FULL:
        loaded_model = model_load_cnn(tree_size, model)

    # predict values:
    if vector_representation == SUMSTATS:
        encoded_tree = scaler.transform(encoded_tree)
        predictions = pd.DataFrame(loaded_model.predict(encoded_tree))
    elif vector_representation == FULL:
        predictions = pd.DataFrame(loaded_model.predict(encoded_tree))

    # annotate predictions:
    predictions = annotator(predictions, model)

    # if required, computation of 95% confidence intervals
    if ci_computation:
        predictions = ci_comp(predictions, model, rescale_factor, len(tree), tree_size, proba_sampling, vector_representation)
    else:
        predictions = rescaler(predictions, rescale_factor)
    return predictions


def main():
    """
    Entry point, calling :py:func:`phylodeep.paramdeep`  with command-line arguments.
    :return: void
    """
    import argparse

    parser = argparse.ArgumentParser(description="Parameter inference for phylodynamics using pretrained neural "
                                                 "networks.",
                                     prog='paramdeep')

    tree_group = parser.add_argument_group('tree-related arguments')
    tree_group.add_argument('-t', '--tree_file', help="input tree in newick format (must be rooted, without polytomies"
                                                      " and of size between 50 adn 500 tips).",
                            type=str, required=True)
    tree_group.add_argument('-p', '--proba_sampling', help="presumed sampling probability for removed tips. Must be "
                                                           "between 0.01 and 1",
                            type=float, required=True)

    prediction_group = parser.add_argument_group('neural-network-prediction arguments')
    prediction_group.add_argument('-m', '--model', choices=[BD, BDEI, BDSS],
                                  required=True, type=str, default=None,
                                  help="Choose one of the models to be for which you want to obtain parameter "
                                       "estimates. For parameter inference,"
                                       " you can choose either BD (basic birth-death with incomplete sampling) or"
                                       " BDEI (BD with exposed-infectious) for trees of size between 50 and 199 tips"
                                       " and BD, BDEI or BDSS (BD with superspreading individuals) for trees of size"
                                       " between 200 and 500 tips.")

    prediction_group.add_argument('-v', '--vector_representation', choices=[FULL, SUMSTATS], required=False, type=str,
                                  default=FULL,
                                  help="Choose neural networks: either FULL: CNN trained on full tree representation or"
                                       " SUMSTATS: FFNN trained on summary statistics. By default set to FULL.")

    prediction_group.add_argument('-c', '--ci_computation', action='store_true', required=False,
                                  help="By default (without --ci_computation option), paramdeep outputs a csv file"
                                       " (comma-separated) with"
                                       " point estimates for each parameter. With --ci_computation option turned on,"
                                       " paramdeep computes and outputs 95%% confidence intervals (2.5%% and 97.5%%)"
                                       " for each estimate using approximated parametric bootstrap.")

    output_group = parser.add_argument_group('output')
    output_group.add_argument('-o', '--output', required=True, type=str, help="The name of the output file.")

    params = parser.parse_args()

    inference = paramdeep(**vars(params))

    inference.to_csv(params.output)


if '__main__' == __name__:
    main()
