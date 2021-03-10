
import pandas as pd

from phylodeep import FULL, SUMSTATS
from phylodeep.encoding import encode_into_summary_statistics, encode_into_most_recent
from phylodeep.model_load import model_scale_load_ffnn, model_load_cnn
from phylodeep.tree_utilities import *

import warnings
warnings.filterwarnings("ignore")

prediction_method_options = [FULL, SUMSTATS]


def modeldeep(tree_file, proba_sampling, vector_representation=FULL, **kvargs):
    """
    Provides model selection between birth-death models models for given tree.
    For trees of size 200-500 tips, it performs a selection between the basic birth-death model with incomplete sampling
    (BD), the birth-death model with exposed and infectious classes (BDEI) and birth-death model with superspreading
    (BDSS).
    For trees of size 50-199, it performs a model selection between BD and BDEI. For more information on the covered
    parameter subspaces, we refer you to the following paper: ...
    :param tree_file: path to a file with a dated tree in newick format (must be rooted, without polytomies and of size
    between 50 adn 500 tips).
    :type tree_file: str
    :param proba_sampling: presumed sampling probability for all input trees, value between 0.01 and 1
    :type proba_sampling: float
    :param vector_representation: option to choose between 'FFNN_SUMSTATS' to select a network trained on summary statistics
    or 'CNN_FULL_TREE' to select a network trained on full tree representation, by default, we use 'CNN FULL TREE'
    :type vector_representation: str
    :return: pd.df, model selection results in the form of probabilities of each model
    """
    # check options
    if proba_sampling > 1 or proba_sampling < 0.01:
        raise ValueError('Incorrect value of \'sampling probability\' parameter')
    if vector_representation not in prediction_method_options:
        raise ValueError('Incorrect value of \'prediction method\' option.')

    # read trees
    tree = read_tree_file(tree_file)

    # check tree size
    tree_size = check_tree_size(tree)

    if tree_size == "LARGE":
        model = "BD_vs_BDEI_vs_BDSS"
    else:
        model = "BD_vs_BDEI"
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
    # if inferred paramater values: rescale back the rates
    predictions = rescaler(predictions, rescale_factor)

    return predictions


def main():
    """
    Entry point, calling :py:func:`phylodeep.modeldeep`  with command-line arguments.
    :return: void
    """
    import argparse

    parser = argparse.ArgumentParser(description="Model selection for phylodynamics using pretrained neural networks.",
                                     prog='modeldeep')

    tree_group = parser.add_argument_group('tree-related arguments')
    tree_group.add_argument('-t', '--tree_file', help="input tree in newick format (must be rooted, without polytomies"
                                                      " and of size between 50 adn 500 tips).",
                            type=str, required=True)
    tree_group.add_argument('-p', '--proba_sampling', help="presumed sampling probability for removed tips. Must be "
                                                           "between 0.01 and 1",
                            type=float, required=True)

    prediction_group = parser.add_argument_group('neural-network-prediction arguments')

    prediction_group.add_argument('-v', '--vector_representation', choices=[FULL, SUMSTATS], required=False, type=str,
                                  default=FULL,
                                  help="Choose a type of tree representation and neural networks."
                                       " You can choose either FULL: CNN trained on full tree representation or"
                                       " SUMSTATS: FFNN trained on summary statistics. By default set to FULL.")

    output_group = parser.add_argument_group('output')
    output_group.add_argument('-o', '--output', required=True, type=str, help="The name of the output csv file"
                                                                              " (comma-separated)"
                                                                              " containing predicted probabilities of"
                                                                              " each model.")

    params = parser.parse_args()

    selection = modeldeep(**vars(params))

    selection.to_csv(params.output)


if '__main__' == __name__:
    main()
