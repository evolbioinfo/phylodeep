import warnings

import pandas as pd

from phylodeep import FULL, SUMSTATS, BD, BDEI, BDSS, MODEL2PARAMS
from phylodeep.ci_comput import ci_comp, CI_2_5, PREDICTED_VALUE, CI_97_5
from phylodeep.encoding import encode_into_summary_statistics, encode_into_most_recent
from phylodeep.model_load import model_scale_load_ffnn, model_load_cnn
from phylodeep.tree_utilities import *

warnings.filterwarnings("ignore")

prediction_method_options = [FULL, SUMSTATS]
param_model_options = [BD, BDEI, BDSS]


def paramdeep(tree_file, proba_sampling, model=BD, vector_representation=FULL, ci_computation=False, **kvargs):
    """
    Provides model selection between selected models for given tree.
    For more information on the covered parameter subspaces, we refer you to the following paper:
    Voznica et al. 2021 doi:10.1101/2021.03.11.435006.

    :param tree_file: path to a file with a dated tree in newick format
        (must be rooted, without polytomies and containing at least 50 tips).
    :type tree_file: str
    :param proba_sampling: presumed sampling probability for all input trees, value between 0.01 and 1
    :type proba_sampling: float
    :param model: option to choose, for a tree of size between 50 and 199 tips,
        you can choose either 'BD' (basic birth-death (BD) model with incomplete sampling),
        'BDEI' (BD with exposed class);
        for a tree of size >= 200 tips, you can choose between 'BD', 'BDEI' or 'BDSS' (BD with superspreading).
    :type model: str
    :param vector_representation: option to choose between
        phylodeep.SUMSTATS to select a network trained on summary statistics
        or phylodeep.FULL to select a network trained on full tree representation,
        by default, we use phylodeep.FULL
    :type vector_representation: str
    :param ci_computation: (optional, default is False) By default (ci_computation=False),
        paramdeep outputs point estimate for each parameter.
        With ci_computation=True, paramdeep computes and outputs 95% confidence intervals (2.5% and 97.5%)
        for estimated value using approximated parametric bootstrap.
    :type ci_computation: bool
    :return: pd.DataFrame, predicted parameter values (and 95% CIs if option chosen)
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
    if tree_size == SMALL and model == BDSS:
        raise ValueError('Parameter inference under {} is available only for trees of size above {}'
                         .format(BDSS, MIN_TREE_SIZE_LARGE))

    if tree_size == HUGE:
        predictions = []
        sizes = []
        # estimate parameters on subtrees
        for subtree in extract_clusters(tree, min_size=MIN_TREE_SIZE_SMALL if model != BDSS else MIN_TREE_SIZE_LARGE,
                                        max_size=MIN_TREE_SIZE_HUGE - 1):
            subtree_size = check_tree_size(subtree)
            predictions.append(
                _paramdeep_tree(subtree, subtree_size, model, proba_sampling, vector_representation, ci_computation))
            sizes.append(len(subtree))
        predictions = pd.concat(predictions)
        df = pd.DataFrame(columns=predictions.columns)
        sizes = np.array(sizes)
        n = sum(sizes)
        pred_val_col = PREDICTED_VALUE if ci_computation else next(iter(predictions.index))
        for col in df.columns:
            predictions_value = predictions.loc[predictions.index == pred_val_col, col].to_numpy()
            val = predictions_value.dot(sizes) / n
            df.loc[pred_val_col, col] = val

            if ci_computation:
                ci_2_5 = predictions_value - predictions.loc[predictions.index == CI_2_5, col]
                ci_97_5 = predictions.loc[predictions.index == CI_97_5, col] - predictions_value

                ci_2_5 = np.power(np.power(ci_2_5, 2).dot(sizes) / n, 0.5)
                ci_97_5 = np.power(np.power(ci_97_5, 2).dot(sizes) / n, 0.5)

                df.loc[CI_2_5, col] = val - ci_2_5
                df.loc[CI_97_5, col] = val + ci_97_5
        return df

    return _paramdeep_tree(tree, tree_size, model, proba_sampling, vector_representation, ci_computation)


def _paramdeep_tree(tree, tree_size, model, proba_sampling, vector_representation, ci_computation):
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
    predictions.columns = MODEL2PARAMS[model]
    # if required, computation of 95% confidence intervals
    if ci_computation:
        predictions = ci_comp(predictions, model, rescale_factor, len(tree), tree_size, proba_sampling,
                              vector_representation)
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
                                                      " and containing at least 50 tips).",
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
                                       " >= 200 tips.")

    prediction_group.add_argument('-v', '--vector_representation', choices=[FULL, SUMSTATS], required=False, type=str,
                                  default=FULL,
                                  help="Choose neural networks: either {full}: CNN trained on full tree representation "
                                       "or {sumstats}: FFNN trained on summary statistics. By default set to {full}."
                                  .format(full=FULL, sumstats=SUMSTATS))

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
