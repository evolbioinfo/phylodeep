import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import gaussian_kde

from phylodeep import FULL, SUMSTATS, BD, BDEI, BDSS
from phylodeep.encoding import encode_into_summary_statistics
from phylodeep.model_load import pca_scale_load, pca_data_load
from phylodeep.tree_utilities import *

prediction_method_options = [FULL, SUMSTATS]
param_model_options = [BD, BDEI, BDSS]


def pca_plot(tree_d, sim_d, variance_explained, png_out):
    """
    Creates a figure with two plots showing the ouput of a priori model adequacy check.
    :param tree_d: four principal components of the input tree representation
    :type tree_d: pd.DataFrame
    :param sim_d: four principal components of 10.000 trees simulated under given model
    :type sim_d: pd.DataFrame
    :param variance_explained: % variance explained by each one of the four principal components
    :type variance_explained: list of floats
    :param png_out: name of output png file
    :type png_out: str
    :return: void
    """
    # sim_d = np.matrix(sim_d, dtype='float')
    # plotting the first and the second PC
    xy = np.vstack([sim_d[:, 0], sim_d[:, 1]])
    z = gaussian_kde(xy)(xy)

    fig, (ax1, ax2) = plt.subplots(1,2)
    fig.tight_layout(pad=3.0)

    ax1.scatter(sim_d[:, 0], sim_d[:, 1], c=z, s=100, edgecolor='white', linewidth=0)
    ax1.plot(tree_d[:, 0], tree_d[:, 1], '*', color='r', markersize=20)

    ax1.set_xlim([min(min(sim_d[:, 0]), tree_d[:, 0]), max(max(sim_d[:, 0]), tree_d[:, 0])])
    ax1.set_ylim([min(min(sim_d[:, 1]), tree_d[:, 1]), max(max(sim_d[:, 1]), tree_d[:, 1])])

    ax1.set(xlabel='PC 1 (% var exp: {:.1f}).'.format(variance_explained[0] * 100),
            ylabel='PC 2 (% var exp: {:.1f}).'.format(variance_explained[1] * 100))

    # plotting third and fourth PC
    xy2 = np.vstack([sim_d[:, 2], sim_d[:, 3]])
    z2 = gaussian_kde(xy2)(xy2)

    ax2.scatter(sim_d[:, 2], sim_d[:, 3], c=z2, s=100, edgecolor='white', linewidth=0)
    ax2.plot(tree_d[:, 2], tree_d[:, 3], '*', color='r', markersize=20)

    ax2.set_xlim([min(min(sim_d[:, 2]), tree_d[:, 2]), max(max(sim_d[:, 2]), tree_d[:, 2])])
    ax2.set_ylim([min(min(sim_d[:, 3]), tree_d[:, 3]), max(max(sim_d[:, 3]), tree_d[:, 3])])

    ax2.set(xlabel='PC 3 (% var exp: {:.1f}).'.format(variance_explained[2] * 100),
            ylabel='PC 4 (% var exp: {:.1f}).'.format(variance_explained[3] * 100))

    # save fig
    plt.savefig(png_out)

    return None


def checkdeep(tree_file, model=BD, outputfile_png='a_priori_check.png', **kvargs):
    """
    Provides a priori model adequacy check for given tree and model. PCA on 10,000 simulations is performed and plotted,
    together with a projected point corresponding to the input tree. For the PCA, summary statistics representation is
    used. Four first principal components are plotted on two plots, together with variability explained for each
    component.
    For more information on the covered parameter subspaces (by simulations under given model), we refer you to the
    following paper: : Voznica et al. 2021 doi:10.1101/2021.03.11.435006.

    :param tree_file: path to a file with a dated tree in newick format
        (must be rooted, without polytomies and containing at least 50 tips).
    :type tree_file: str
    :param model: option to choose, for a tree of size between 50 and 199 tips, you can choose either 'BD' (basic
    birth-death model with incomplete sampling BD), 'BDEI' (BD with exposed class); for a tree of size >= 200 tips,
    you can choose between 'BD', 'BDEI' and  'BDSS' (BD with superspreading).
    :type model: str
    :param outputfile_png: name (with path) of the output png file, showing the result of the a priori check
    :type outputfile_png: str
    :return: void
    """
    # check options
    if model not in param_model_options:
        raise ValueError('Incorrect value of \'model\' option.')

    # read trees
    tree = read_tree_file(tree_file)

    # check tree size
    tree_size = check_tree_size(tree)

    # only inference on short trees available for BDSS
    if tree_size == "SMALL" and model == "BDSS":
        raise ValueError('Parameter inference under BDSS is available only for trees of size >= 200 tips')

    # encode the tree
    if tree_size == HUGE:
        encoded_subtrees = []
        sizes = []
        # estimate model probabilities on subtrees
        for subtree in extract_clusters(tree, min_size=MIN_TREE_SIZE_LARGE, max_size=MIN_TREE_SIZE_HUGE - 1):
            encoded_subtrees.append(encode_into_summary_statistics(subtree, sampling_proba=0)[0])
            sizes.append(len(subtree))
        encoded_subtrees = pd.concat(encoded_subtrees)
        encoded_tree = pd.DataFrame(columns=encoded_subtrees.columns)
        encoded_subtrees['weight'] = sizes
        encoded_subtrees['weight'] /= sum(sizes)
        for col in encoded_tree.columns:
            encoded_tree.loc[0, col] = (encoded_subtrees[col] * encoded_subtrees['weight']).sum()
        tree_size = LARGE
    else:
        encoded_tree = encode_into_summary_statistics(tree, sampling_proba=0)[0]

    # removing sampling proba, that is not used here
    encoded_tree = encoded_tree.iloc[:, :-1]

    # load PCA model, scaler and representations of simulated data
    loaded_pca, scaler = pca_scale_load(tree_size, model)
    pca_data = pca_data_load(tree_size, model)
    # rescale and project the input tree representation
    encoded_tree = scaler.transform(encoded_tree)
    pca_encoded_tree = loaded_pca.transform(encoded_tree)

    # plot the a priori check
    pca_plot(pca_encoded_tree, pca_data, loaded_pca.explained_variance_ratio_, outputfile_png)

    return None


def main():
    """
    Entry point, calling :py:func:`phylodeep.checkdeep`  with command-line arguments.
    :return: void
    """
    import argparse

    parser = argparse.ArgumentParser(description="A priori model adequacy check of phylogenetic trees for phylodynamic"
                                                 " models. Recommended to perform before selecting phylodynamic models"
                                                 " and estimating parameters.",
                                     prog='checkdeep')

    tree_group = parser.add_argument_group('tree-related arguments')
    tree_group.add_argument('-t', '--tree_file',
                            help="input tree in newick format "
                                 "(must be rooted, without polytomies and containing at least 50 tips).",
                            type=str, required=True)

    model_group = parser.add_argument_group('phylodynamic model arguments')
    model_group.add_argument('-m', '--model', choices=[BD, BDEI, BDSS],
                             required=True, type=str, default=None,
                             help="Choose one of the models for the a priori check. For trees of size,"
                                  " between 50 and 199 tips you can choose either BD (constant-rate birth-death"
                                  " with incomplete sampling), or BDEI (BD with exposed-infectious class). For"
                                  " trees of size >= 200 tips, you can choose between BD, BDEI and"
                                  " BDSS (BD with superspreading).")

    output_group = parser.add_argument_group('output')
    output_group.add_argument('-o', '--outputfile_png', required=True, type=str, help="The name of the output file (in"
                                                                                      " png format).")

    params = parser.parse_args()

    checkdeep(**vars(params))


if '__main__' == __name__:
    main()
