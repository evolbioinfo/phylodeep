from ete3 import Tree


def read_tree(newick_tree):
    """ Tries all nwk formats and returns an ete3 Tree

    :param newick_tree: str, a tree in newick format
    :return: ete3.Tree
    """
    tree = None
    for f in (3, 2, 5, 0, 1, 4, 6, 7, 8, 9):
        try:
            tree = Tree(newick_tree, format=f)
            break
        except:
            continue
    if not tree:
        raise ValueError('Could not read the tree {}. Is it a valid newick?'.format(newick_tree))
    return tree


def read_tree_file(tree_path):
    with open(tree_path, 'r') as f:
        nwk = f.read().replace('\n', '').split(';')
        if nwk[-1] == '':
            nwk = nwk[:-1]
    if not nwk:
        raise ValueError('Could not find any trees (in newick format) in the file {}.'.format(tree_path))
    if len(nwk) > 1:
        raise ValueError('There are more than 1 tree in the file {}. Now, we accept only one tree per inference.'.format(tree_path))
    return read_tree(nwk[0] + ';')


def check_tree_size(tre):
    """
    Verifies whether input tree is of correct size and determines the tree size range for models to use
    :param tre: ete3.Tree
    :return: int, tree_size
    """
    if 49 < len(tre) < 200:
        tre_size = 'SMALL'
    elif 199 < len(tre) < 501:
        tre_size = 'LARGE'
    else:
        raise ValueError('Your input tree is of incorrect size (either smaller than 50 tips or larger than 500 tips.')

    return tre_size


def annotator(predict, mod):
    """
    annotates the pd.DataFrame containing predicted values
    :param predict: predicted values
    :type: pd.DataFrame
    :param mod: model under which the parameters were estimated
    :type: str
    :return:
    """

    if mod == "BD":
        predict.columns = ["R_naught", "Infectious_period"]
    elif mod == "BDEI":
        predict.columns = ["R_naught", "Infectious_period", "Incubation_period"]
    elif mod == "BDSS":
        predict.columns = ["R_naught", "Infectious_period", "X_transmission", "Superspreading_individuals_fraction"]
    elif mod == "BD_vs_BDEI_vs_BDSS":
        predict.columns = ["Probability_BDEI", "Probability_BD", "Probability_BDSS"]
    elif mod == "BD_vs_BDEI":
        predict.columns = ["Probability_BD", "Probability_BDEI"]
    return predict


def rescaler(predict, rescale_f):
    """
    rescales the predictions back to the initial tree scale (e.g. days, weeks, years)
    :param predict: predicted values
    :type: pd.DataFrame
    :param rescale_f: rescale factor by which the initial tree was scaled
    :type: float
    :return:
    """

    for elt in predict.columns:
        if "period" in elt:
            predict[elt] = predict[elt]*rescale_f

    return predict
