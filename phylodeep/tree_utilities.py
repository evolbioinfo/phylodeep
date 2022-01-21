from ete3 import Tree

HUGE = 'HUGE'

LARGE = 'LARGE'

SMALL = 'SMALL'

MIN_TREE_SIZE_SMALL = 50
MIN_TREE_SIZE_LARGE = 200
MIN_TREE_SIZE_HUGE = 501


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
    # First try if ETE3 can handle the tree directly
    try:
        return read_tree(tree_path)
    except:
        # If it can't check if there are multiple trees in the file, etc.
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
    tree_size = len(tre)
    if tree_size < MIN_TREE_SIZE_SMALL:
        raise ValueError('Your input tree is of too small: {} tips while minimal size is {}.'
                         .format(tree_size, MIN_TREE_SIZE_SMALL))
    if tree_size < MIN_TREE_SIZE_LARGE:
        return SMALL
    if tree_size < MIN_TREE_SIZE_HUGE:
        return LARGE
    return HUGE


def extract_clusters(tre, min_size, max_size):
    """
    Cuts the given tree into subtrees within a given size (s) range: min_size <= s <= max_size.
    The initial tree object is modified.

    :param max_size: minimal number of tips for a subtree (inclusive)
    :param min_size: maximal number of tips for a subtree (inclusive)
    :param tre: ete3.Tree
    :return: a generator of extracted subtrees
    """
    todo = [tre]
    while todo:
        n = todo.pop()
        if len(n) > max_size:
            todo.extend(n.children)
        elif len(n) >= min_size:
            n.detach()
            yield n


def remove_certain_leaves(tr, to_remove=lambda node: False):
    """
    Removes all the branches leading to leaves identified positively by to_remove function.
    :param tr: the tree of interest (ete3 Tree)
    :param to_remove: a method to check is a leaf should be removed.
    :return: void, modifies the initial tree.
    """

    tips = [tip for tip in tr if to_remove(tip)]
    for node in tips:
        if node.is_root():
            return None
        parent = node.up
        parent.remove_child(node)
        # If the parent node has only one child now, merge them.
        if len(parent.children) == 1:
            brother = parent.children[0]
            brother.dist += parent.dist
            if parent.is_root():
                brother.up = None
                tr = brother
            else:
                grandparent = parent.up
                grandparent.remove_child(parent)
                grandparent.add_child(brother)
    return tr


def extract_root_cluster(tre, size):
    """
    Prunes the tree to include a given number (size) of oldest tips.

    :param size: number of tips for the subtree
    :param tre: ete3.Tree
    :return: the subtree
    """
    n2T = {}
    for n in tre.traverse('preorder'):
        n2T[n] = (0 if n.is_root() else n2T[n.up]) + n.dist
    tips = sorted([_ for _ in tre], key=lambda _: n2T[_])
    return remove_certain_leaves(tre, lambda _: _ not in tips[:size])


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
