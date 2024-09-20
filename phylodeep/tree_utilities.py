import logging

import numpy as np
from ete3 import Tree
from phylodeep import TIME_DEPENDENT_COLUMNS

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


def _merge(l1, l2, key, max_size=np.inf):
    """
    Merges two sorted arrays
    :param l1: array 1
    :param l2: array 2
    :return: merged array
    """
    res = []
    i, j = 0, 0
    while len(res) < max_size and (i < len(l1) or j < len(l2)):
        if i == len(l1):
            res.extend(l2[j:min(len(l2), j + max_size - len(res))])
            break
        if j == len(l2):
            res.extend(l1[i: min(len(l1), i + max_size - len(res))])
            break
        if key(l1[i]) <= key(l2[j]):
            res.append(l1[i])
            i += 1
        else:
            res.append(l2[j])
            j += 1
    return res


def _annotate_relative_dates(tre, date_feature):
    """
    Annotates tree nodes with the relative date (time since the root).
    To be able to uniquely sort equal dates, we make a tuple of the date: (relative_date, i),
    where i in the level order of the node
    """
    for i, n in enumerate(tre.traverse('levelorder')):
        n.add_feature(date_feature, (((0 if n.is_root() else getattr(n.up, date_feature)[0]) + n.dist), i))


def extract_clusters(tre, min_size, max_size):
    """
    Cuts the given tree into subtrees within a given size (s) range: min_size <= s <= max_size.
    The initial tree object is modified.

    :param max_size: minimal number of tips for a subtree (inclusive)
    :param min_size: maximal number of tips for a subtree (inclusive)
    :param tre: ete3.Tree
    :return: a generator of extracted subtrees
    """
    date_feature = 'date'
    sorted_tips_feature = 'sorted-tips'
    taken_num_feature = 'taken'
    selection_strategy_feature = 'how'
    strategy_top = 'top'
    strategy_recursive = 'recurse'
    strategy_mixed = 'mixed_{}'

    _annotate_relative_dates(tre, date_feature)

    def get_oldest_date(m):
        return getattr(getattr(m, sorted_tips_feature)[0], date_feature)

    def get_youngest_date(m):
        return getattr(getattr(m, sorted_tips_feature)[-1], date_feature)

    for n in tre.traverse('postorder'):
        n.add_feature(sorted_tips_feature,
                      [n] if n.is_leaf()
                      else _merge(*(getattr(_, sorted_tips_feature) for _ in n.children),
                                  key=lambda _: getattr(_, date_feature),
                                  max_size=max_size))
        n_size = len(n)

        if n_size < min_size:
            n.add_feature(taken_num_feature, 0)
        elif n_size <= max_size:
            n.add_feature(taken_num_feature, n_size)
            n.add_feature(selection_strategy_feature, strategy_top)
        else:
            taken = sum(getattr(_, taken_num_feature) for _ in n.children)
            how = strategy_recursive

            # if all the top leaves would come from just one of the children anyway,
            # the mixed solution will give the same result as recurse
            older_child, younger_child = sorted(n.children, key=get_oldest_date)
            if not (len(older_child) >= max_size and get_youngest_date(older_child) < get_oldest_date(younger_child)):
                tips = getattr(n, sorted_tips_feature)
                next_todo = list(n.children)
                for i in range(min_size, min(n_size, max_size) + 1):
                    date = getattr(tips[i - 1], date_feature)
                    size = i
                    todo = next_todo
                    next_todo = []
                    while todo:
                        m = todo.pop()

                        # if there is nothing to take here, no need to descend further
                        if not getattr(m, taken_num_feature):
                            continue

                        if getattr(getattr(m, sorted_tips_feature)[0], date_feature) <= date:
                            todo.extend(m.children)
                        else:
                            size += getattr(m, taken_num_feature)
                            next_todo.append(m)

                    if size > taken:
                        taken = size
                        how = strategy_mixed.format(i) if size > i else strategy_top
            n.add_feature(taken_num_feature, taken)
            n.add_feature(selection_strategy_feature, how)

    n_branches = 2 * len(tre) - 2
    n_subtrees = 0
    n_subtree_branches = 0
    for subtree in _dissect_tree(tre, min_size, max_size, date_feature,
                                 selection_strategy_feature, sorted_tips_feature, strategy_recursive, strategy_top):
        yield subtree
        n_subtrees += 1
        n_subtree_branches += 2 * len(subtree) - 2

    get_logger('subtree_picker').info('Picked {} subtrees covering {} out of {} branches ({:.1f}%).'
                                      .format(n_subtrees, n_subtree_branches, n_branches,
                                              100 * n_subtree_branches / n_branches))


def _dissect_tree(tre, min_size, max_size, date_feature, selection_strategy_feature,
                 sorted_tips_feature, strategy_recursive, strategy_top):
    todo = [tre]
    while todo:
        n = todo.pop()
        if len(n) < min_size:
            continue
        how = getattr(n, selection_strategy_feature)
        if strategy_recursive == how:
            todo.extend(n.children)
            continue
        if strategy_top == how:
            n.detach()
            if len(n) <= max_size:
                yield n
                continue
            # tips should contain exactly max_size oldest tips
            tips = getattr(n, sorted_tips_feature)
            yield remove_certain_leaves(n, lambda _: _ not in tips)
            continue
        # strategy mixed in action
        i = int(how[6:])
        date = getattr(getattr(n, sorted_tips_feature)[i - 1], date_feature)
        child_todo = list(n.children)
        while child_todo:
            m = child_todo.pop()
            if getattr(getattr(m, sorted_tips_feature)[0], date_feature) <= date:
                child_todo.extend(m.children)
            else:
                parent = m.up
                todo.append(m.detach())
                if parent.up:
                    for c in parent.children:
                        parent.up.add_child(c, dist=c.dist + parent.dist)
                    parent.up.remove_child(parent)
        yield n.detach()


def get_logger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level=level)
    logger.propagate = False
    if not logger.hasHandlers():
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(levelname)s:%(name)s:%(asctime)s %(message)s', datefmt="%Y-%m-%d_%H:%M:%S")
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger


def subtree_picker(in_nwk, out_nwk, min_size=MIN_TREE_SIZE_SMALL, max_size=MIN_TREE_SIZE_HUGE - 1):
    """
    Cuts the given tree into subtrees within a given size (s) range: min_size <= s <= max_size.

    :param max_size: minimal number of tips for a subtree (inclusive)
    :param min_size: maximal number of tips for a subtree (inclusive)
    :param in_nwk: input tree in newick format
        (must be rooted, without polytomies and containing at least --min_size tips)
    :param out_nwk: output newick file to store the generated subtrees
    """
    if not min_size or not max_size or min_size < 1 or max_size < min_size:
        raise ValueError('Minimal and maximal subtree sizes must be positive integers such that min_size <= max_size.')
    tree = read_tree(in_nwk)
    if min_size > len(tree):
        raise ValueError('Minimal subtree size cannot be greater than the input tree size.')

    with open(out_nwk, 'w+') as f:
        for subtree in extract_clusters(tree, min_size=min_size, max_size=max_size):
            f.write(subtree.write() + '\n')


def subtree_picker_main():
    """
    Entry point, calling :py:func:`phylodeep.tree_utilities.subtree_picker`  with command-line arguments.
    :return: void
    """
    import argparse

    parser = argparse.ArgumentParser(description="Cuts the given tree into subtrees within a given size (s) range: "
                                                 "min_size <= s <= max_size.",
                                     prog='subtree_picker')

    tree_group = parser.add_argument_group('tree-related arguments')
    tree_group.add_argument('-i', '--in_nwk',
                            help="input tree in newick format "
                                 "(must be rooted, without polytomies and containing at least --min_size tips).",
                            type=str, required=True)
    tree_group.add_argument('-o', '--out_nwk',
                            help="output newick file to store the generated subtrees.",
                            type=str, required=True)

    size_group = parser.add_argument_group('subtree size-related arguments')
    size_group.add_argument('-m', '--min_size', required=False, type=int, default=MIN_TREE_SIZE_SMALL,
                             help="minimal number of tips for a subtree (inclusive).")
    size_group.add_argument('-M', '--max_size', required=False, type=int, default=MIN_TREE_SIZE_HUGE - 1,
                             help="maximal number of tips for a subtree (inclusive).")

    params = parser.parse_args()
    subtree_picker(**vars(params))


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
        if elt in TIME_DEPENDENT_COLUMNS:
            predict[elt] *= rescale_f

    return predict
