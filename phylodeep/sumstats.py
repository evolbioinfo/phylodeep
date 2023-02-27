import numpy as np

from scipy.stats import linregress
from math import floor

DISTANCE_TO_ROOT = "dist_to_root"

DEPTH = "depth"

HEIGHT = "height"

LADDER = "ladder"

VISITED = "visited"

col_chains = [
        'number_sumchain', 'mean_sumchain', 'min_sumchain', '1st_decile_sumchain', '2nd_decile_sumchain',
        '3rd_decile_sumchain', '4th_decile_sumchain', 'median_sumchain', '6th_decile_sumchain', '7th_decile_sumchain',
        '8th_decile_sumchain', '9th_decile_sumchain', 'max_sumchain', 'var_sumchain'
    ]


def add_depth_and_get_max(tre):
    """
    adds depth to each node. Adapted to polytomial (non binary) trees.
    :param tre: ete3.Tree, the tree to which depth should be added
    :return: modifies the original tree + maximum depth
    """
    maxdepth = 0
    for node in tre.traverse('levelorder'):
        if not node.is_root():
            # initiation at children of root
            if node.up.is_root():
                node.add_feature("depth", 1)
            # then, add 1 depth at each level
            else:
                node.add_feature("depth", getattr(node.up, "depth", False) + 1)
                if getattr(node, "depth", False) > maxdepth:
                    maxdepth = getattr(node, "depth", False)
    return maxdepth


def add_ladder(tr):
    """
    adds ladder score to each node.
    :param tr: ete3.Tree, the tree to which ladder score should be added
    :return: modifies the original tree
    """
    for node in tr.traverse('levelorder'):
        if not node.is_root():
            if node.up.is_root():
                if not node.is_leaf():
                    if node.children[0].is_leaf() or node.children[1].is_leaf():
                        node.add_feature("ladder", 0)
                    else:
                        node.add_feature("ladder", -1)
                else:
                    node.add_feature("ladder", -1)
            else:
                if not node.is_leaf():
                    if node.children[0].is_leaf() and node.children[1].is_leaf():
                        node.add_feature("ladder", 0)
                    elif node.children[0].is_leaf() or node.children[1].is_leaf():
                        node.add_feature("ladder", getattr(node.up, "ladder", False) + 1)
                    else:
                        node.add_feature("ladder", 0)
                else:
                    node.add_feature("ladder", -1)
        else:
            node.add_feature("ladder", -1)
    return None


def tree_height(tre):
    """
    Returns the maximum and minimum height of leaves
    :param tre: ete3.Tree, tree on which these metrics are computed
    :return: set of floats, max and min leaf height
    """
    root_to_tips = [leaf.dist_to_root for leaf in tre]
    max_h = max(root_to_tips)
    min_h = min(root_to_tips)
    return [max_h, min_h]


def branches(tre):
    """
    Returns branch length metrics (all branches taken into account and external only)
    :param tre: ete3.Tree, tree on which these metrics are computed
    :return: set of floats, metrics on all branches
    """
    # branch lengths
    dist_all = [node.dist for node in tre.traverse("levelorder")]
    # external branch lengths
    dist_ext = [leaf.dist for leaf in tre]

    all_bl_mean = np.mean(dist_all)
    all_bl_median = np.median(dist_all)
    all_bl_var = np.nanvar(dist_all)

    ext_bl_mean = np.mean(dist_ext)
    ext_bl_median = np.median(dist_ext)
    ext_bl_var = np.nanvar(dist_ext)

    return [all_bl_mean, all_bl_median, all_bl_var, ext_bl_mean, ext_bl_median, ext_bl_var]


def piecewise_branches(tre, all_max, e_bl_mean, e_bl_median, e_bl_var):
    """
    Returns piecewise branch length metrics
    :param tre: ete3.Tree, tree on which these metrics are computed
    :param all_max: float, tree height
    :param e_bl_mean: float, mean length of external branches
    :param e_bl_median: float, median length of external branches
    :param e_bl_var: float, variance of length of external branches
    :return: list of 18 floats, summary statistics on piecewise branch length
    """
    dist_all_1 = []
    dist_all_2 = []
    dist_all_3 = []

    for node in tre.traverse("levelorder"):
        if node.dist_to_root < all_max / 3 and not node.is_leaf():
            dist_all_1.append(node.dist)
        elif node.dist_to_root < 2 * all_max / 3 and not node.is_leaf():
            dist_all_2.append(node.dist)
        elif node.dist_to_root > 2 * all_max / 3 and not node.is_leaf():
            dist_all_3.append(node.dist)

    if len(dist_all_1) > 0:
        i_bl_mean_1 = np.mean(dist_all_1)
        i_bl_median_1 = np.median(dist_all_1)
        i_bl_var_1 = np.nanvar(dist_all_1)

        ie_bl_mean_1 = np.mean(dist_all_1) / e_bl_mean
        ie_bl_median_1 = np.median(dist_all_1) / e_bl_median
        ie_bl_var_1 = np.nanvar(dist_all_1) / e_bl_var

    else:
        i_bl_mean_1 = 0
        i_bl_median_1 = 0
        i_bl_var_1 = 0

        ie_bl_mean_1 = 0
        ie_bl_median_1 = 0
        ie_bl_var_1 = 0

    if len(dist_all_2) > 0:
        i_bl_mean_2 = np.mean(dist_all_2)
        i_bl_median_2 = np.median(dist_all_2)
        i_bl_var_2 = np.nanvar(dist_all_2)

        ie_bl_mean_2 = np.mean(dist_all_2) / e_bl_mean
        ie_bl_median_2 = np.median(dist_all_2) / e_bl_median
        ie_bl_var_2 = np.nanvar(dist_all_2) / e_bl_var
    else:
        i_bl_mean_2 = 0
        i_bl_median_2 = 0
        i_bl_var_2 = 0

        ie_bl_mean_2 = 0
        ie_bl_median_2 = 0
        ie_bl_var_2 = 0

    if len(dist_all_3) > 0:
        i_bl_mean_3 = np.mean(dist_all_3)
        i_bl_median_3 = np.median(dist_all_3)
        i_bl_var_3 = np.nanvar(dist_all_3)

        ie_bl_mean_3 = np.mean(dist_all_3) / e_bl_mean
        ie_bl_median_3 = np.median(dist_all_3) / e_bl_median
        ie_bl_var_3 = np.nanvar(dist_all_3) / e_bl_var

    else:
        i_bl_mean_3 = 0
        i_bl_median_3 = 0
        i_bl_var_3 = 0

        ie_bl_mean_3 = 0
        ie_bl_median_3 = 0
        ie_bl_var_3 = 0

    return [i_bl_mean_1, i_bl_median_1, i_bl_var_1, ie_bl_mean_1, ie_bl_median_1, ie_bl_var_1, i_bl_mean_2, \
           i_bl_median_2, i_bl_var_2, ie_bl_mean_2, ie_bl_median_2, ie_bl_var_2, i_bl_mean_3, i_bl_median_3, \
           i_bl_var_3, ie_bl_mean_3, ie_bl_median_3, ie_bl_var_3]


def colless(tre):
    """
    Returns colless metric of given tree
    :param tre: ete3.Tree, tree on which these metrics are computed
    :return: float, colless metric
    """
    colless_score = 0
    for node in tre.traverse("levelorder"):
        if not node.is_leaf():
            if len(node.children) == 2:
                child1, child2 = node.children
                colless_score += abs(len(child1) - len(child2))
            else:
                scores = []
                for j in range(len(node.children)):
                    for k in range(j + 1, len(node.children)):
                        scores.append(abs(len(node.children[j]) - len(node.children[k])))
                colless_score = np.average(scores)
    return colless_score


def sackin(tre):
    """
    Returns sackin metric
    :param tre: ete3.Tree, tree on which these metrics are computed
    :return: float, sackin score computed on the whole tree (sum of this score on all branches)
    """
    sackin_score = 0
    for node in tre.traverse("levelorder"):
        if node.is_leaf():
            sackin_score += int(getattr(node, DEPTH, False))
    return sackin_score


def wd_ratio_delta_w(tre, max_dep):
    """
    Returns two metrics of tree width
    :param tre: ete3.Tree, tree on which these metrics are computed
    :param max_dep: float, maximal depth of tre
    :return: set of two floats, ratio and difference of maximum width and depth
    """
    width_count = np.zeros(max_dep + 1)
    for node in tre.traverse("levelorder"):
        if not node.is_root():
            width_count[int(getattr(node, DEPTH))] += 1
    max_width = max(width_count)
    delta_w = 0
    for j in range(0, len(width_count) - 1):
        if delta_w < abs(width_count[j] - width_count[j - 1]):
            delta_w = abs(width_count[j] - width_count[j - 1])
    return [max_width / max_dep, delta_w]


def max_ladder_il_nodes(tre):
    """
    Returns the maximal ladder score and proportion of internal nodes in ladder
    :param tre: ete3.Tree, tree on which these metrics are computed
    :return: set of two floats, metrics
    """
    max_ladder_score = 0
    il_nodes = 0
    for node in tre.traverse("preorder"):
        if not node.is_leaf():
            if node.ladder > max_ladder_score:
                max_ladder_score = node.ladder
            if node.ladder > 0:
                il_nodes += 1
    return [max_ladder_score / len(tre), il_nodes / (len(tre) - 1)]


def staircaseness(tre):
    """
    Returns staircaseness metrics
    :param tre: ete3.Tree, tree on which these metrics are computed
    :return: set of two floats, metrics
    """
    nb_imbalanced_in = 0
    ratio_imbalance = []
    for node in tre.traverse("preorder"):
        if not node.is_leaf():
            if abs(len(node.children[0]) - len(node.children[1])) > 0:
                nb_imbalanced_in += 1
            if len(node.children[0]) > len(node.children[1]):
                ratio_imbalance.append(len(node.children[1]) / len(node.children[0]))
            else:
                ratio_imbalance.append(len(node.children[0]) / len(node.children[1]))
    return [nb_imbalanced_in / (len(tre) - 1), np.mean(ratio_imbalance)]


def ltt_plot(tre):
    """
    Returns an event (branching, removal) matrix
    :param tre: ete3.Tree, tree on which these metrics are computed
    :return: np.matrix, branching and removal events
    """
    events = []

    for node in tre.traverse("levelorder"):
        if node.is_leaf():
            events.append([node.dist_to_root, -1, 0])
        else:
            for j in range(1, len(node.children)):
                events.append([node.dist_to_root, 1, 0])

    events = np.asmatrix(events)
    events = np.sort(events.view('i8, i8, i8'), order=['f0'], axis=0).view(float)

    events[0, 2] = float(events[0, 1]) + 1
    for j in np.arange(1, events.shape[0]):
        events[j, 2] = float(events[j - 1, 2]) + float(events[j, 1])

    return events


def ltt_plot_comput(event_mat):
    """
    Returns LTT plot based metrics
    :param event_mat: np.matrix, branching and removal events
    :return: set of 9 floats, LTT plot based metrics
    """
    # PART1 find max and max time of LTT plot
    events = event_mat.copy()
    max_lineages = 0
    max_lineages_t = 0
    index_slope_change = 0
    for j in range(events.shape[0]):
        if events[j, 2] > max_lineages:
            max_lineages = events[j, 2]
            max_lineages_t = events[j, 0]
            index_slope_change = j

    # PART2 slopes before and after max
    slope_1 = linregress(np.squeeze(events[0:index_slope_change, 2]), np.squeeze(events[0:index_slope_change, 0]))[0]
    slope_2 = linregress(np.squeeze(events[index_slope_change:, 2]), np.squeeze(events[index_slope_change:, 0]))[0]
    slope_rat = slope_1 / slope_2

    all_max = events[-1, 0]

    # PART3 mean sampling time, mean branching times

    # all sampling and branching times
    sampling_times = []
    branching_times_1 = []
    branching_times_2 = []
    branching_times_3 = []
    for j in range(events.shape[0]):
        if events[j, 1] == -1:
            sampling_times.append(events[j, 0])
        elif events[j, 0] < all_max / 3:
            branching_times_1.append(events[j, 0])
        elif events[j, 0] < 2 * all_max / 3:
            branching_times_2.append(events[j, 0])
        else:
            branching_times_3.append(events[j, 0])

    # differences of consecutive sampling/branching times leading to mean sampling and branching (1st, 2nd and 3rd
    # part) times
    diff_samp_times = []
    diff_b_times_1 = []
    diff_b_times_2 = []
    diff_b_times_3 = []

    for j in range(0, len(sampling_times) - 1):
        diff_samp_times.append(sampling_times[j + 1] - sampling_times[j])
    for j in range(0, len(branching_times_1) - 1):
        diff_b_times_1.append(branching_times_1[j + 1] - branching_times_1[j])
    for j in range(0, len(branching_times_2) - 1):
        diff_b_times_2.append(branching_times_2[j + 1] - branching_times_2[j])
    for j in range(0, len(branching_times_3) - 1):
        diff_b_times_3.append(branching_times_3[j + 1] - branching_times_3[j])

    # mean sampling time
    mean_s_time = np.mean(diff_samp_times)
    if len(diff_b_times_1) > 0:
        mean_b_time_1 = np.mean(diff_b_times_1)
    else:
        mean_b_time_1 = 0

    if len(diff_b_times_2) > 0:
        mean_b_time_2 = np.mean(diff_b_times_2)
    else:
        mean_b_time_2 = 0

    if len(diff_b_times_3) > 0:
        mean_b_time_3 = np.mean(diff_b_times_3)
    else:
        mean_b_time_3 = 0

    return [max_lineages, max_lineages_t, slope_1, slope_2, slope_rat, mean_s_time, mean_b_time_1, mean_b_time_2, mean_b_time_3]


def coordinates_comp(events):
    """
    Returns representation of LTT plot under 20 bins (20 x-axis and 20 y axis coordinates)
    :param events: np.matrix, branching and removal events
    :return: list of 40 floats, y- and x-axis coordinates from LTT plot
    """
    # from math import floor
    binscor = np.linspace(0, events.shape[0], 21)
    y_axis = []
    x_axis = []
    for j in range(int(len(binscor) - 1)):
        y_axis.append(np.average(events[int(floor(binscor[j])):int(floor(binscor[j + 1])), 0]))
        x_axis.append(np.average(events[int(floor(binscor[j])):int(floor(binscor[j + 1])), 2]))

    y_axis.extend(x_axis)
    return y_axis


def add_height(tre):
    """
    adds height to each internal node.
    :param tre: ete3.Tree, the tree to which height should be added
    :return: void, modifies the original tree
    """
    for node in tre.traverse('postorder'):
        if node.is_leaf():
            node.add_feature("height", 0)
        else:
            max_child = 0
            for child in node.children:
                if getattr(child, "height", False) > max_child:
                    max_child = getattr(child, "height", False)
            node.add_feature("height", max_child + 1)
    return None


def compute_chain(node, order=4):
    """
    Return a list of shortest descending path from given node (i.e. 'transmission chain'), of given order at maximum
    :param node: ete3.node, node on which the descending path will be computed
    :param order: int, order of transmission chain
    :return: list of floats, of maximum length (order)
    """
    chain = []
    contin = True
    while len(chain) < order and contin:
        children_dist = [child.dist for child in node.children]

        chain.append(min(children_dist))
        node = node.children[children_dist.index(min(children_dist))]
        if node.is_leaf():
            contin = False
    return chain


def compute_chain_stats(tre, order=4):
    """
    Returns mean, min, deciles and max of all transmission chains of given order
    :param tre: ete3.Tree, tree on which these metrics are computed
    :param order: int, order of transmission chain
    :return: list of floats
    """
    chain_sumlengths = []
    for node in tre.traverse():
        if getattr(node, 'height', False) > (order - 1):
            node_chain = compute_chain(node, order=order)
            if len(node_chain) == order:
                chain_sumlengths.append(sum(node_chain))
    sumstats_chain = [len(chain_sumlengths)]
    if len(chain_sumlengths) > 1:
        # mean
        sumstats_chain.append(np.mean(chain_sumlengths))
        # deciles
        sumstats_chain.extend(np.percentile(chain_sumlengths, np.arange(0, 101, 10)))
        # var
        sumstats_chain.append(np.var(chain_sumlengths))
    else:
        sumstats_chain = [0 for _ in range(len(col_chains))]
    return sumstats_chain
