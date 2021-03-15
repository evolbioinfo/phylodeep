#!/usr/bin/env python3

# import packages
import sys
import argparse
import io
import pandas as pd
import numpy as np

from ete3 import Tree


# import file with a table of parameter values

parser = argparse.ArgumentParser(description='Generates a tree')
parser.add_argument('inputFile', help='an input file with parameters for the tree')
parser.add_argument('maxTime', type=float, help='an input float for the maximal simulation time')

args = parser.parse_args()

# read experiment design
with open(args.inputFile, 'r') as des:
    des_data = des.read()
des.close()

design = pd.read_table(io.StringIO(des_data), index_col='index')

design = design.loc[:,
         ['R_nought', 'transmission_rate', 'removal_rate', 'sampling_proba', 'infectious_time', 'tree_size'
          ]]

nb_samples = len(design)

# should not be constraining (eg Infinity), depends on your experiment
maxTime = args.maxTime


sys.setrecursionlimit(100000)

# initiate attributes of nodes
# reasons why a branch stops (transmission, removed/sampled/unsampled tips before the end of simulations)
STOP_REASON = 'stop_reason'
STOP_UNKNOWN = 0
STOP_TRANSMISSION = 1
STOP_REMOVAL_WOS = 2
STOP_SAMPLING = 3
STOP_TIME = 4

HISTORY = 'history'
SAMPLING = 'sampling'
TRANSMISSION = 'transmission'
DIST_TO_START = 'DIST_TO_START'
PROCESSED = 'processed'


def simulate_bd_tree_gillespie(transmission_r, removal_r, sampling_p, max_s, max_t):
    """
    Simulates the tree evolution with infectious hosts based on the given transmission rate,
    removal rate, sampling probabilities and number of tips
    :param transmission_r: float of transmission rate
    :param removal_r: float of removal rate
    :param sampling_p: float, between 0 and 1, probability for removed leave to be immediately sampled
    :param max_s: maximum number of sampled leaves in a tree
    :param max_t: maximum time from root simulation
    :return: the simulated tree (ete3.Tree).
    """
    right_size = 0
    trial = 0

    def update_rates(rates, metrics_dc):
        """
        updates rates dictionary
        :param rates: dict, all rate values at previous step
        :param metrics_dc: dict, counts of different individuals, list(s) of different branch types
        :return:
        """
        rates['transmission_rate_i'] = transmission_r * metrics_dc['number_infectious_leaves']
        rates['removal_rate_i'] = removal_r * metrics_dc['number_infectious_leaves']
        rates['sum_rates_i'] = rates['transmission_rate_i'] + rates['removal_rate_i']
        return None

    def transmission_event(which_lf):
        """
        updates the tree, the metrics and leaves_dict following a tranmission event that affects which_lf
        :param which_lf: ete3.Tree, a leaf affected by the transmission event
        :return: void, modifies which_lf, leaves_dict and metrics
        """
        metrics['total_branches'] += 1
        metrics['number_infectious_leaves'] += 1
        # the leaf becomes a transmission node
        which_lf.add_feature(STOP_REASON, STOP_TRANSMISSION)

        # the leaf branches into two new branches
        recipient, donor = which_lf.add_child(dist=0), which_lf.add_child(dist=0)
        donor.add_feature(DIST_TO_START, which_lf.DIST_TO_START)
        recipient.add_feature(DIST_TO_START, which_lf.DIST_TO_START)

        # two new infectious leaves
        leaves_dict['infectious_leaves'].append(donor)
        leaves_dict['infectious_leaves'].append(recipient)
        return None

    def removal_event(which_lf):
        """
        updates the tree, the metrics and leaves_dict following a removal event that affects which_lf
        :param which_lf: ete3.Tree, a leaf affected by the transmission event
        :return: void, modifies which_lf, leaves_dict and metrics
        """
        metrics['number_infectious_leaves'] -= 1
        which_lf.add_feature(PROCESSED, True)

        # sampling upon removal?
        if np.random.rand() < sampling_p:
            metrics['number_sampled'] += 1
            which_lf.add_feature(STOP_REASON, STOP_SAMPLING)
            metrics['total_removed'] += 1
        # removal without sampling
        else:
            which_lf.add_feature(STOP_REASON, STOP_REMOVAL_WOS)
            metrics['total_removed'] += 1
        return None

    # up to 100 times retrial of simulation until reaching correct size
    while right_size == 0 and trial < 100:
        # start a tree
        root = Tree(dist=0)
        root.add_feature(DIST_TO_START, 0)

        # INITIATE: metrics counting leaves and branches of different types, leaves_dict storing all leaves alive, and
        # rates_i with all rates at given time, for Gillespie algorithm
        metrics = {'total_branches': 1, 'total_removed': 0, 'number_infectious_leaves': 1, 'number_sampled': 0}
        leaves_dict = {'infectious_leaves': root.get_leaves()}

        rates_i = {'removal_rate_i': 0, 'transmission_rate_i': 0, 'sum_rates_i': 0}
        time = 0

        # simulate while [1] the epidemics do not go extinct, [2] given number of patients were not sampled,
        # [3] maximum time of simulation was not reached
        while metrics['number_infectious_leaves'] > 0 and metrics['number_sampled'] < max_s and time < max_t:
            # first we need to re-calculate the rates and take its sum
            update_rates(rates_i, metrics)
            # when does next event take place?
            time_to_next = np.random.exponential(1 / rates_i['sum_rates_i'], 1)[0]
            time = time + time_to_next
            # which leaf will be affected?
            nb_which_leaf = int(np.floor(np.random.uniform(0, metrics['number_infectious_leaves'], 1)))
            which_leaf = leaves_dict['infectious_leaves'][nb_which_leaf]
            del leaves_dict['infectious_leaves'][nb_which_leaf]

            # now let us see which event will happen at next event
            random_event = np.random.uniform(0, 1, 1) * rates_i['sum_rates_i']
            which_leaf.dist = abs(time - which_leaf.DIST_TO_START)
            which_leaf.add_feature(DIST_TO_START, time)

            # transmission event
            if random_event < rates_i['transmission_rate_i']:
                transmission_event(which_leaf)
            else:
                removal_event(which_leaf)

        # at the end of simulation, tag the non-removed tips
        for leaflet in root.get_leaves():
            if getattr(leaflet, STOP_REASON, False) != 2 and getattr(leaflet, STOP_REASON, False) != 3:
                leaflet.dist = abs(time - leaflet.DIST_TO_START)
                leaflet.add_feature(DIST_TO_START, time)
                leaflet.add_feature(STOP_REASON, STOP_TIME)

        # we sampled correct number of patients
        if metrics['number_sampled'] == max_s:
            right_size = 1
        # we retry again a simulation
        else:
            trial += 1

    # statistics on the number of branches, removed tips, sampled tips, time of simulation and number of sim trials
    del metrics['number_infectious_leaves']
    vector_count = list(metrics.values())
    vector_count.extend([time, trial])

    # print(root.get_ascii(attributes=['dist', DIST_TO_START, STOP_REASON]))
    return root, vector_count


def _merge_node_with_its_child(nd, child=None, state_feature=STOP_REASON):
    if not child:
        child = nd.get_children()[0]
    nd_hist = getattr(nd, HISTORY, [(getattr(nd, state_feature, ''), 0)])
    nd_hist += [('!', nd.dist - sum(it[1] for it in nd_hist))] \
               + getattr(child, HISTORY, [(getattr(child, state_feature, ''), 0)])
    child.add_features(**{HISTORY: nd_hist})
    child.dist += nd.dist
    if nd.is_root():
        child.up = None
    else:
        parent = nd.up
        parent.remove_child(nd)
        parent.add_child(child)
    return child


def remove_certain_leaves(tre, to_remove=lambda node: False, state_feature=STOP_REASON):
    """
    Removes all the branches leading to naive leaves from the given tree.
    :param tre: the tree of interest (ete3 Tree)
    [(state_1, 0), (state_2, time_of_transition_from_state_1_to_2), ...]. Branch removals will be added as '!'.
    :param to_remove: a method to check if a leaf should be removed.
    :param state_feature: the node feature to store the state
    :return: the tree with naive branches removed (ete3 Tree) or None is all the leaves were naive in the initial tree.
    """

    for nod in tre.traverse("postorder"):
        # If this node has only one child branch
        # it means that the other child branch used to lead to a naive leaf and was removed.
        # We can merge this node with its child
        # (the child was already processed and either is a leaf or has 2 children).
        if len(nod.get_children()) == 1:
            merged_node = _merge_node_with_its_child(nod, state_feature=state_feature)
            if merged_node.is_root():
                tre = merged_node
        elif nod.is_leaf() and to_remove(nod):
            if nod.is_root():
                return None
            nod.up.remove_child(nod)
    return tre


# PREPARE EXPORT
# stock all trees in a list
forest = []

# col names of export statistics
col = ['tree']
forest_export = pd.DataFrame(index=design.index, columns=col)

col2 = ['total_leaves', 'removed_leaves', 'sampled_leaves', 'time_of_simulation', 'nb_trials']
stats_export = pd.DataFrame(index=design.index, columns=col2)

# subpopulations_export = pd.DataFrame(index=design.index, columns=col_trial)

# SIMULATE the trees
for experiment_id in range(nb_samples):

    params = design.iloc[experiment_id, ]

    # simulation
    tr, vector_counter = simulate_bd_tree_gillespie(transmission_r=params[1], removal_r=params[2], sampling_p=params[3],
                                                    max_s=params[5], max_t=maxTime)
    # for display purposes
    i = 0
    for node in tr.traverse("levelorder"):
        node.name = "n" + str(i)
        i += 1

    # STOCK the tree
    # remove unsampled tips
    tr = remove_certain_leaves(tr, to_remove=lambda node: getattr(node, STOP_REASON) != STOP_SAMPLING)
    if tr is not None:
        forest_export.iloc[experiment_id][0] = tr.write(
            features=['DIST_TO_START', 'stop_reason'], format_root_node=True,
            format=3)
    else:
        forest_export.iloc[experiment_id][0] = "NA"

    stats_export.iloc[experiment_id] = vector_counter


# EXPORT
# eventually for complete forest including unsampled tips: export to a file
# complete_forest_export.to_csv(path_or_buf="complete_forest_export.txt", sep='\t', index=True, header=True)

# subpopulations to export as csv
stats_export.to_csv(path_or_buf="subpopulations.txt", sep='\t', index=True, header=True)

# for the pipe : export to stdout
sys.stdout.write(forest_export.to_csv(sep='\t', index=True, header=True))
