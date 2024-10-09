#!/usr/bin/env python3

# import packages
import sys
import argparse
import io
import pandas as pd
import numpy as np

from ete3 import Tree

sys.setrecursionlimit(100000)

# import file with a table of parameter values and the maximum time of simulation (large number e.g. 500)
parser = argparse.ArgumentParser(description='Generates a tree')
parser.add_argument('inputFile', help='an input file with parameters for the tree')
parser.add_argument('maxTime', type=float, help='an input float for the maximal simulation time')

args = parser.parse_args()

with open(args.inputFile, 'r') as des:
    des_data = des.read()
des.close()

design = pd.read_table(io.StringIO(des_data), index_col='index')
# design = design.loc[:,['R_nought_null', 'transmission_rate', 'removal_rate', 'sample_proba', 'pseudo_sampling_rate']]

design = design.loc[:, ['R_nought', 'tr_rate_1_1', 'tr_rate_2_2', 'tr_rate_1_2', 'tr_rate_2_1', 'removal_rate',
                        'sampling_proba', 'R_nought_1', 'R_nought_2', 'R_nought_verif', 'tree_size', 'x_transmission',
                        'fraction_1', 'infectious_period']]

nb_samples = len(design)

# should not be constraining (eg Infinity), depends on your experiment
maxTime = args.maxTime

# initiate attributes of nodes
# reasons why a branch stops (transmission, removed/sampled/unsampled tips before the end of simulations)
STOP_REASON = 'stop_reason'
STOP_UNKNOWN = 0
STOP_TRANSMISSION = 1
STOP_REMOVAL_WOS = 2
STOP_SAMPLING = 3
STOP_TIME = 4

# infectious types: their meaning (eg normal/superspreading) depends on the values of parameters
I_1 = 1
I_2 = 2
I_T = 'i_t'

HISTORY = 'history'
SAMPLING = 'sampling'
TRANSMISSION = 'transmission'
DIST_TO_START = 'DIST_TO_START'
PROCESSED = 'processed'


def simulate_bdss_tree_gillespie(tr_r11, tr_r12, tr_r21, tr_r22, removal_r, sampling_p, max_s, max_t,
                                 fraction_1):
    """
    Simulates the tree evolution with heterogeneous hosts (of type t1 and t2) based on the given transmission rates,
     removal rate, sampling probabilities and number of tips
    :param tr_r11: float of transmission rate from t1 to t1 type spreader
    :param tr_r12: float of transmission rate from t1 to t2 type spreader
    :param tr_r21: float of transmission rate from t2 to t1 type spreader
    :param tr_r22: float of transmission rate from t2 to t2 type spreader
    :param removal_r: float of removal rate of both t1 and t2 type spreaders
    :param sampling_p: float, between 0 and 1, probability for removed t1 tip/spreader to be immediately sampled
    :param max_s: int, maximum number of sampled leaves in a tree
    :param max_t: float, maximum time from root simulation
    :param fraction_1: float, between 0 and 1, fraction of type 1 spreaders
    :return: the simulated tree (ete3.Tree).
    """
    right_size = 0
    trial = 0

    def initialize(init_type):
        if init_type == 1:
            metrics['number_inf1_leaves'] = 1
            leaves_dict['inf1_leaves'] = root.get_leaves()

            metrics['total_inf1'] = 1
        else:
            metrics['number_inf2_leaves'] = 1
            leaves_dict['inf2_leaves'] = root.get_leaves()

            metrics['total_inf2'] = 1
        return None

    def update_rates(rates, metrics_dc):
        """
        updates rates dictionary
        :param rates: dict, all rate values at previous step
        :param metrics_dc: dict, counts of different individuals, list(s) of different branch types
        :return:
        """
        rates['transmission_rate_1_1_i'] = tr_r11 * metrics_dc['number_inf1_leaves']
        rates['transmission_rate_1_2_i'] = tr_r12 * metrics_dc['number_inf1_leaves']
        rates['transmission_rate_2_1_i'] = tr_r21 * metrics_dc['number_inf2_leaves']
        rates['transmission_rate_2_2_i'] = tr_r22 * metrics_dc['number_inf2_leaves']

        rates['removal_rate_t1_i'] = removal_r * metrics_dc['number_inf1_leaves']
        rates['removal_rate_t2_i'] = removal_r * metrics_dc['number_inf2_leaves']

        return None

    def transmission(t_donor, t_recipient):
        if t_donor == 1:
            # which leaf will be affected by the event?
            nb_which_leaf = np.random.randint(0, metrics['number_inf1_leaves'])
            which_leaf = leaves_dict['inf1_leaves'][nb_which_leaf]
            del leaves_dict['inf1_leaves'][nb_which_leaf]
        else:
            # which leaf will be affected by the event?
            nb_which_leaf = np.random.randint(0, metrics['number_inf2_leaves'])
            which_leaf = leaves_dict['inf2_leaves'][nb_which_leaf]
            del leaves_dict['inf2_leaves'][nb_which_leaf]

        # which_leaf becomes an internal node
        which_leaf.dist = abs(time - which_leaf.DIST_TO_START)
        which_leaf.add_feature(DIST_TO_START, time)
        which_leaf.add_feature(STOP_REASON, STOP_TRANSMISSION)

        # which_leaf gives birth to recipent and donor
        recipient, donor = which_leaf.add_child(dist=0), which_leaf.add_child(dist=0)
        recipient.add_feature(DIST_TO_START, which_leaf.DIST_TO_START)
        donor.add_feature(DIST_TO_START, which_leaf.DIST_TO_START)

        # add recipient to its lists, add it its attributes:
        if t_recipient == 1:
            recipient.add_feature(I_T, I_1)
            leaves_dict['inf1_leaves'].append(recipient)
            metrics['total_inf1'] += 1
            metrics['number_inf1_leaves'] += 1
        else:
            recipient.add_feature(I_T, I_2)
            leaves_dict['inf2_leaves'].append(recipient)
            metrics['total_inf2'] += 1
            metrics['number_inf2_leaves'] += 1

        # add donor to its lists, add it its attributes:
        if t_donor == 1:
            donor.add_feature(I_T, I_1)
            leaves_dict['inf1_leaves'].append(donor)
        else:
            donor.add_feature(I_T, I_2)
            leaves_dict['inf2_leaves'].append(donor)

        metrics['total_branches'] += 1
        return None

    def removal(removed_type):
        """
        updates the tree, the metrics and leaves_dict following a removal event
        :param removed_type: int, either 1: a branch of type 1 undergoes removal; 2: a branch of type 2 undergoes
        removal
        :return:
        """
        # which leaf is removed?: "which_leaf"
        if removed_type == 1:
            nb_which_leaf = np.random.randint(0, metrics['number_inf1_leaves'])
            which_leaf = leaves_dict['inf1_leaves'][nb_which_leaf]
            del leaves_dict['inf1_leaves'][nb_which_leaf]
            metrics['number_inf1_leaves'] -= 1
        else:
            nb_which_leaf = np.random.randint(0, metrics['number_inf2_leaves'])
            which_leaf = leaves_dict['inf2_leaves'][nb_which_leaf]
            del leaves_dict['inf2_leaves'][nb_which_leaf]
            metrics['number_inf2_leaves'] -= 1

        # which_leaf becomes a tip
        which_leaf.dist = abs(time - which_leaf.DIST_TO_START)
        which_leaf.add_feature(DIST_TO_START, time)
        which_leaf.add_feature(PROCESSED, True)

        # was which_leaf sampled?
        if np.random.rand() < sampling_p:
            metrics['number_sampled'] += 1
            which_leaf.add_feature(STOP_REASON, STOP_SAMPLING)
            metrics['total_removed'] += 1
            if removed_type == 1:
                metrics['sampled_inf1'] += 1
            else:
                metrics['sampled_inf2'] += 1
        else:
            which_leaf.add_feature(STOP_REASON, STOP_REMOVAL_WOS)
            metrics['total_removed'] += 1
        return None

    # up to 100 times retrial of simulation until reaching correct size
    while right_size == 0 and trial < 100:

        root = Tree(dist=0)
        root.add_feature(DIST_TO_START, 0)

        # initiate the time of simulation
        time = 0

        # INITIATE: metrics counting leaves and branches of different types, leaves_dict storing all leaves alive, and
        # rates_i with all rates at given time, for Gillespie algorithm
        metrics = {'total_branches': 1, 'total_removed': 0, 'number_sampled': 0, 'total_inf1': 0, 'total_inf2': 0,
                   'sampled_inf1': 0, 'sampled_inf2': 0, 'number_inf1_leaves': 0, 'number_inf2_leaves': 0}
        leaves_dict = {'inf1_leaves': [], 'inf2_leaves': []}

        rates_i = {'removal_rate_t1_i': 0, 'removal_rate_t2_i': 0, 'transmission_rate_1_1_i': 0,
                   'transmission_rate_1_2_i': 0, 'transmission_rate_2_1_i': 0, 'transmission_rate_2_2_i': 0}

        # INITIATE: of which type is the first branch?
        # first individual will be of type 1 with probability frac_1 (frequence of type 1 at equilibrium)
        if np.random.rand() < fraction_1:
            initialize(1)

        else:
            initialize(2)

        # simulate while [1] the epidemics do not go extinct, [2] given number of patients were not sampled,
        # [3] maximum time of simulation was not reached
        while (metrics['number_inf1_leaves'] + metrics['number_inf2_leaves']) > 0 \
                and (metrics['number_sampled'] < max_s) and (time < max_t):
            # first we need to re-calculate the rates and take its sum
            update_rates(rates_i, metrics)
            sum_rates_i = sum(rates_i.values())

            # when does the next event take place?
            time_to_next = np.random.exponential(1 / sum_rates_i, 1)[0]
            time = time + time_to_next

            # which event will happen
            random_event = np.random.uniform(0, 1, 1) * sum_rates_i

            if random_event < rates_i['transmission_rate_1_1_i']:
                # there will be a transmission event from t1 to t1 type spreader
                transmission(1, 1)

            elif random_event < (rates_i['transmission_rate_1_1_i'] + rates_i['transmission_rate_1_2_i']):
                # there will be a transmission event from t1 to t2 type spreader
                transmission(1, 2)

            elif random_event < (rates_i['transmission_rate_1_1_i'] + rates_i['transmission_rate_1_2_i'] +
                                 rates_i['transmission_rate_2_1_i']):
                # there will be a transmission event from t2 to t1 type spreader
                transmission(2, 1)

            elif random_event < (rates_i['transmission_rate_1_1_i'] + rates_i['transmission_rate_1_2_i'] +
                                 rates_i['transmission_rate_2_1_i'] + rates_i['transmission_rate_2_2_i']):
                transmission(2, 2)

            elif random_event < \
                    (sum_rates_i - rates_i['removal_rate_t2_i']):
                # there will be a removal event of t1 type spreader
                removal(removed_type=1)

            else:
                removal(removed_type=2)
                # there will be a removal event of a t2 type spreader

        # tag non-removed tips at the end of simulation
        for leaflet in root.get_leaves():
            if getattr(leaflet, STOP_REASON, False) != 2 and getattr(leaflet, STOP_REASON, False) != 3:
                leaflet.dist = abs(time - leaflet.DIST_TO_START)
                leaflet.add_feature(DIST_TO_START, time)
                leaflet.add_feature(STOP_REASON, STOP_TIME)

        if metrics['number_sampled'] == max_s:
            right_size = 1
        else:
            trial += 1

    # statistics on the simulation
    del metrics['number_inf1_leaves']
    del metrics['number_inf2_leaves']
    vector_count = list(metrics.values())
    vector_count.extend([time, trial])

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

col = ['tree']
forest_export = pd.DataFrame(index=design.index, columns=col)

col2 = ['total_leaves', 'removed_leaves', 'sampled_leaves', 'total_inf1_leaves', 'total_inf2_leaves',
        'sampled_inf1_leaves', 'sampled_inf2_leaves', 'time_of_simulation', 'nb_trials']
stats_export = pd.DataFrame(index=design.index, columns=col2)

# SIMULATE the trees
for experiment_id in range(nb_samples):
    params = design.iloc[experiment_id, ]

    # simulation
    tr, vector_counter = simulate_bdss_tree_gillespie(tr_r11=params[1], tr_r12=params[3], tr_r21=params[4],
                                                      tr_r22=params[2], removal_r=params[5], sampling_p=params[6],
                                                      max_s=params[10], max_t=maxTime, fraction_1=params[12])
    # for display purposes
    i = 0
    for node in tr.traverse("levelorder"):
        node.name = "n" + str(i)
        i += 1

    # remove unsampled tips
    tr = remove_certain_leaves(tr, to_remove=lambda node: getattr(node, STOP_REASON) != STOP_SAMPLING)
    if tr is not None:
        forest_export.iloc[experiment_id][0] = tr.write(features=['DIST_TO_START', 'stop_reason', 'i_t'],
                                                        format_root_node=True, format=3)
    else:
        forest_export.iloc[experiment_id][0] = "NA"

    stats_export.iloc[experiment_id] = vector_counter

# EXPORT
# subpopulations to export as csv
stats_export.to_csv(path_or_buf="subpopulations.txt", sep='\t', index=True, header=True)

# For the pipe : export to stdout
sys.stdout.write(forest_export.to_csv(sep='\t', index=True, header=True))
