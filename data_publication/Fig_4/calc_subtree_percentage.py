import gzip
import glob
import re
import time

import numpy as np
from phylodeep.tree_utilities import extract_clusters, read_tree, MIN_TREE_SIZE_SMALL, MIN_TREE_SIZE_LARGE, MIN_TREE_SIZE_HUGE
from ete3 import Tree

def mean(l):
    return sum(l) / len(l)

TREES = 'test_trees/BDSS_huge_100.nwk.gz'

if __name__ == "__main__":

    percentages = []
    times = []
    with gzip.open(TREES, 'rt') as f:
        for nwk in f.read().strip().split(';'):
            if nwk:
                tre = read_tree(nwk + ';')
                n = 2 * len(tre) - 2
                m = 0
                t = time.time()
                for st in extract_clusters(tre, min_size=MIN_TREE_SIZE_LARGE, max_size=MIN_TREE_SIZE_HUGE - 1):
                    m += 2 * len(st) - 2
                times.append(time.time() - t)
                percentages.append(100 * m / n)
    print('Tree picker conserved {} [{}-{}] % of branches'.format(mean(percentages), min(percentages), max(percentages)))
    print('Tree picker took {} [{}-{}] s per tree ({} s for 100 trees)'.format(mean(times), min(times), max(times), sum(times)))
