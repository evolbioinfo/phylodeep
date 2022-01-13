import gzip

import numpy as np
import pandas as pd
from ete3 import Tree
from pastml.tree import remove_certain_leaves

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extracts subtrees of given size.")
    parser.add_argument('--min_n', required=True, type=int, help="desired min total number of leaves")
    parser.add_argument('--max_n', required=True, type=int, help="desired max total number of leaves")
    parser.add_argument('--in_log', required=False, default=None, type=str, help="log file")
    parser.add_argument('--in_nwk', required=False, default=None, type=str, help="forest file")
    parser.add_argument('--out_log', required=False, default=None, type=str, help="log file")
    parser.add_argument('--out_nwk', required=False, default=None, type=str, help="forest file")
    parser.add_argument('--root', action='store_true', help="extract the root subtree")
    params = parser.parse_args()

    df = pd.read_csv(params.in_log, sep=',')
    with gzip.open(params.in_nwk, 'rt') as f:
        nwks = f.read().split(';')[:-1]
    out_nwks = []
    for nwk, index in zip(nwks, df.index):
        tree = Tree(nwk.strip('\n') + ';', format=3)
        if params.root:
            for n in tree.traverse('preorder'):
                n.add_feature('T', (0 if n.is_root() else getattr(n.up, 'T')) + n.dist)
            tips = sorted([_ for _ in tree], key=lambda _: getattr(_, 'T'))
            n = int(np.round(params.min_n + np.random.random() * (params.max_n - params.min_n), 0))
            clade = remove_certain_leaves(tree, lambda _: _ not in tips[:n])
        else:
            todo = [tree]
            clade = None
            while todo:
                n = todo.pop()
                if len(n) > params.max_n:
                    todo.extend(n.children)
                elif len(n) >= params.min_n:
                    n.detach()
                    clade = n
                    break
        df.loc[index, 'tree_size'] = len(clade)
        out_nwks.append(clade.write(format=3, format_root_node=True))

    df.to_csv(params.out_log, index=False)
    with gzip.open(params.out_nwk, 'wb') as f:
        f.write('\n'.join(out_nwks).encode())
