import gzip
import os

import numpy as np
import pandas as pd
from ete3 import Tree
from pastml.tree import remove_certain_leaves
from phylodeep import FULL, BDEI, BD, SUMSTATS

from phylodeep.paramdeep import paramdeep

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extracts subtrees of given size.")
    parser.add_argument('--min_n', default=50, type=int, help="desired min total number of leaves")
    parser.add_argument('--max_n', default=199, type=int, help="desired max total number of leaves")
    parser.add_argument('--in_log', default="/home/azhukova/projects/phylodeep/data_publication/Fig_3/predicted_values/BDEI_large/TARGET.csv.gz", type=str, help="log file")
    parser.add_argument('--in_nwk', default="/home/azhukova/projects/phylodeep/data_publication/Fig_3/test_trees/BDEI_large_100.nwk.gz", type=str, help="forest file")
    parser.add_argument('--out_log', default="/home/azhukova/projects/phylodeep/data_publication/Supp_Fig_8/predicted_values/BDEI_subtree/TARGET.csv.gz", type=str, help="log file")
    parser.add_argument('--out_est_CNN', default="/home/azhukova/projects/phylodeep/data_publication/Supp_Fig_8/predicted_values/BDEI_subtree/CNN_CBLV.csv.gz", type=str, help="estimated paramers CNN")
    parser.add_argument('--out_est_FFNN', default="/home/azhukova/projects/phylodeep/data_publication/Supp_Fig_8/predicted_values/BDEI_subtree/FFNN_SS.csv.gz", type=str, help="estimated paramers FFNN")
    parser.add_argument('--out_nwk', default="/home/azhukova/projects/phylodeep/data_publication/Supp_Fig_8/test_trees/BDEI_subtree_100.nwk.gz", type=str, help="forest file")
    parser.add_argument('--root', action='store_true', help="extract the root subtree")
    parser.add_argument('--model', choices=(BD, BDEI), default=BDEI, help="model")
    params = parser.parse_args()

    df = pd.read_csv(params.in_log, sep=',')
    df_CNN = pd.DataFrame()
    df_FFNN = pd.DataFrame()
    with gzip.open(params.in_nwk, 'rt') as f:
        nwks = f.read().split(';')[:-1]
    out_nwks = []
    sampling_col = next(c for c in df.columns if 'sampling' in c)
    for nwk, index in zip(nwks, df.index):
        tree = Tree(nwk.strip('\n') + ';', format=3)
        if params.root:
            for n in tree.traverse('preorder'):
                n.add_feature('T', (0 if n.is_root() else getattr(n.up, 'T')) + n.dist)
            tips = sorted([_ for _ in tree], key=lambda _: getattr(_, 'T'))
            n = params.max_n - 9  # int(np.round(params.min_n + np.random.random() * (params.max_n - params.min_n), 0))
            tree = remove_certain_leaves(tree, lambda _: _ not in tips[:n])
            df.loc[index, 'cluster_tree_size'] = len(tree)
            clade = tree.write(format=3, format_root_node=True)
            out_nwks.append(clade)
            temp_nwk = params.out_nwk + '{}.nwk'.format(index)
            with open(temp_nwk, 'w') as f:
                f.write(clade)
            print('{}: Extracted a root clade of size {}'.format(index, len(tree)))
            df_CNN = df_CNN.append(paramdeep(temp_nwk, df.loc[index, sampling_col], model=params.model,
                                             vector_representation=FULL, ci_computation=False))
            df_FFNN = df_FFNN.append(paramdeep(temp_nwk, df.loc[index, sampling_col], model=params.model,
                                               vector_representation=SUMSTATS, ci_computation=False))
            os.remove(temp_nwk)
        else:
            todo = [tree]
            ts = []
            df_CNN_local = pd.DataFrame()
            df_FFNN_local = pd.DataFrame()
            while todo:
                n = todo.pop()
                if len(n) > params.max_n:
                    todo.extend(n.children)
                elif len(n) >= params.min_n:
                    n.detach()
                    clade = n.write(format=3, format_root_node=True)
                    out_nwks.append(clade)
                    ts.append(len(n))
                    temp_nwk = params.out_nwk + '{}.nwk'.format(index)
                    with open(temp_nwk, 'w') as f:
                        f.write(clade)
                    df_CNN_local = df_CNN_local.append(
                        paramdeep(temp_nwk, df.loc[index, sampling_col], model=params.model,
                                  vector_representation=FULL, ci_computation=False))
                    df_FFNN_local = df_FFNN_local.append(
                        paramdeep(temp_nwk, df.loc[index, sampling_col], model=params.model,
                                  vector_representation=SUMSTATS, ci_computation=False))
                    os.remove(temp_nwk)
            df.loc[index, 'cluster_tree_size'] = np.mean(ts)
            df.loc[index, 'tree_number'] = len(ts)
            print('{}: Extracted {} clades of sizes {}-{}'.format(index, len(ts), min(ts), max(ts)))
            df_CNN = df_CNN.append(df_CNN_local.describe().loc[['mean'], :])
            df_FFNN = df_FFNN.append(df_FFNN_local.describe().loc[['mean'], :])

    df.to_csv(params.out_log, compression='gzip', index=False)
    df_CNN.to_csv(params.out_est_CNN, compression='gzip', index=False)
    df_FFNN.to_csv(params.out_est_FFNN, compression='gzip', index=False)
    with gzip.open(params.out_nwk, 'wb') as f:
        f.write('\n'.join(out_nwks).encode())
