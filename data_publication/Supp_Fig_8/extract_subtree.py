import gzip
import os

import pandas as pd
from ete3 import Tree

from phylodeep import FULL, BDEI, BD, SUMSTATS
from phylodeep.paramdeep import paramdeep
from phylodeep.tree_utilities import extract_clusters

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extracts subtrees of given size.")
    parser.add_argument('--min_n', default=50, type=int, help="desired min total number of leaves")
    parser.add_argument('--max_n', default=199, type=int, help="desired max total number of leaves")
    parser.add_argument('--in_log', type=str, help="log file")
    parser.add_argument('--in_nwk', type=str, help="forest file")
    parser.add_argument('--out_log', type=str, help="log file")
    parser.add_argument('--out_est_CNN', type=str, help="estimated paramers CNN")
    parser.add_argument('--out_est_FFNN', type=str, help="estimated paramers FFNN")
    parser.add_argument('--out_nwk', type=str, help="forest file")
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
        tree_size = len(tree)
        ts = []
        df_CNN_local = pd.DataFrame()
        df_FFNN_local = pd.DataFrame()
        for clade in extract_clusters(tree, params.min_n, params.max_n):
            ts.append(len(clade))
            temp_nwk = params.out_nwk + '{}.nwk'.format(index)
            clade = clade.write(format=3, format_root_node=True)
            out_nwks.append(clade)
            with open(temp_nwk, 'w') as f:
                f.write(clade)
            df_CNN_local = df_CNN_local.append(
                paramdeep(temp_nwk, df.loc[index, sampling_col], model=params.model,
                          vector_representation=FULL, ci_computation=False))
            df_FFNN_local = df_FFNN_local.append(
                paramdeep(temp_nwk, df.loc[index, sampling_col], model=params.model,
                          vector_representation=SUMSTATS, ci_computation=False))
            os.remove(temp_nwk)
        for (global_df, local_df) in ((df_CNN, df_CNN_local), (df_FFNN, df_FFNN_local)):
            cols = list(local_df.columns)
            local_df['weight'] = ts
            local_df['weight'] /= sum(ts)
            for col in cols:
                global_df.loc[index, col] = (local_df[col] * local_df['weight']).sum()
        df.loc[index, 'cluster_tips'] = sum(ts)
        df.loc[index, 'cluster_branches'] = sum((2 * _ - 1) for _ in ts)
        df.loc[index, 'tree_branches'] = (2 * tree_size - 1)
        print('{}: Extracted {} clades of sizes {}-{}'.format(index, len(ts), min(ts), max(ts)))

    df.to_csv(params.out_log, compression='gzip', index=False)
    df_CNN.to_csv(params.out_est_CNN, compression='gzip', index=False)
    df_FFNN.to_csv(params.out_est_FFNN, compression='gzip', index=False)
    with gzip.open(params.out_nwk, 'wb') as f:
        f.write('\n'.join(out_nwks).encode())
