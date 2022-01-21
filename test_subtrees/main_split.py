import gzip
import os

import pandas as pd

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extracts subtrees of given size.")
    parser.add_argument('--in_nwk', required=True, type=str, help="forest file")
    parser.add_argument('--in_log', required=True, type=str, help="forest parameter file")
    parser.add_argument('--out_nwk', nargs='+', type=str, help="tree files")
    parser.add_argument('--out_log', nargs='+', type=str, help="tree parameter files")
    params = parser.parse_args()

    df = pd.read_csv(params.in_log, sep=',')
    with gzip.open(params.in_nwk, 'rt') as f:
        nwks = f.read().split(';')[:-1]
    for (i, tree, nwk, log) in zip(df.index, nwks, params.out_nwk, params.out_log):
        with open(nwk, 'w+') as f:
            f.write('{};\n'.format(tree))
        df.loc[[i], :].to_csv(log, index=False)
