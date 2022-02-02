import gzip
import glob
import re

import numpy as np
import pandas as pd

files = glob.glob('/home/azhukova/projects/phylodeep/data_publication/Supp_Fig_10/predicted_values/*_huge/FFNN_SS_log.csv.gz')

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Splits a forest into trees.")
    parser.add_argument('--in_log', nargs='+', default=files, type=str, help="tree files")
    params = parser.parse_args()

    for f in files:
        print(re.findall(r'(\w+)_huge', f)[0])
        percentages = []
        with gzip.open(f, 'rt') as f:
            for line in f:
                n_sub, n_all = (int(_) for _ in re.findall(r'\d+', line))
                percentages.append(100 * n_sub / n_all)
        percentages = np.array(percentages)
        print(percentages.mean(), percentages.min(), '-', percentages.max())
