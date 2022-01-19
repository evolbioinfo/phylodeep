import logging
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.offsetbox import TextArea, HPacker, AnchoredOffsetbox, VPacker
from statsmodels.stats.weightstats import CompareMeans


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plots errors.")
    parser.add_argument('--estimates', type=str, help="estimated parameters")
    parser.add_argument('--png', type=str, help="plot")
    params = parser.parse_args()

    df = pd.read_csv(params.estimates, sep='\t', index_col=0)

    real_df = df.loc[df['type'] == 'real', :]
    real_df_subtree = df.loc[df['type'] == 'real-subtrees', :]
    df = df.loc[(df['type'] != 'real') & (df['type'] != 'real-subtrees'), :]
    types = sorted(df['type'].unique(), key=lambda _: (_[0] != 'B', len(_)))
    pars = [c for c in df.columns if c not in ['type', 'p']]

    for type in types:
        mask = df['type'] == type
        for par in pars:
            if 'subtree' in type:
                df.loc[mask, '{}_error'.format(par)] = (df.loc[mask, par] - real_df_subtree[par]) / real_df_subtree[par]
            else:
                df.loc[mask, '{}_error'.format(par)] = (df.loc[mask, par] - real_df[par]) / real_df[par]

    plt.clf()
    n_types = len(types)
    print(types)
    fig, ax = plt.subplots(figsize=(5 * len(pars), 8))
    rc = {'font.size': 12, 'axes.labelsize': 10, 'legend.fontsize': 10, 'axes.titlesize': 10, 'xtick.labelsize': 10,
          'ytick.labelsize': 10}
    sns.set(style="whitegrid")
    sns.set(rc=rc)

    abs_error_or_1 = lambda _: min(abs(_), 1)

    data = []
    par2type2avg_error = defaultdict(lambda: dict())
    for type in types:
        for par in pars:
            data.extend([[par, _, type]
                         for _ in df.loc[df['type'] == type, '{}_error'.format(par)].apply(abs_error_or_1)])
            par2type2avg_error[par][type] = '{:.2f} ({:.2f})'.format(np.mean(np.abs(df.loc[df['type'] == type, '{}_error'.format(par)])), np.mean(df.loc[df['type'] == type, '{}_error'.format(par)]))

    par2types2pval = defaultdict(lambda: dict())
    for par in pars:
        for i in range(n_types):
            type_1 = types[i]
            for j in range(i + 1, n_types):
                type_2 = types[j]
                pval_abs = CompareMeans.from_data(data1=df.loc[df['type'] == type_1, '{}_error'.format(par)].apply(np.abs),
                                                  data2=df.loc[df['type'] == type_2, '{}_error'.format(par)].apply(np.abs)).ztest_ind()[1]
                par2types2pval[par][(type_1, type_2)] = pval_abs
                if pval_abs < 0.05:
                    print(par, type_1, type_2, np.round(pval_abs, 3))

    ERROR_COL = 'relative error'
    plot_df = pd.DataFrame(data=data, columns=['parameter', ERROR_COL, 'config'])
    palette = sns.color_palette("colorblind")
    sns.swarmplot(x="parameter", y=ERROR_COL, palette=palette, data=plot_df, alpha=.8, hue="config", ax=ax, dodge=True)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    min_error = min(min(df['{}_error'.format(_)]) for _ in pars)
    max_error = max(max(df['{}_error'.format(_)]) for _ in pars)
    abs_error = max(max_error, abs(min_error))
    ax.set_yticks(list(np.arange(0, min(1.1, abs_error + 0.1), step=0.2 if abs_error >= 1 else 0.1)))
    if abs_error >= 1:
        ax.set_yticklabels(['{:.1f}'.format(_) for _ in np.arange(0, 1.0, step=0.2)] + [u"\u22651"])
    ax.set_ylim(0, min(1.1, abs_error + 0.1))
    ax.yaxis.grid()

    def get_xbox(par):
        boxes = [TextArea(text, textprops=dict(color=color, ha='center', va='center', fontsize='x-small', fontweight='bold'))
                 for text, color in zip((par2type2avg_error[par][_] for _ in types), palette)]
        return HPacker(children=boxes, align="center", pad=0, sep=3)
    xbox = HPacker(children=[get_xbox(par) for par in pars], align="center", pad=0, sep=14)
    anchored_xbox = AnchoredOffsetbox(loc=3, child=xbox, pad=0, frameon=False,
                                      bbox_to_anchor=(0.06, -0.14),
                                      bbox_transform=ax.transAxes, borderpad=0.)
    ax.set_xlabel('')
    ax.add_artist(anchored_xbox)

    def get_pbox(par):
        EMPTY = ' ' * 10
        LONG_DASH = u"\u2014"
        FILLED = LONG_DASH * 10
        boxes = []
        for i in range(n_types - 1):
            type_1 = types[i]
            s = EMPTY * max(0, i)
            for j in range(i + 1, n_types):
                type_2 = types[j]
                pval = par2types2pval[par][(type_1, type_2)]
                if pval < 0.05:
                    boxes.append(TextArea(s + LONG_DASH * 3 + '{:.3f}'.format(pval) + LONG_DASH * 3 + EMPTY * (n_types - j - 1),
                                          textprops=dict(color='black', ha='center', va='center', fontsize='x-small', fontweight='bold', family='monospace')))
                else:
                    boxes.append(TextArea(EMPTY * n_types, textprops=dict(color='black', ha='center', va='center', fontsize='x-small', fontweight='bold', family='monospace')))
                s += FILLED
        return VPacker(children=list(reversed(boxes)), mode='equal', pad=0, sep=3) if len(boxes) > 1 else boxes[0]
    xbox = HPacker(children=[get_pbox(par) for par in pars], align="center", pad=0, sep=43)
    anchored_xbox = AnchoredOffsetbox(loc=3, child=xbox, pad=0, frameon=False,
                                      bbox_to_anchor=(0.22 if len(pars) == 2 else 0.16, 1),
                                      bbox_transform=ax.transAxes, borderpad=0.)
    ax.set_xlabel('')
    ax.add_artist(anchored_xbox)



    leg = ax.legend()

    fig.set_size_inches(5 * len(pars), 8)
    plt.tight_layout()
    plt.savefig(params.png, dpi=300)
