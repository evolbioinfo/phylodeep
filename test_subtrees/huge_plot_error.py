from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.offsetbox import TextArea, HPacker, AnchoredOffsetbox

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plots errors.")
    parser.add_argument('--estimates', type=str, help="estimated parameters")
    parser.add_argument('--png', type=str, help="plot")
    params = parser.parse_args()

    df = pd.read_csv(params.estimates, sep='\t', index_col=0)

    real_df = df.loc[df['type'] == 'real', :]
    df = df.loc[df['type'] != 'real', :]
    types = sorted(df['type'].unique(),
                   key=lambda _: (0 if '(large)' in _ else 1, 0 if 'subtree' in _ else 1, 0 if 'FFNN' in _ else 1))
    pars = [c for c in df.columns if c not in ['type', 'p']]

    for type in types:
        mask = df['type'] == type
        for par in pars:
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
            par2type2avg_error[par][type] = \
                '{:.2f} ({:.2f})'.format(np.mean(np.abs(df.loc[df['type'] == type, '{}_error'.format(par)])),
                                         np.mean(df.loc[df['type'] == type, '{}_error'.format(par)]))

    ERROR_COL = 'relative error'
    plot_df = pd.DataFrame(data=data, columns=['parameter', ERROR_COL, 'config'])
    # CNN-large, FFNN-subtrees, CNN-subtrees, FFNN-huge-on-large
    palette = [sns.color_palette("colorblind")[7]] + sns.color_palette("colorblind")[3:]
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
        boxes = [TextArea(text, textprops=dict(color=color, ha='center', va='center', fontsize='x-small',
                                               fontweight='bold'))
                 for text, color in zip((par2type2avg_error[par][_] for _ in types), palette)]
        return HPacker(children=boxes, align="center", pad=0, sep=12)
    xbox = HPacker(children=[get_xbox(par) for par in pars], align="center", pad=0, sep=50 if len(pars) <= 3 else 65)
    anchored_xbox = AnchoredOffsetbox(loc=3, child=xbox, pad=0, frameon=False,
                                      bbox_to_anchor=(0.08 if len(pars) == 2 else 0.06 if len(pars) == 3 else 0.05,
                                                      -0.05),
                                      bbox_transform=ax.transAxes, borderpad=0.)
    ax.set_xlabel('')
    ax.add_artist(anchored_xbox)
    leg = ax.legend()

    fig.set_size_inches(5 * len(pars), 8)
    plt.tight_layout()
    plt.savefig(params.png, dpi=300)
