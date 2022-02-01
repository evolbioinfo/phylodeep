import pandas as pd


def get_col2col(columns):
    res = {}
    for c in columns:
        if 'sampling' in c:
            res['p'] = c
        elif 'nfecti' in c:
            res['infectious period'] = c
        elif 'ransmi' in c:
            res['SS transmission ratio'] = c
        elif 'ractio' in c:
            res['SS fraction'] = c
        elif 'ncubation' in c:
            res['incubation time'] = c
        elif 'R' in c:
            res['R0'] = c
    return res


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plots errors.")
    parser.add_argument('--estimated_FFNN', type=str, help="estimated parameters")
    parser.add_argument('--estimated_CNN', type=str, help="estimated parameters")
    parser.add_argument('--estimated_CNN_subtrees', type=str, help="estimated parameters")
    parser.add_argument('--estimated_FFNN_subtrees', type=str, help="estimated parameters")
    parser.add_argument('--estimated_CNN_large', type=str, help="estimated parameters", default=None)
    parser.add_argument('--estimated_FFNN_large', type=str, help="estimated parameters", default=None)
    parser.add_argument('--real', type=str, help="real parameters for subtrees")
    parser.add_argument('--real_b', type=str, default=None, help="real parameters")
    parser.add_argument('--tab', type=str, help="estimate table")
    parser.add_argument('--small', action="store_true", help="estimate table")
    parser.add_argument('--model', type=str, choices=['BD', 'BDEI', 'BDSS'], help="model")
    params = parser.parse_args()

    if params.model == 'BD':
        ps = ['R0', 'infectious period']
    elif params.model == 'BDEI':
        ps = ['R0', 'infectious period', 'incubation time']
    else:
        ps = ['R0', 'SS transmission ratio', 'SS fraction', 'infectious period']

    df = pd.DataFrame(columns=['type'] + ps)

    for (rdf, label) in zip((params.real, params.real_b if params.real_b else params.real), ('real-subtrees', 'real')):
        rdf = pd.read_csv(rdf, header=0)
        rdf.index = rdf.index.map(int)
        col2col = get_col2col(rdf.columns)
        cols = [col2col[_] for _ in ps] + [col2col['p']]
        rdf = rdf[cols]
        rdf.columns = ps + ['p']
        rdf['type'] = label
        rdf.index = rdf.index.map(lambda _: '{}.{}'.format(_, label))
        df = df.append(rdf)


    for (label, csv) in zip(('CNN', 'FFNN',
                             'CNN (on {})'.format('large' if params.small else 'small'),
                             'FFNN (on {})'.format('large' if params.small else 'small'),
                             'CNN (subtrees)', 'FFNN (subtrees)'),
                            (params.estimated_CNN_large, params.estimated_FFNN_large,
                             params.estimated_CNN, params.estimated_FFNN,
                             params.estimated_CNN_subtrees, params.estimated_FFNN_subtrees)):
        if csv:
            dldf = pd.read_csv(csv, header=0)
            dldf.index = dldf.index.map(int)
            col2col = get_col2col(dldf.columns)
            cols = [col2col[_] for _ in ps]
            dldf = dldf[cols]
            dldf.columns = ps
            dldf['type'] = label
            dldf.index = dldf.index.map(lambda _: '{}.{}'.format(_, label))
            df = df.append(dldf)

    df.index = df.index.map(lambda _: int(_.split('.')[0]))
    df.sort_index(inplace=True)
    df[['type', 'p'] + ps].to_csv(params.tab, sep='\t')
