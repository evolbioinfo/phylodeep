import math
import os

from tree_utilities import read_tree, subtree_picker, independent_picker, top_picker, subsampled_picker

STYLE_FILE_HEADER_TEMPLATE = """DATASET_STYLE

SEPARATOR TAB
DATASET_LABEL	{column}
COLOR	#ffffff

LEGEND_COLORS	{colours}
LEGEND_LABELS	{states}
LEGEND_SHAPES	{shapes}
LEGEND_TITLE	{column}

DATA
#NODE_ID TYPE   NODE  COLOR SIZE_FACTOR LABEL_OR_STYLE
"""

COLOURS = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e']


def generate_itol_style(tree, subtrees, itol_file, strategy):
    name2n = {}
    for n in tree.traverse('postorder'):
        if n.is_leaf():
            n.add_feature('tips', {n.name})
        else:
            n.add_feature('tips', set.union(*[getattr(_, 'tips') for _ in n.children]))
        name2n[n.name] = n

    colours = COLOURS[: len(subtrees)] if len(subtrees) <= len(COLOURS) \
        else (COLOURS * math.ceil(len(subtrees) / len(COLOURS)))[: len(subtrees)]
    colour_header = '\t'.join(colours)
    state_header = '\t'.join('cluster_{}'.format(i) for i in range(1, len(subtrees) + 1))
    shapes = '\t'.join(['1'] * len(subtrees))
    tree2colour = dict(zip(subtrees, colours))

    with open(itol_file, 'w+') as ssf:
        ssf.write(STYLE_FILE_HEADER_TEMPLATE.format(column=strategy,
                                                    colours=colour_header,
                                                    states=state_header, shapes=shapes))

        for subtree in subtrees:
            tips = {_.name for _ in subtree}
            root = name2n[subtree.name]
            colour = tree2colour[subtree]
            for node in root.traverse():
                if tips & getattr(node, 'tips'):
                    ssf.write('{}\tbranch\tnode\t{}\t4\tnormal\n'.format(node.name, colour))


IN_NWK = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data', 'tree.nwk')
m = 50
M = 75

tree = read_tree(IN_NWK)
subtrees = subtree_picker(in_nwk=IN_NWK, out_nwk=IN_NWK.replace('.nwk', '.picked.nwk'),
                          min_size=m, max_size=M)
tree = read_tree(IN_NWK)
generate_itol_style(tree, subtrees, IN_NWK.replace('.nwk', '.picked.txt'), strategy='subtree')

subtrees = independent_picker(in_nwk=IN_NWK, out_nwk=IN_NWK.replace('.nwk', '.clade.nwk'),
                              min_size=m, max_size=M)
tree = read_tree(IN_NWK)
generate_itol_style(tree, subtrees, IN_NWK.replace('.nwk', '.clade.txt'), strategy='clades')

subtrees = top_picker(in_nwk=IN_NWK, out_nwk=IN_NWK.replace('.nwk', '.top.nwk'),
                      size=M)
tree = read_tree(IN_NWK)
generate_itol_style(tree, subtrees, IN_NWK.replace('.nwk', '.top.txt'), strategy='top')


subtrees = subsampled_picker(in_nwk=IN_NWK, out_nwk=IN_NWK.replace('.nwk', '.subsampled.nwk'),
                             size=M)
tree = read_tree(IN_NWK)
generate_itol_style(tree, subtrees, IN_NWK.replace('.nwk', '.subsampled.txt'), strategy='subsampled')
