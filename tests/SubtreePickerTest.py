import unittest

from ete3 import TreeNode

from phylodeep.tree_utilities import extract_clusters, _annotate_relative_dates


def get_tree():
    tree = TreeNode(dist=0, name='root')

    tree.add_child(child=None, dist=1, name='1')
    right = tree.add_child(child=None, dist=1, name='r')

    right.add_child(child=None, dist=1, name='r1')
    rr = right.add_child(child=None, dist=2, name='rr')

    rrl = rr.add_child(child=None, dist=2, name='rrl')
    rrr = rr.add_child(child=None, dist=3, name='rrr')

    rrl.add_child(child=None, dist=2, name='rrl1')
    rrlr = rrl.add_child(child=None, dist=1, name='rrlr')

    rrlr.add_child(child=None, dist=1, name='rrlr1')
    rrlr.add_child(child=None, dist=1, name='rrlr2')

    rrr.add_child(child=None, dist=2, name='rrr1')
    rrrr = rrr.add_child(child=None, dist=2, name='rrrr')

    rrrr.add_child(child=None, dist=1, name='rrrr1')
    rrrr.add_child(child=None, dist=2, name='rrrr2')

    return tree


class SubtreePickerTest(unittest.TestCase):

    def test_dates(self):
        tree = get_tree()
        _annotate_relative_dates(tree, 'date')
        print(tree.get_ascii(attributes=['name', 'dist', 'date']))
        self.assertEqual((7.0, 11), getattr(next(_ for _ in tree if 'rrlr1' == _.name), 'date'))
        self.assertEqual((1.0, 1), getattr(next(_ for _ in tree if '1' == _.name), 'date'))
        self.assertEqual((2.0, 3), getattr(next(_ for _ in tree if 'r1' == _.name), 'date'))

    def test_dissection1(self):
        tree = get_tree()
        print(tree.get_ascii(attributes=['name', 'dist']))

        subtrees = list(extract_clusters(tree, min_size=3, max_size=4))
        self.assertEqual(2, len(subtrees), msg='Was expecting 2 subtrees')

        self.assertEqual('r', subtrees[0].name)
        self.assertEqual(4, len(subtrees[0]))
        self.assertListEqual(['r1', 'rrl1', 'rrlr1', 'rrlr2'], [_.name for _ in subtrees[0]])

        self.assertEqual('rrr', subtrees[1].name)
        self.assertEqual(3, len(subtrees[1]))
        self.assertListEqual(['rrr1', 'rrrr1', 'rrrr2'], [_.name for _ in subtrees[1]])

    def test_dissection2(self):
        tree = get_tree()
        print(tree.get_ascii(attributes=['name', 'dist']))

        subtrees = list(extract_clusters(tree, min_size=4, max_size=4))
        self.assertEqual(1, len(subtrees), msg='Was expecting 2 subtrees')

        self.assertEqual('rr', subtrees[0].name)
        self.assertEqual(4, len(subtrees[0]))
        self.assertListEqual(['rrl1', 'rrlr1', 'rrlr2', 'rrr1'], [_.name for _ in subtrees[0]])
