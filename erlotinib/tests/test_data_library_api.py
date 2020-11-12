#
# This file is part of the erlotinib repository
# (https://github.com/DavAug/erlotinib/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

import unittest

import pandas as pd

import erlotinib as erlo


class TestDataLibrary(unittest.TestCase):
    """
    Tests the erlotinib.DataLibrary class.
    """

    @classmethod
    def setUpClass(cls):
        cls.data_library = erlo.DataLibrary()

    def test_existence_tumour_growth_inhibition_pd_model(self):
        data = self.data_library.lung_cancer_control_group()

        self.assertIsInstance(data, pd.DataFrame)


class TestLungCancerControlGroup(unittest.TestCase):
    """
    Tests the erlotinib.DataLibrary.lung_cancer_control_group method.
    """

    @classmethod
    def setUpClass(cls):
        lib = erlo.DataLibrary()
        cls.data = lib.lung_cancer_control_group()

    def test_column_keys(self):
        keys = self.data.keys()

        n_keys = len(keys)
        self.assertEqual(n_keys, 4)

        self.assertEqual(keys[0], '#ID')
        self.assertEqual(keys[1], 'TIME in day')
        self.assertEqual(keys[2], 'TUMOUR VOLUME in cm^3')
        self.assertEqual(keys[3], 'BODY WEIGHT in g')

    def test_individuals(self):
        ids = sorted(self.data['#ID'].unique())

        n_ids = len(ids)
        self.assertEqual(n_ids, 8)

        self.assertEqual(ids[0], 40)
        self.assertEqual(ids[1], 94)
        self.assertEqual(ids[2], 95)
        self.assertEqual(ids[3], 136)
        self.assertEqual(ids[4], 140)
        self.assertEqual(ids[5], 155)
        self.assertEqual(ids[6], 169)
        self.assertEqual(ids[7], 170)


if __name__ == '__main__':
    unittest.main()
