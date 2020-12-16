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

    def test_existence_lung_cancer_control_group(self):
        data = self.data_library.lung_cancer_control_group()

        self.assertIsInstance(data, pd.DataFrame)

    def test_existence_lung_cancer_high_erlotinib_dose_group(self):
        data = self.data_library.lung_cancer_high_erlotinib_dose_group()

        self.assertIsInstance(data, pd.DataFrame)

    def test_existence_lung_cancer_low_erlotinib_dose_group(self):
        data = self.data_library.lung_cancer_low_erlotinib_dose_group()

        self.assertIsInstance(data, pd.DataFrame)

    def test_existence_lung_cancer_medium_erlotinib_dose_group(self):
        data = self.data_library.lung_cancer_medium_erlotinib_dose_group()

        self.assertIsInstance(data, pd.DataFrame)


class TestLungCancerControlGroup(unittest.TestCase):
    """
    Tests the erlotinib.DataLibrary.lung_cancer_control_group method.
    """

    @classmethod
    def setUpClass(cls):
        lib = erlo.DataLibrary()
        cls.data = lib.lung_cancer_control_group()
        cls.standardised_data = lib.lung_cancer_control_group(True)

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

    def test_standardised_column_keys(self):
        keys = self.standardised_data.keys()

        n_keys = len(keys)
        self.assertEqual(n_keys, 4)

        self.assertEqual(keys[0], 'ID')
        self.assertEqual(keys[1], 'Time')
        self.assertEqual(keys[2], 'Biomarker')
        self.assertEqual(keys[3], 'BODY WEIGHT in g')


class TestLungCancerHighErlotinibDoseGroup(unittest.TestCase):
    """
    Tests the erlotinib.DataLibrary.lung_cancer_high_erlotinib_dose_group
    method.
    """

    @classmethod
    def setUpClass(cls):
        lib = erlo.DataLibrary()
        cls.data = lib.lung_cancer_high_erlotinib_dose_group()

    def test_column_keys(self):
        keys = self.data.keys()

        n_keys = len(keys)
        self.assertEqual(n_keys, 6)

        self.assertEqual(keys[0], '#ID')
        self.assertEqual(keys[1], 'TIME in day')
        self.assertEqual(keys[2], 'DOSE in mg')
        self.assertEqual(keys[3], 'PLASMA CONCENTRATION in mg/L')
        self.assertEqual(keys[4], 'TUMOUR VOLUME in cm^3')
        self.assertEqual(keys[5], 'BODY WEIGHT in g')

    def test_individuals(self):
        ids = sorted(self.data['#ID'].unique())

        n_ids = len(ids)
        self.assertEqual(n_ids, 6)

        self.assertEqual(ids[0], 6)
        self.assertEqual(ids[1], 11)
        self.assertEqual(ids[2], 28)
        self.assertEqual(ids[3], 67)
        self.assertEqual(ids[4], 128)
        self.assertEqual(ids[5], 134)


class TestLungCancerLowErlotinibDoseGroup(unittest.TestCase):
    """
    Tests the erlotinib.DataLibrary.lung_cancer_low_erlotinib_dose_group
    method.
    """

    @classmethod
    def setUpClass(cls):
        lib = erlo.DataLibrary()
        cls.data = lib.lung_cancer_low_erlotinib_dose_group()

    def test_column_keys(self):
        keys = self.data.keys()

        n_keys = len(keys)
        self.assertEqual(n_keys, 6)

        self.assertEqual(keys[0], '#ID')
        self.assertEqual(keys[1], 'TIME in day')
        self.assertEqual(keys[2], 'DOSE in mg')
        self.assertEqual(keys[3], 'PLASMA CONCENTRATION in mg/L')
        self.assertEqual(keys[4], 'TUMOUR VOLUME in cm^3')
        self.assertEqual(keys[5], 'BODY WEIGHT in g')

    def test_individuals(self):
        ids = sorted(self.data['#ID'].unique())

        n_ids = len(ids)
        self.assertEqual(n_ids, 8)

        self.assertEqual(ids[0], 31)
        self.assertEqual(ids[1], 38)
        self.assertEqual(ids[2], 98)
        self.assertEqual(ids[3], 119)
        self.assertEqual(ids[4], 131)
        self.assertEqual(ids[5], 139)
        self.assertEqual(ids[6], 161)
        self.assertEqual(ids[7], 162)


class TestLungCancerMediumErlotinibDoseGroup(unittest.TestCase):
    """
    Tests the erlotinib.DataLibrary.lung_cancer_medium_erlotinib_dose_group
    method.
    """

    @classmethod
    def setUpClass(cls):
        lib = erlo.DataLibrary()
        cls.data = lib.lung_cancer_medium_erlotinib_dose_group()

    def test_column_keys(self):
        keys = self.data.keys()

        n_keys = len(keys)
        self.assertEqual(n_keys, 6)

        self.assertEqual(keys[0], '#ID')
        self.assertEqual(keys[1], 'TIME in day')
        self.assertEqual(keys[2], 'DOSE in mg')
        self.assertEqual(keys[3], 'PLASMA CONCENTRATION in mg/L')
        self.assertEqual(keys[4], 'TUMOUR VOLUME in cm^3')
        self.assertEqual(keys[5], 'BODY WEIGHT in g')

    def test_individuals(self):
        ids = sorted(self.data['#ID'].unique())

        n_ids = len(ids)
        self.assertEqual(n_ids, 8)

        self.assertEqual(ids[0], 34)
        self.assertEqual(ids[1], 52)
        self.assertEqual(ids[2], 91)
        self.assertEqual(ids[3], 108)
        self.assertEqual(ids[4], 122)
        self.assertEqual(ids[5], 129)
        self.assertEqual(ids[6], 163)
        self.assertEqual(ids[7], 167)


if __name__ == '__main__':
    unittest.main()
