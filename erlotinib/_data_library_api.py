#
# This file is part of the erlotinib repository
# (https://github.com/DavAug/erlotinib/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

import os

import pandas as pd


class DataLibrary(object):
    r"""
    A collection of Erlotinib PKPD datasets.

    Each method corresponds to a separate dataset, which will return
    the corresponding dataset in form of a :class:`pandas.DataFrame`.
    """

    def __init__(self):
        # Get path to data library
        self._path = os.path.dirname(os.path.abspath(__file__))
        self._path += '/data_library/'

    def lung_cancer_control_group(self):
        r"""
        Returns the lung cancer control group data published in [1]_ as a
        :class:`pandas.DataFrame`.

        The dataset contains the time series data of 8 mice with
        patient-derived lung cancer implants. The tumour volume of each
        mouse was monitored over a period of 30 days and measured a couple
        times a week.

        .. [1] Eigenmann, M. J. et al., Combining Nonclinical Experiments with
               Translational PKPD Modeling to Differentiate Erlotinib and
               Gefitinib, Mol Cancer Ther. 2016; 15(12):3110-3119.
        """
        file_name = 'lxf_control_growth.csv'
        data = pd.read_csv(self._path + file_name)

        return data
