#
# This file is part of the erlotinib repository
# (https://github.com/DavAug/erlotinib/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

import os


class ModelLibrary(object):
    """
    Contains references to SBML models.
    """

    def __init__(self):
        # Get path to model library
        self._path = os.path.dirname(os.path.abspath(__file__))
        self._path += '/model_library/'

    def tumour_growth_inhibition_pd_model(self):
        r"""
        Returns the absolute path to a tumour growth inhibition
        pharmacodynamic model introduced by Koch et al. in [1].

        References:
        -----------
        [1] Koch, G. et al. Modeling of tumor growth and anticancer effects of
        combination therapy. J Pharmacokinet Pharmacodyn 36, 179–197 (2009).
        """
        file_name = 'TGI_Koch_2009.xml'

        return self._path + file_name
