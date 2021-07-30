#
# This file is part of the chi repository
# (https://github.com/DavAug/chi/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

import unittest

import numpy as np
import xarray as xr

from chi import plots


class TestParameterEstimatePlot(unittest.TestCase):
    """
    Tests the chi.plots.ParameterEstimatePlot class.
    """

    @classmethod
    def setUpClass(cls):
        # Test case I: Samples for no population model
        # Create a posterior samples
        n_chains = 4
        n_draws = 10
        n_ids = 1
        samples = np.random.normal(size=(n_chains, n_draws, n_ids))
        samples = xr.DataArray(
            data=samples,
            dims=['chain', 'draw', 'individual'],
            coords={
                'chain': list(range(n_chains)),
                'draw': list(range(n_draws)),
                'individual': ['ID 1']})
        cls.posterior_samples = xr.Dataset({
            param: samples for param
            in ['Parameter 1', 'Parameter 2']})

        # Test case II: Samples for population model
        # Create a posterior samples
        n_chains = 4
        n_draws = 10
        n_ids = 2
        samples = np.random.normal(size=(n_chains, n_draws, n_ids))
        samples = xr.DataArray(
            data=samples,
            dims=['chain', 'draw', 'individual'],
            coords={
                'chain': list(range(n_chains)),
                'draw': list(range(n_draws)),
                'individual': ['ID 1', 'ID 2']})
        pop_samples = xr.DataArray(
            data=samples[:, :, 0],
            dims=['chain', 'draw'],
            coords={
                'chain': list(range(n_chains)),
                'draw': list(range(n_draws))})
        cls.pop_post_samples = xr.Dataset({
            'Pooled myokit.tumour_volume': pop_samples,
            'myokit.drug_concentration': samples,
            'Pooled myokit.kappa': pop_samples,
            'Pooled myokit.lambda_0': pop_samples,
            'Pooled myokit.lambda_1': pop_samples,
            'Pooled Sigma base': pop_samples,
            'Sigma rel.': samples,
            'Mean Sigma rel.': pop_samples,
            'Std. Sigma rel.': pop_samples})

        # Test case III: Pooled model (no individuals in posterior)
        cls.pooled_samples = xr.Dataset({
            'Param 1': pop_samples,
            'Param 2': pop_samples})

        # Create figure
        cls.fig = plots.MarginalPosteriorPlot()

    def test_add_data(self):
        # Test case I: Add data for individual posteriors
        self.fig.add_data(self.posterior_samples)

        # Test case II: Add data for population posteriors
        self.fig.add_data(self.pop_post_samples)

        # Test case III: Add data for pooled posteriors
        self.fig.add_data(self.pooled_samples)

    def test_add_data_bad_input(self):
        # Bad type
        data = 'bad type'
        with self.assertRaisesRegex(TypeError, 'The data has to be'):
            self.fig.add_data(data)

        # The dimensions have the wrong names (3 dimensions)
        posterior_samples = self.pop_post_samples.copy()
        posterior_samples = posterior_samples.rename(
            {'chain': 'wrong name'})
        with self.assertRaisesRegex(ValueError, 'The data must have'):
            self.fig.add_data(posterior_samples)

        # The dimensions have the wrong names (2 dimensions)
        posterior_samples = posterior_samples.drop_dims('individual')
        with self.assertRaisesRegex(ValueError, 'The data must have'):
            self.fig.add_data(posterior_samples)

        # The dimensions are just generally wrong
        posterior_samples = posterior_samples.drop_dims('draw')
        with self.assertRaisesRegex(ValueError, 'The data must have'):
            self.fig.add_data(posterior_samples)


if __name__ == '__main__':
    unittest.main()
