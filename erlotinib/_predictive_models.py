#
# This file is part of the erlotinib repository
# (https://github.com/DavAug/erlotinib/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

import erlotinib as erlo


class PredictiveModel(object):
    """
    Implements a model that predicts the change of observable biomarkers over
    time.

    This model takes an instance of a :class:`MechanisticModel` and an instance
    of an :class:`ErrorModel` for each mechanistic model output, and predicts
    biomarker values that may be measured in preclinical or clinical
    experiments.

    Parameters
    ----------
    mechanistic_model
        An instance of a :class:`MechanisticModel`.
    error_models
        A list of :class:`ErrorModel` instances, one for each model output of
        the mechanistic model.
    outputs
        A list of the model outputs, which maps the error models to the model
        outputs. If ``None`` the error models are assumed to be listed in the
        same order as the model outputs.
    """

    def __init__(self, mechanistic_model, error_models, outputs=None):
        super(PredictiveModel, self).__init__()

        # Check inputs
        if not isinstance(mechanistic_model, erlo.MechanisticModel):
            raise ValueError(
                'The provided mechanistic model has to be an instance of a '
                'erlotinib.MechanisticModel.')

        for error_model in error_models:
            if not isinstance(error_model, erlo.ErrorModel):
                raise ValueError(
                    'All provided error models have to be instances of a '
                    'erlo.ErrorModel.')

        # Set ouputs
        if outputs is not None:
            mechanistic_model.set_outputs(outputs)

        # Get number of outputs
        n_outputs = mechanistic_model.n_outputs()

        if len(error_models) != n_outputs:
            raise ValueError(
                'Wrong number of error models. One error model has to be '
                'provided for each mechanistic error model.')

        # Remember models
        self._mechanistic_model = mechanistic_model
        self._error_models = error_models

        # Set parameter names and number of parameters
        parameter_names = self._mechanistic_model.parameters()
        for error_model in error_models:
            parameter_names += error_models.get_parameter_names()
        self._parameter_names = parameter_names
        self._n_parameters = len(self._parameter_names)

    def get_parameter_names(self):
        """
        Returns the parameter names of the predictive model.
        """
        return self._parameter_names

    def n_parameters(self):
        """
        Returns the number of parameters of the predictive model.
        """
        return self._n_parameters
