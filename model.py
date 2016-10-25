import copy

from numbers import Number

import pandas as pd

import data


class ModelException(Exception):
    """Exception base class for the model.py module."""
    pass


class InvalidModelVariableNameError(ModelException):
    """Raised when an invalid """


class SurrogateModel:
    """"""

    def __init__(self, constituent_data, surrogate_data, response_variable=None, explanatory_variables=None):
        """

        :param constituent_data:
        :type constituent_data: data.ConstituentData
        :param surrogate_data:
        :type surrogate_data: data.SurrogateData
        """

        self._constituent_data = constituent_data
        self._surrogate_data = surrogate_data

        self._response_variable = response_variable
        self._explanatory_variables = explanatory_variables

        self._removed_observations = pd.tseries.index.DatetimeIndex([], name='DateTime')
        self._mdl_dataset = pd.DataFrame()

        self._regression_type = 'ols'
        self._linear_regression_model = None

        self._surrogate_variable_avg_window = {}
        for variable in surrogate_data.get_variable_names():
            self._surrogate_variable_avg_window[variable] = 0

        if self._response_variable and self._explanatory_variables:
            self._create_mdl_dataset()

    def _check_constituent_variable(self, constituent_variable):
        """

        :param constituent_variable:
        :type constituent_variable: str
        :return:
        """

        self._check_variable_name(constituent_variable, self.get_constituent_variable_names())

    def _check_surrogate_variables(self, surrogate_variables):
        """

        :param surrogate_variables:
        :type surrogate_variables: list
        :return:
        """

        for var_name in surrogate_variables:
            self._check_variable_name(var_name, self.get_surrogate_variable_names())

    @staticmethod
    def _check_variable_name(variable_name, list_of_variable_names):
        """

        :param variable_name:
        :type variable_name: str
        :param list_of_variable_names:
        :type list_of_variable_names: list
        :return:
        """

        # if not isinstance(variable_name, str):
        #     raise TypeError("Variable name must be a string.", variable_name)

        if variable_name not in list_of_variable_names:
            raise InvalidModelVariableNameError("Invalid variable name: {}".format(variable_name), variable_name)

    def _create_mdl_dataset(self):
        """

        :return:
        """

        assert (self._response_variable is not None) and (self._explanatory_variables is not None)

        mdl_dataset = self._constituent_data.get_variable(self._response_variable)

        for variable in self._explanatory_variables:

            for index, row in mdl_dataset.iterrows():

                mdl_dataset.ix[index, variable] = \
                    self._surrogate_data.get_avg_variable_observation(variable, index,
                                                                      self._surrogate_variable_avg_window[variable])

        self._mdl_dataset = mdl_dataset

    def _create_regression(self):
        """

        :return:
        """

        self._linear_regression_model = None

    def get_constituent_variable_names(self):
        """

        :return:
        """

        return self._constituent_data.get_variable_names()

    def get_surrogate_variable_names(self):
        """Return a list of the surrogate variables

        :return:
        """

        return self._surrogate_data.get_variable_names()

    def get_surrogate_var_avg_window(self, surrogate_variable):
        """

        :param surrogate_variable:
        :return:
        """

        self._check_surrogate_variables([surrogate_variable])

        return self._surrogate_variable_avg_window[surrogate_variable]

    def get_mdl_observations(self):
        """

        :return:
        """

        return self._mdl_dataset.copy(deep=True)

    def remove_mdl_observations(self, observation_index):
        """

        :param observation_index:
        :return:
        """

        self._removed_observations = self._removed_observations.append(observation_index)
        self._removed_observations = self._removed_observations.sort_values()
        self._removed_observations = self._removed_observations.drop_duplicates()

    def restore_mdl_observations(self, observation_index):
        """

        :param observation_index:
        :return:
        """

        restored_index = self._removed_observations.isin(observation_index)

        self._removed_observations = self._removed_observations[~restored_index]

    def restore_all_mdl_observations(self):
        """

        :return:
        """

        self.restore_mdl_observations(self._removed_observations)

    def set_response_variable(self, response_variable):
        """Set the response variable used in the model.

        The response, or dependent, variable is the constituent variable.

        :param response_variable: Name of constituent variable
        :type response_variable: str
        :return: None
        """

        self._check_constituent_variable(response_variable)

        self._response_variable = response_variable

        if self._response_variable and self._explanatory_variables:
            self._create_mdl_dataset()
            self._create_regression()

    def set_explanatory_variables(self, explanatory_variables):
        """Set the surrogate variable(s) used in the model.py.

        The surrogate variable(s) is(are) the explanatory, or independent, variable(s).

        :param explanatory_variables:
        :return:
        """

        self._check_surrogate_variables(explanatory_variables)

        self._explanatory_variables = copy.deepcopy(explanatory_variables)

        if self._response_variable and self._explanatory_variables:
            self._create_mdl_dataset()
            self._create_regression()

    def set_surrogate_var_avg_window(self, surrogate_variable, avg_window):
        """

        :param surrogate_variable:
        :param avg_window:
        :return:
        """

        self._check_surrogate_variables([surrogate_variable])

        if not isinstance(avg_window, Number):
            raise TypeError('Expected numeric value.', avg_window)

        self._surrogate_variable_avg_window[surrogate_variable] = avg_window

        if self._response_variable and self._explanatory_variables:
            self._create_mdl_dataset()
            self._create_regression()
