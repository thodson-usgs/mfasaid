import abc
import copy

from numbers import Number

from numpy import log, log10, power
import pandas as pd
import statsmodels.formula.api as smf

import data


class ModelException(Exception):
    """Exception base class for the model.py module."""
    pass


class InvalidModelVariableNameError(ModelException):
    """Raised when an invalid model variable name is encountered"""
    pass


class InvalidVariableTransformError(ModelException):
    """Raise when an invalid variable transform is encountered."""
    pass


class RatingModel(abc.ABC):
    """"""

    _transform_functions = {None: 'x',
                           'log': 'log(x)',
                           'log10': 'log10(x)',
                           'pow2': 'power(x, 2)'}

    @abc.abstractmethod
    def __init__(self, response_data, explanatory_data, response_variable=None):
        """

        :param response_data:
        :type response_data: data.ConstituentData
        :param explanatory_data:
        :type explanatory_data: data.SurrogateData
        :param response_variable:
        :param explanatory_variable:
        """

        self._response_data = response_data
        self._explanatory_data = explanatory_data

        if (response_variable is None) or (response_variable in response_data.get_variable_names()):
            self._response_variable = response_variable
        else:
            raise InvalidModelVariableNameError("{} is not a valid response variable name.".format(response_variable))

        self._excluded_observations = pd.tseries.index.DatetimeIndex([], name='DateTime')
        self._model_dataset = pd.DataFrame()

        self._explanatory_variable_avg_window = {}
        for variable in explanatory_data.get_variable_names():
            self._explanatory_variable_avg_window[variable] = 0

        self._response_variable_transform = {}
        for variable in response_data.get_variable_names():
            self._response_variable_transform[variable] = None

    def _check_response_variable(self, response_variable):
        """

        :param response_variable:
        :type response_variable: str
        :return:
        """

        self._check_variable_name(response_variable, self.get_variable_names()[0])

    @classmethod
    def _check_transform(cls, transform):
        """

        :param transform:
        :return:
        """
        if transform not in cls._transform_functions.keys():
            raise InvalidVariableTransformError("{} is an unrecognized transformation.".format(transform))

    @staticmethod
    def _check_variable_name(variable_name, list_of_variable_names):
        """

        :param variable_name:
        :type variable_name: str
        :param list_of_variable_names:
        :type list_of_variable_names: abc.Iterable
        :return:
        """

        if variable_name not in list_of_variable_names:
            raise InvalidModelVariableNameError("{} is not a valid variable name.".format(variable_name), variable_name)

    @abc.abstractmethod
    def _create_model(self):
        pass

    @abc.abstractmethod
    def _create_model_dataset(self):
        pass

    @classmethod
    def _get_variable_transform(cls, variable, transform):
        """

        :param variable:
        :param transform:
        :return:
        """

        return cls._transform_functions[transform].replace('x', variable)

    def exclude_observation(self, observation_time):
        """Include the observation given the time of the response variable observation."""

        self._excluded_observations = self._excluded_observations.append(observation_time)
        self._excluded_observations = self._excluded_observations.sort_values()
        self._excluded_observations = self._excluded_observations.drop_duplicates()

        self._create_model()

    def get_explanatory_avg_window(self):
        """Returns a dictionary containing the averaging time, in minutes, for the explanatory variables."""

        return copy.deepcopy(self._explanatory_variable_avg_window)

    def get_excluded_observations(self):
        """Returns a time series of observations that have been excluded from the model."""

        return copy.deepcopy(self._excluded_observations)

    def get_model_dataset(self):
        """Return a DataFrame containing the observations used in the current model."""

        model_dataset = pd.DataFrame(self._model_dataset.copy(deep=True))

        if model_dataset.shape != (0, 0):
            model_dataset.ix[:, 'Missing'] = model_dataset.isnull().any(axis=1)
            model_dataset.ix[:, 'Excluded'] = model_dataset.index.isin(self._excluded_observations)

        return model_dataset

    @abc.abstractmethod
    def get_model_formula(self):
        """Return the formula use in the creation of the model."""
        pass

    @abc.abstractmethod
    def get_model_summary(self):
        """Return the summary for the current model."""
        pass

    def get_response_variable(self):
        """Return the name of the current response variable"""

        return self._response_variable

    def get_variable_names(self):
        """Return a tuple containing a tuple of possible response variables and a tuple of possible explanatory
        variables.
        """
        response_variables = tuple(self._response_data.get_variable_names())
        explanatory_variables = set(self._explanatory_data.get_variable_names())

        explanatory_variables = tuple(explanatory_variables.difference(response_variables))

        return response_variables, explanatory_variables

    def include_all_observations(self):
        """Include all observations that have previously been excluded."""

        self.include_observation(self._excluded_observations)
        self._create_model()

    def include_observation(self, observation_time):
        """Include the observation given the time of the response variable observation."""

        restored_index = self._excluded_observations.isin(observation_time)

        self._excluded_observations = self._excluded_observations[~restored_index]

        self._create_model()

    def set_explanatory_avg_window(self, variable, avg_window):
        """Set the averaging window, in minutes, for the given variable"""

        self._check_variable_name(variable, self._explanatory_data.get_variable_names())

        self._explanatory_variable_avg_window[variable] = avg_window

        self.update_model()

    def set_response_variable(self, response_variable):
        """Set the response variable."""

        self._check_response_variable(response_variable)

        self._response_variable = response_variable

        self.update_model()

    def transform_response_variable(self, transform):
        """Transform the response variable."""

        self._check_transform(transform)
        self._response_variable_transform[self._response_variable] = transform

        self._create_model()

    @abc.abstractmethod
    def predict_response_variable(self, explanatory_variable):
        """Predict the value of the response variable given values for the explanatory variable."""
        pass

    def update_model(self):
        """

        :return:
        """

        self._create_model_dataset()
        self._create_model()


class SingleExplanatoryVariableModel(RatingModel, abc.ABC):
    """"""

    @abc.abstractmethod
    def __init__(self, response_data, explanatory_data, response_variable=None, explanatory_variable=None):
        """

        :param response_data:
        :param explanatory_data:
        :param response_variable:
        :param explanatory_variable:
        """

        if (explanatory_variable is None) or (explanatory_variable in explanatory_data.get_variable_names()):
            self._explanatory_variable = explanatory_variable
        else:
            raise InvalidModelVariableNameError(
                "{} is not a valid explanatory variable name.".format(explanatory_variable))

        super().__init__(response_data, explanatory_data, response_variable)

    def _check_explanatory_variable(self, explanatory_variable):
        """

        :param explanatory_variable:
        :return:
        """

        self._check_variable_name(explanatory_variable, self.get_variable_names()[1])

    def _create_model_dataset(self):
        """

        :return:
        """

        try:

            mdl_dataset = self._response_data.get_variable(self._response_variable)

        except ValueError as err:
            if err.args[1] is None:
                mdl_dataset = pd.DataFrame()
            else:
                raise err

        try:

            for index, row in mdl_dataset.iterrows():
                avg_window = self._explanatory_variable_avg_window[self._explanatory_variable]
                mdl_dataset.ix[index, self._explanatory_variable] = \
                    self._explanatory_data.get_avg_variable_observation(self._explanatory_variable, index, avg_window)

        except TypeError as err:
            if err.args[0] == "'NoneType' object is not iterable":
                mdl_dataset = pd.DataFrame()
            else:
                raise err
        except KeyError as err:
            if err.args[0] is None:
                mdl_dataset = pd.DataFrame()
            else:
                raise err

        self._model_dataset = mdl_dataset

    def get_explanatory_variable(self):
        """

        :return:
        """

        return self._explanatory_variable

    def set_explanatory_variable(self, variable):
        """

        :param variable:
        :return:
        """

        self._check_explanatory_variable(variable)
        self._explanatory_variable = variable
        self._create_model_dataset()
        self._create_model()


class MultipleExplanatoryVariableModel(RatingModel, abc.ABC):
    """"""

    @abc.abstractmethod
    def __init__(self, response_data, explanatory_data, response_variable=None, explanatory_variables=None):
        """

        :param response_data:
        :param explanatory_data:
        :param response_variable:
        :param explanatory_variables:
        """

        if explanatory_variables is not None:
            self.set_explanatory_variables(explanatory_variables)
        else:
            self._explanatory_variables = explanatory_variables

        super().__init__(response_data, explanatory_data, response_variable)

    def _check_explanatory_variables(self, explanatory_variables):
        """

        :param explanatory_variables:
        :type explanatory_variables: abc.Iterable
        :return:
        """

        for var_name in explanatory_variables:
            self._check_variable_name(var_name, self.get_variable_names()[1])

    def _create_model_dataset(self):
        """

        :return:
        """

        try:

            mdl_dataset = self._response_data.get_variable(self._response_variable)

        except ValueError as err:
            if err.args[1] is None:
                mdl_dataset = pd.DataFrame()
            else:
                raise err

        try:

            for variable in self._explanatory_variables:
                for index, row in mdl_dataset.iterrows():
                    avg_window = self._explanatory_variable_avg_window[variable]
                    mdl_dataset.ix[index, variable] = \
                        self._explanatory_data.get_avg_variable_observation(variable, index, avg_window)

        except TypeError as err:
            if err.args[0] == "'NoneType' object is not iterable":
                mdl_dataset = pd.DataFrame()
            else:
                raise err

        self._model_dataset = mdl_dataset

    def get_explanatory_variables(self):
        """

        :return:
        """

        return tuple(self._explanatory_variables)

    def set_explanatory_variables(self, variables):
        """

        :param variables:
        :return:
        """

        self._check_explanatory_variables(variables)

        self._explanatory_variables = tuple(variables)


class SimpleLinearRatingModel(SingleExplanatoryVariableModel):

    def __init__(self, response_data, explanatory_data, response_variable=None, explanatory_variable=None):
        """

        :param response_data:
        :param explanatory_data:
        :param response_variable:
        :param explanatory_variable:
        """

        self._model = None
        self._explanatory_variable_transform = {}
        for variable in explanatory_data.get_variable_names():
            self._explanatory_variable_transform[variable] = None

        super().__init__(response_data, explanatory_data, response_variable, explanatory_variable)

    def _create_model(self):
        """

        :return:
        """

        if self._response_variable and self._explanatory_variable:

            model_formula = self.get_model_formula()

            removed_observation_index = self._model_dataset.index.isin(self._excluded_observations)

            try:
                model = smf.ols(model_formula,
                                data=self._model_dataset,
                                subset=~removed_observation_index,
                                missing='drop')
            except ValueError as err:
                if err.args[0] == "zero-size array to reduction operation maximum which has no identity":
                    model = None
                else:
                    raise err

        else:
            model = None

        self._model = model

    def get_model_formula(self):
        """

        :return:
        """

        if self._response_variable and self._explanatory_variable:

            response_var_transform = self._response_variable_transform[self._response_variable]
            model_response_var = self._get_variable_transform(self._response_variable, response_var_transform)

            explanatory_var_transform = self._explanatory_variable_transform[self._explanatory_variable]
            model_explanatory_var = self._get_variable_transform(self._explanatory_variable, explanatory_var_transform)

            model_formula = model_response_var + ' ~ ' + model_explanatory_var

        else:

            model_formula = None

        return model_formula

    def get_model_summary(self):
        """

        :return:
        """

        try:
            summary = self._model.fit().summary()
        except AttributeError:
            summary = None

        return summary

    def predict_response_variable(self, explanatory_variable):
        """

        :param explanatory_variable:
        :return:
        """
        # TODO: Implement predict_response_variable() for SimpleLinearRatingModel
        pass

    def transform_explanatory_variable(self, transform):
        """

        :param transform:
        :return:
        """

        self._check_transform(transform)
        self._explanatory_variable_transform[self._explanatory_variable] = transform

        self._create_model()
