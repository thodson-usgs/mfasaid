import abc
import copy

import numpy as np

# needed for variable transformations
from numpy import log, log10, power

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.sandbox.regression.predstd import wls_prediction_std

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


class ModelVariableRangeError(ModelException):
    """Raise when an invalid variable range is encountered."""
    pass


class RatingModel(abc.ABC):
    """Base class for rating models."""

    _transform_variable_names = {None: 'x',
                                 'log': 'log(x)',
                                 'log10': 'log10(x)',
                                 'pow2': 'power(x, 2)'
                                 }
    _transform_functions = {None: lambda x: x,
                            'log': log,
                            'log10': log10,
                            'pow2': lambda x: power(x, 2)
                            }
    _inverse_transform_functions = {None: lambda x: x,
                                    'log': np.exp,
                                    'log10': lambda x: power(10, x),
                                    'pow2': lambda x: power(x, 1/2)
                                    }

    def __init__(self, data_manager, response_variable=None):
        """Initialize a RatingModel object.

        :param data_manager:
        :type data_manager: data.DataManager
        :param response_variable:
        """

        if not isinstance(data_manager, data.DataManager):
            raise TypeError("data_manager must be type data.DataManager")

        self._data_manager = data_manager

        variable_names = data_manager.get_variable_names()

        # set the response variable, make sure it's valid
        if response_variable is None:
            self._response_variable = variable_names[0]
        else:
            if response_variable in variable_names:
                self._response_variable = response_variable
            else:
                raise InvalidModelVariableNameError(
                    "{} is not a valid response variable name.".format(response_variable))

        # initialize the explanatory variables attribute
        self._explanatory_variables = tuple(variable_names[1:])

        # noinspection PyUnresolvedReferences
        self._excluded_observations = pd.tseries.index.DatetimeIndex([], name='DateTime')
        self._model_dataset = pd.DataFrame()

        # initialize the variable transform dictionary
        self._variable_transform = {}
        for variable in variable_names:
            self._variable_transform[variable] = None

        # initialize the model attribute
        self._model = None

    @classmethod
    def _check_transform(cls, transform):
        """

        :param transform:
        :return:
        """
        if transform not in cls._transform_variable_names.keys():
            raise InvalidVariableTransformError("{} is an unrecognized transformation.".format(transform))

    def _check_variable_names(self, variable_names):
        """

        :param variable_names:
        :type variable_names: abc.Iterable
        :return:
        """
        for variable in variable_names:
            if variable not in self._data_manager.get_variable_names():
                raise InvalidModelVariableNameError("{} is not a valid variable name.".format(variable),
                                                    variable)

    @abc.abstractmethod
    def _create_model(self):
        pass

    def _create_model_dataset(self):
        """

        :return:
        """

        mdl_dataset = self._data_manager.get_variable(self._response_variable)

        for variable in self._explanatory_variables:
            mdl_dataset[variable] = self._data_manager.get_variable(variable)

        self._model_dataset = mdl_dataset

    @classmethod
    def _get_variable_transform(cls, variable, transform):
        """

        :param variable:
        :param transform:
        :return:
        """

        return cls._transform_variable_names[transform].replace('x', variable)

    def exclude_observation(self, observation_time):
        """Exclude the observation given by observation_time from the model

        :param observation_time:
        :return:
        """

        self._excluded_observations = self._excluded_observations.append(observation_time)
        self._excluded_observations = self._excluded_observations.sort_values()
        self._excluded_observations = self._excluded_observations.drop_duplicates()

        self._create_model()

    def get_excluded_observations(self):
        """Returns a time series of observations that have been excluded from the model.

        :return:
        """

        return copy.deepcopy(self._excluded_observations)

    def get_model_dataset(self):
        """Return a DataFrame containing the observations used in the current model."""

        model_dataset = pd.DataFrame(self._model_dataset.copy(deep=True))

        if model_dataset.shape != (0, 0):
            model_dataset.ix[:, 'Missing'] = model_dataset.isnull().any(axis=1)
            model_dataset.ix[:, 'Excluded'] = model_dataset.index.isin(self._excluded_observations)

        return model_dataset

    @abc.abstractmethod
    def get_model_summary(self):
        pass

    def get_response_variable(self):
        """Return the name of the current response variable"""

        return self._response_variable

    def get_variable_names(self):
        """Return a tuple containing the variable names within the model.
        """

        return tuple(self._data_manager.get_variable_names())

    def include_all_observations(self):
        """Include all observations that have previously been excluded."""

        self.include_observation(self._excluded_observations)
        self._create_model()

    def include_observation(self, observation_time):
        """Include the observation given the time of the response variable observation."""

        restored_index = self._excluded_observations.isin(observation_time)

        self._excluded_observations = self._excluded_observations[~restored_index]

        self._create_model()

    def set_response_variable(self, response_variable):
        """Set the response variable."""

        self._check_variable_names([response_variable])

        self._response_variable = response_variable

        self.update_model()

    def transform_response_variable(self, transform):
        """Transform the response variable."""

        self._check_transform(transform)
        self._variable_transform[self._response_variable] = transform

        self._create_model()

    @abc.abstractmethod
    def predict_response_variable(self, explanatory_variable):
        """Predict the value of the response variable given values for the explanatory variable."""
        pass

    def update_model(self):
        """Update the regression model.

        :return:
        """

        self._create_model_dataset()
        self._create_model()


class OLSModel(RatingModel, abc.ABC):
    """Ordinary least squares (OLS) regression based rating model abstract class."""

    def _create_model(self):
        """

        :return:
        """

        if self._response_variable and self._explanatory_variables[0]:

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

    @abc.abstractmethod
    def _get_exogenous_matrix(self, exogenous_df):
        pass

    @abc.abstractmethod
    def get_model_formula(self):
        pass

    def get_model_summary(self):
        """

        :return:
        """

        try:
            summary = self._model.fit().summary()
        except AttributeError:
            summary = None

        return summary

    def predict_response_variable(self, explanatory_data=None, bias_correction=False, prediction_interval=False):
        """

        :param explanatory_data:
        :param bias_correction:
        :param prediction_interval:
        :return:
        """

        # TODO: Implement OLSModel.predict_response_variable()

        if self._model:

            # get the model results
            res = self._model.fit()

            # get the explanatory data DataFrame
            if explanatory_data:
                explanatory_df = explanatory_data.get_data()
            else:
                explanatory_df = self._data_manager.get_data()

            exog = self._get_exogenous_matrix(explanatory_df)

            # predicted response variable
            mean_response = self._model.predict(res.params, exog=exog)
            mean_response = np.expand_dims(mean_response, axis=1)

            if prediction_interval:

                # confidence level for two - sided hypothesis
                confidence_level = 0.1  # 90% prediction interval
                confidence_level_text = '{:.1f}'.format(100*(1-confidence_level))

                _, interval_l, interval_u = wls_prediction_std(res, exog=exog, alpha=confidence_level)

                interval_l = np.expand_dims(interval_l, axis=1)
                interval_u = np.expand_dims(interval_u, axis=1)

                response_data = np.dstack((interval_l, mean_response, interval_u))

                columns = [self._response_variable + '_L' + confidence_level_text,
                           self._response_variable,
                           self._response_variable + '_U' + confidence_level_text
                           ]

            else:

                response_data = np.expand_dims(mean_response, axis=2)

                columns = [self._response_variable]

            if bias_correction:

                residuals = res.resid.as_matrix()
                residuals = np.expand_dims(residuals, axis=0)
                residuals = np.expand_dims(residuals, axis=2)
                residuals = np.tile(residuals, (response_data.shape[0], 1, response_data.shape[2]))

                response_data = np.tile(response_data, (1, residuals.shape[1], 1))

                prediction_results = np.mean(response_data + residuals, axis=1)

            else:

                prediction_results = np.squeeze(response_data, axis=1)

            response_variable_transform = self._variable_transform[self._response_variable]

            predicted_data = self._inverse_transform_functions[response_variable_transform](prediction_results)

            predicted = pd.DataFrame(data=predicted_data, index=explanatory_df.index, columns=columns)

        else:

            predicted = pd.DataFrame(columns=[self._response_variable])

        return predicted


class SimpleLinearRatingModel(OLSModel):
    """Class for OLS simple linear regression (SLR) ratings."""

    def __init__(self, data_manager, response_variable=None, explanatory_variable=None):
        """

        :param data_manager:
        :param response_variable:
        :param explanatory_variable:
        """

        super().__init__(data_manager, response_variable)

        if explanatory_variable:
            self.set_explanatory_variable(explanatory_variable)
        else:
            self.set_explanatory_variable(data_manager.get_variable_names()[1])

        self.update_model()

    def _get_exogenous_matrix(self, exogenous_df):
        """

        :param exogenous_df:
        :return:
        """
        # TODO: Implement SimpleLinearRatingModel._get_exogenous_matrix()

        explanatory_variable = self.get_explanatory_variable()

        assert(explanatory_variable in exogenous_df.keys())

        exog = pd.DataFrame()

        explanatory_transform = self._variable_transform[explanatory_variable]
        transformed_variable_name = self._get_variable_transform(explanatory_variable, explanatory_transform)
        transform_function = self._transform_functions[explanatory_transform]
        exog[transformed_variable_name] = transform_function(exogenous_df[explanatory_variable])
        exog = sm.add_constant(exog)

        return exog

    def get_explanatory_variable(self):
        """Returns the name of the explanatory variable used in the SLR.

        :return: Name of explanatory variable
        """

        return self._explanatory_variables[0]

    def get_model_formula(self):
        """

        :return:
        """

        if self._response_variable and self._explanatory_variables[0]:

            explanatory_variable = self.get_explanatory_variable()

            response_var_transform = self._variable_transform[self._response_variable]
            model_response_var = self._get_variable_transform(self._response_variable, response_var_transform)

            explanatory_var_transform = self._variable_transform[explanatory_variable]
            model_explanatory_var = self._get_variable_transform(explanatory_variable, explanatory_var_transform)

            model_formula = model_response_var + ' ~ ' + model_explanatory_var

        else:

            model_formula = None

        return model_formula

    def set_explanatory_variable(self, variable):
        """

        :param variable:
        :return:
        """

        self._check_variable_names([variable])
        self._explanatory_variables = (variable,)
        self.update_model()

    def transform_explanatory_variable(self, transform):
        """

        :param transform:
        :return:
        """

        self._check_transform(transform)
        self._variable_transform[self._explanatory_variables[0]] = transform

        self._create_model()


class MultipleLinearRatingModel(OLSModel):
    """"""

    def __init__(self, data_manager, response_variable=None, explanatory_variables=None):
        """

        :param data_manager:
        :param response_variable:
        :param explanatory_variables:
        :return:
        """

        super().__init__(data_manager, response_variable)

        if explanatory_variables:
            self.set_explanatory_variables(explanatory_variables)

        self.update_model()

    def _get_exogenous_matrix(self, exogenous_df):
        """

        :param exogenous_df:
        :return:
        """
        # TODO: Implement SimpleLinearRatingModel._get_exogenous_matrix()
        pass

    def get_explanatory_variables(self):
        """

        :return:
        """

        return tuple(self._explanatory_variables)

    def get_model_formula(self):
        """

        :return:
        """

        if self._response_variable and self._explanatory_variables[0]:

            response_var_transform = self._variable_transform[self._response_variable]
            model_response_var = self._get_variable_transform(self._response_variable, response_var_transform)

            explanatory_vars_transform = []
            for variable in self._explanatory_variables:
                explan_transform = self._variable_transform[variable]
                explanatory_vars_transform.append(self._get_variable_transform(variable, explan_transform))

            model_formula = model_response_var + ' ~ ' + ' + '.join(explanatory_vars_transform)

        else:

            model_formula = None

        return model_formula

    def set_explanatory_variables(self, variables):
        """

        :param variables:
        :return:
        """

        self._check_variable_names(variables)

        self._explanatory_variables = tuple(variables)

        self.update_model()

    def transform_explanatory_variable(self, explanatory_variable, transform):
        """

        :param explanatory_variable:
        :param transform:
        :return:
        """

        self._check_transform(transform)
        self._check_variable_names([explanatory_variable])
        self._variable_transform[explanatory_variable] = transform

        self._create_model()


class ComplexRatingModel(OLSModel):
    """"""

    def __init__(self, data_manager, response_variable=None, explanatory_variable=None):
        """

        :param data_manager:
        :param response_variable:
        :param explanatory_variable:
        """

        super().__init__(data_manager, response_variable)

        self._explanatory_variable_transform = [None]

        if explanatory_variable:
            self.set_explanatory_variable(explanatory_variable)

        self.update_model()

    def _get_exogenous_matrix(self, exogenous_df):
        """

        :return:
        """

        explanatory_variable = self.get_explanatory_variable()

        assert(explanatory_variable in exogenous_df.keys())

        exog = pd.DataFrame()

        for transform in self._explanatory_variable_transform:

            transform_variable_name = self._get_variable_transform(explanatory_variable, transform)
            transform_function = self._transform_functions[transform]
            exog[transform_variable_name] = transform_function(exogenous_df[explanatory_variable])

        exog = sm.add_constant(exog)

        return exog

    def add_explanatory_var_transform(self, transform):
        """

        :param transform:
        :return:
        """

        self._check_transform(transform)

        self._explanatory_variable_transform.append(transform)

        self._create_model()

    def get_explanatory_variable(self):
        """

        :return:
        """

        return self._explanatory_variables[0]

    def get_model_formula(self):
        """

        :return:
        """

        if self._response_variable and self._explanatory_variables[0]:

            response_var_transform = self._variable_transform[self._response_variable]
            model_response_var = self._get_variable_transform(self._response_variable, response_var_transform)

            explanatory_variables = []
            for transform in self._explanatory_variable_transform:

                explanatory_variables.append(self._get_variable_transform(self._explanatory_variables[0], transform))

            model_formula = model_response_var + ' ~ ' + ' + '.join(explanatory_variables)

        else:

            model_formula = None

        return model_formula

    def remove_explanatory_var_transform(self, transform):
        """

        :param transform:
        :return:
        """

        if transform in self._explanatory_variable_transform:
            self._explanatory_variable_transform.remove(transform)
            if len(self._explanatory_variable_transform) < 1:
                self._explanatory_variable_transform.append(None)

        self._create_model()

    def reset_explanatory_var_transform(self):
        """

        :return:
        """

        self._explanatory_variable_transform = [None]

        self._create_model()

    def set_explanatory_variable(self, variable):
        """

        :param variable:
        :return:
        """

        self._check_variable_names([variable])
        self._explanatory_variables = (variable,)
        self.update_model()


class CompoundRatingModel(RatingModel):
    """"""

    def __init__(self, data_manager, response_variable=None, explanatory_variable=None):

        super().__init__(data_manager, response_variable)

        if explanatory_variable:
            self.set_explanatory_variable(explanatory_variable)
        else:
            self.set_explanatory_variable(data_manager.get_variable_names()[1])

        self._explanatory_variable_transform = [[None]]

        self._breakpoints = np.array([-np.inf, np.inf])

        self._model = []

        self.update_model()

    def _check_segment_number(self, segment_number):
        """

        :param segment:
        :return:
        """

        if not 0 < segment_number and segment_number <= len(self._model):
            raise ValueError("Invalid segment number.")

    def _create_model(self):
        """

        :return:
        """

        self._model = []

        removed_observation_index = self._model_dataset.index.isin(self._excluded_observations)

        if self.get_response_variable() and self.get_explanatory_variable():

            for i in range(len(self._breakpoints)-1):
                lower_bound = self._breakpoints[i]
                upper_bound = self._breakpoints[i+1]

                segment_range_index = (lower_bound < self._model_dataset.ix[:, self._explanatory_variables[0]]) & \
                                      (self._model_dataset.ix[:, self._explanatory_variables[0]] <= upper_bound)

                model_subset = segment_range_index & ~removed_observation_index

                model_formula = self.get_model_formula(i+1)

                try:
                    segment_model = smf.ols(model_formula,
                                            data=self._model_dataset,
                                            subset=model_subset,
                                            missing='drop')
                except ValueError as err:
                    if err.args[0] == "zero-size array to reduction operation maximum which has no identity":
                        segment_model = None
                    else:
                        raise err

                self._model.append(segment_model)

    def add_breakpoint(self, new_breakpoint):
        """

        :param new_breakpoint:
        :type new_breakpoint: abc.Numeric
        :return:
        """

        breakpoints = np.append(self._breakpoints, new_breakpoint)
        breakpoints = breakpoints[~np.isnan(breakpoints)]
        self._breakpoints = np.sort(breakpoints)

        self.reset_explanatory_var_transform()

        self._create_model()

    def add_explanatory_var_transform(self, segment, transform):
        """

        :param segment:
        :param transform:
        :return:
        """

        self._check_segment_number(segment)
        self._check_transform(transform)

        self._explanatory_variable_transform[segment-1].append(transform)

        self._create_model()

    def get_breakpoints(self):
        """

        :return:
        """

        return copy.deepcopy(self._breakpoints)

    def get_explanatory_variable(self):
        """

        :return:
        """

        return self._explanatory_variables[0]

    def get_model_formula(self, segment=None):
        """

        :param segment:
        :return:
        """

        if segment:

            if self.get_response_variable() and self.get_explanatory_variable():

                self._check_segment_number(segment)

                model_response_variable = self._get_variable_transform(
                    self._response_variable, self._variable_transform[self._response_variable])

                model_explanatory_variables = []

                for transform in self._explanatory_variable_transform[segment-1]:
                    model_explanatory_variables.append(
                        self._get_variable_transform(self._explanatory_variables[0], transform))

                model_formula = model_response_variable + ' ~ ' + ' + '.join(model_explanatory_variables)

            else:

                model_formula = None

        else:

            model_formula = []

            for i in range(0, self.get_number_of_segments()):

                model_formula.append(self.get_model_formula(i+1))

        return model_formula

    def get_model_summary(self):
        """

        :return:
        """

        # TODO: Implement CompoundRatingModel.get_model_summary()

        pass

    def get_number_of_segments(self):
        """

        :return:
        """

        return len(self._breakpoints)-1

    def remove_breakpoint(self, breakpoint):
        """

        :param breakpoint:
        :return:
        """

        new_breakpoints = self._breakpoints[~(self._breakpoints == breakpoint)]
        if np.inf not in new_breakpoints:
            new_breakpoints = np.append(new_breakpoints, np.inf)
        if -np.inf not in new_breakpoints:
            new_breakpoints = np.append(new_breakpoints, -np.inf)

        self._breakpoints = np.sort(new_breakpoints)

        self.reset_explanatory_var_transform()

        self._create_model()

    def remove_explanatory_var_transform(self, segment, transform):
        """

        :param segment:
        :param transform:
        :return:
        """

        self._check_segment_number(segment)

        if transform in self._explanatory_variable_transform:
            self._explanatory_variable_transform[segment-1].remove(transform)
            if len(self._explanatory_variable_transform[segment-1]) < 1:
                self.reset_explanatory_var_transform(segment)

    def reset_breakpoints(self):
        """

        :return:
        """

        self._breakpoints = [-np.inf, np.inf]

        self.reset_explanatory_var_transform()

        self._create_model()

    def reset_explanatory_var_transform(self, segment=None):
        """

        :param segment:
        :return:
        """

        if segment:

            self._check_segment_number(segment)

            self._explanatory_variable_transform[segment-1] = [None]

        else:

            self._explanatory_variable_transform = [[None] for _ in range(self.get_number_of_segments())]

        self._create_model()

    def set_explanatory_variable(self, explanatory_variable):
        """

        :param explanatory_variable:
        :return:
        """

        self._check_variable_names([explanatory_variable])
        self._explanatory_variables = (explanatory_variable,)

    def predict_response_variable(self, explanatory_variable):
        """

        :param explanatory_variable:
        :return:
        """

        # TODO: Implement CompoundRatingModel.predict_response_variable)

        pass
