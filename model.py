import abc
import copy

import numpy as np
from numpy import log, log10, power

from scipy import stats

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.iolib.table import SimpleTable
from statsmodels.iolib.summary import Summary
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

        :param data_manager: Data manager containing response and explanatory variables.
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
        self._model_data_origin = pd.DataFrame(columns=['variable', 'origin'])

        # initialize the variable transform dictionary
        self._variable_transform = {}
        for variable in variable_names:
            self._variable_transform[variable] = None

        # initialize the model attribute
        self._model = None

    @staticmethod
    def _calc_plotting_position(x, a=0.4):
        """

        :param data:
        :param a:
        :return:
        """

        x = np.asarray(x)

        Nx = x.shape[0]

        sorted_index = np.argsort(x)

        rank = np.zeros(Nx, int)
        rank[sorted_index] = np.arange(Nx) + 1

        pp = (rank - a) / (Nx + 1 - 2 * a)

        return pp

    @staticmethod
    def _calc_quantile(x, q):
        """

        :param x:
        :param q:
        :return:
        """

        pp = RatingModel._calc_plotting_position(x)

        sorted_index = np.argsort(x)

        xp = x[sorted_index]
        pp = pp[sorted_index]

        quantile = np.interp(q, pp, xp)

        return quantile

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

        origin_data = []
        for variable in (self._response_variable,) + self._explanatory_variables:
            for origin in self._data_manager.get_variable_origin(variable):
                origin_data.append([variable, origin])

        self._model_data_origin = pd.DataFrame(data=origin_data, columns=['variable', 'origin'])

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
        :type observation_time:
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
        """Set the response variable of the model.

        :param response_variable:
        :type response_variable: abc.basestring
        :return:
        """

        self._check_variable_names([response_variable])

        self._response_variable = response_variable

        self.update_model()

    def transform_response_variable(self, transform):
        """

        :param transform:
        :return:
        """

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

    def _calc_ppcc(self):
        """Calculate the probability plot correlation coefficient

        :return:
        """

        res = self._model.fit()
        normal_quantile = self._calc_res_normal_quantile()

        ppcc, _ = stats.pearsonr(normal_quantile, res.resid)

        return ppcc

    def _calc_res_normal_quantile(self):
        """

        :return:
        """

        res = self._model.fit()
        plotting_position = self._calc_plotting_position(res.resid)
        loc, scale = stats.norm.fit(res.resid)
        dist = stats.norm(loc, scale)
        normal_quantile = dist.ppf(plotting_position)

        quantile_series = pd.Series(index=res.resid.index, data=normal_quantile, name='Normal quantile of residual')

        return quantile_series

    def _create_model(self):
        """

        :return:
        """

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

        self._model = model

    @abc.abstractmethod
    def _get_exogenous_matrix(self, exogenous_df):
        pass

    def _get_model_equation(self):
        """

        :return:
        """

        res = self._model.fit()

        # get the model equation with the estimated coefficients
        response_variable = self._get_variable_transform(self._response_variable,
                                                         self._variable_transform[self._response_variable])
        explanatory_variables = []
        for variable in self._explanatory_variables:
            variable_name = self._get_variable_transform(variable, self._variable_transform[variable])
            explanatory_variables.append('{:.5g}'.format(res.params[variable_name]) + variable_name)
        model_equation = response_variable + ' = ' \
            + '{:.5g}'.format(res.params['Intercept']) + ' + '\
            + ' + '.join(explanatory_variables)

        return SimpleTable(data=[[model_equation]], headers=['Linear regression model:'])

    def _get_variable_summary(self, model_variables, table_title):
        """

        :param model_variables:
        :param table_title:
        :return:
        """

        table_data = [[''], ['Minimum'], ['1st Quartile'], ['Median'], ['Mean'], ['3rd Quartile'], ['Maximum']]

        number_format_str = '{:.5g}'

        q = np.array([0, 0.25, 0.5, 0.75, 1])

        for variable in model_variables:

            variable_series = self._model_dataset[variable]

            quantiles = self._calc_quantile(variable_series, q)

            table_data[0].append(variable)
            table_data[1].append(number_format_str.format(quantiles[0]))
            table_data[2].append(number_format_str.format(quantiles[1]))
            table_data[3].append(number_format_str.format(quantiles[2]))
            table_data[4].append(number_format_str.format(variable_series.mean()))
            table_data[5].append(number_format_str.format(quantiles[3]))
            table_data[6].append(number_format_str.format(quantiles[4]))

            variable_transform = self._variable_transform[variable]

            if variable_transform:

                variable_transform_name = self._get_variable_transform(variable, variable_transform)

                transform_function = self._transform_functions[variable_transform]

                variable_transform_series = transform_function(variable_series)

                table_data[0].append(variable_transform_name)
                table_data[1].append(number_format_str.format(variable_transform_series.min()))
                table_data[2].append(number_format_str.format(variable_transform_series.quantile(0.25)))
                table_data[3].append(number_format_str.format(variable_transform_series.quantile(0.5)))
                table_data[4].append(number_format_str.format(variable_transform_series.mean()))
                table_data[5].append(number_format_str.format(variable_transform_series.quantile(0.75)))
                table_data[6].append(number_format_str.format(variable_transform_series.max()))

        table_header = [table_title]

        table_header.extend([''] * (len(table_data[0])-1))

        variable_summary = SimpleTable(data=table_data, headers=table_header)

        return variable_summary

    def get_explanatory_variable_summary(self):
        """

        :return:
        """

        table_title = 'Explanatory variable summary'

        return self._get_variable_summary(self._explanatory_variables, table_title)

    def get_model_dataset(self):
        """Returns a pandas DataFrame containing the following columns:

            Observed response variable
            Observed explanatory variables
            Missing and excluded observation indicators
            Fitted transformed response variable
            Raw residual
            Estimated response variable, with Duan smearing estimate applied
            Normal quantile
            Standardized (internally studentized) residual
            Leverage
            Cook's distance
            DFFITS

        :return:
        """

        model_dataset = super().get_model_dataset()

        res = self._model.fit()

        model_data_index = res.resid.index

        response_variable = self.get_response_variable()

        # add fitted values
        response_variable_transform = self._variable_transform[response_variable]
        transformed_response_variable_name = self._get_variable_transform(response_variable,
                                                                          response_variable_transform)
        fitted_values = res.fittedvalues.rename('Fitted ' + transformed_response_variable_name)

        # add raw residuals
        raw_residuals = res.resid.rename('Raw Residual')

        # add estimated response
        explanatory_data = data.DataManager(self._model_dataset.ix[model_data_index, :], self._model_data_origin)
        predicted_response = self.predict_response_variable(explanatory_data=explanatory_data, bias_correction=True)
        estimated_response = predicted_response[response_variable]
        estimated_response = estimated_response.rename('Estimated ' + response_variable)

        # add quantile
        quantile_series = self._calc_res_normal_quantile()

        ols_influence = res.get_influence()

        # add standardized residuals (also known as internally studentized residuals)
        standardized_residuals = pd.Series(index=model_data_index,
                                           data=ols_influence.resid_studentized_internal,
                                           name='Standardized Residual')

        # add leverage
        leverage = pd.Series(index=model_data_index,
                             data=ols_influence.hat_matrix_diag,
                             name='Leverage')

        # add Cook's D
        cooks_distance = pd.Series(index=model_data_index,
                                   data=ols_influence.cooks_distance[0],
                                   name="Cook's Distance")

        # add DFFITS
        dffits = pd.Series(index=model_data_index,
                           data=ols_influence.dffits[0],
                           name="DFFITS")

        model_dataset = pd.concat([model_dataset,
                                   fitted_values,
                                   raw_residuals,
                                   estimated_response,
                                   quantile_series,
                                   standardized_residuals,
                                   leverage,
                                   cooks_distance,
                                   dffits], axis=1)

        return model_dataset

    @abc.abstractmethod
    def get_model_formula(self):
        pass

    def get_model_report(self):
        """

        :return:
        """

        # get a table for the data origins
        data_origin = []
        for variable in (self._response_variable, ) + self._explanatory_variables:
            for origin in self._data_manager.get_variable_origin(variable):
                if origin not in data_origin:
                    data_origin.append([origin])
        origin_table = SimpleTable(data=data_origin, headers=['Data file location'])

        model_equation = self._get_model_equation()

        # get the model summary
        model_report = self.get_model_summary()

        # create a SimpleTable for the model dataset
        model_dataset = self.get_model_dataset()
        index_as_str = np.expand_dims(model_dataset.index.astype(str), 1)
        observation_data = np.column_stack((index_as_str, model_dataset.as_matrix()))
        observation_data_headers = ['DateTime']
        observation_data_headers.extend(model_dataset.keys())
        observation_table = SimpleTable(data=observation_data, headers=observation_data_headers)

        response_variable_summary = self.get_response_variable_summary()
        explanatory_variable_summary = self.get_explanatory_variable_summary()

        model_report.tables.extend([model_equation,
                                    origin_table,
                                    response_variable_summary,
                                    explanatory_variable_summary,
                                    observation_table])

        return model_report

    def get_model_summary(self):
        """

        :return:
        """

        summary = Summary()
        res = self._model.fit()

        string_format = '{:.5g}'

        number_of_observations = ('Number of observations', [string_format.format(res.nobs)])
        error_degrees_of_freedom = ('Error degrees of freedom', [string_format.format(res.df_resid)])
        rmse = ('Root mean squared error', [string_format.format(np.sqrt(res.mse_resid))])
        rsquared = ('R-squared', [string_format.format(res.rsquared)])
        adjusted_rsquared = ('Adjusted R-squared', [string_format.format(res.rsquared_adj)])
        fvalue = ('F-statistic vs. constant model', [string_format.format(res.fvalue)])
        pvalue = ('p-value', [string_format.format(res.f_pvalue)])

        gleft = [number_of_observations, error_degrees_of_freedom, rmse]
        gright = [rsquared, adjusted_rsquared, fvalue, pvalue]

        summary.add_table_2cols(res, gleft=gleft, gright=gright)

        summary.add_table_params(res, alpha=0.1)

        return summary

    def get_response_variable_summary(self):
        """

        :return:
        """

        table_title = 'Response variable summary'

        return self._get_variable_summary((self._response_variable, ), table_title)

    def predict_response_variable(self, explanatory_data=None, bias_correction=False, prediction_interval=False):
        """

        :param explanatory_data:
        :param bias_correction:
        :param prediction_interval:
        :return:
        """

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
            predicted = predicted.join(explanatory_df[list(self._explanatory_variables)], how='outer')

        else:

            predicted = pd.DataFrame(columns=[self._response_variable] + list(self._explanatory_variables))

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

        # self.update_model()

    def _get_exogenous_matrix(self, exogenous_df):
        """

        :param exogenous_df:
        :return:
        """

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
        else:
            self.set_explanatory_variables(data_manager.get_variable_names()[1:])

        # self.update_model()

    def _get_exogenous_matrix(self, exogenous_df):
        """

        :param exogenous_df:
        :return:
        """

        for variable in self._explanatory_variables:
            assert(variable in exogenous_df.keys())

        exog = pd.DataFrame()

        for variable in self._explanatory_variables:

            transform = self._variable_transform[variable]
            transform_function = self._transform_functions[transform]
            transformed_variable_name = self._get_variable_transform(variable, transform)
            exog[transformed_variable_name] = transform_function(exogenous_df[variable])

        exog = sm.add_constant(exog)

        return exog

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
        else:
            self.set_explanatory_variable(data_manager.get_variable_names()[1])

        # self.update_model()

    def _get_exogenous_matrix(self, exogenous_df):
        """

        :return:
        """

        explanatory_variable = self.get_explanatory_variable()

        assert(explanatory_variable in exogenous_df.keys())

        exog = pd.DataFrame()

        for transform in self._explanatory_variable_transform:

            transformed_variable_name = self._get_variable_transform(explanatory_variable, transform)
            transform_function = self._transform_functions[transform]
            exog[transformed_variable_name] = transform_function(exogenous_df[explanatory_variable])

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

        self._explanatory_variable_transform = [[None]]

        self._breakpoints = np.array([-np.inf, np.inf])

        self._model = []
        # self._create_model_dataset()

        if explanatory_variable:
            self.set_explanatory_variable(explanatory_variable)
        else:
            self.set_explanatory_variable(data_manager.get_variable_names()[1])

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

        for i in range(self.get_number_of_segments()):
            lower_bound = self._breakpoints[i]
            upper_bound = self._breakpoints[i+1]

            segment_range_index = (lower_bound <= self._model_dataset.ix[:, self._explanatory_variables[0]]) & \
                                  (self._model_dataset.ix[:, self._explanatory_variables[0]] < upper_bound)

            origin_data = []
            for variable in self._response_variable, self._explanatory_variables[0]:
                for origin in self._data_manager.get_variable_origin(variable):
                    origin_data.append([variable, origin])
            model_data_origin = pd.DataFrame(data=origin_data, columns=['variable', 'origin'])

            segment_data_manager = data.DataManager(self._model_dataset.ix[segment_range_index, :], model_data_origin)

            segment_model = ComplexRatingModel(segment_data_manager,
                                               response_variable=self.get_response_variable(),
                                               explanatory_variable=self.get_explanatory_variable())
            segment_model.exclude_observation(self.get_excluded_observations())

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

    def add_explanatory_var_transform(self, transform, segment=None):
        """

        :param segment:
        :param transform:
        :return:
        """

        self._check_transform(transform)

        if segment:
            self._check_segment_number(segment)
            self._model[segment-1].add_explanatory_var_transform(transform)
        else:
            for segment_model in self._model:
                segment_model.add_explanatory_var_transform(transform)

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

    def get_model_dataset(self):
        """

        :return:
        """

        model_dataset = pd.DataFrame()

        for i in range(self.get_number_of_segments()):
            segment_model_dataset = self._model[i].get_model_dataset()
            segment_model_dataset['Segment'] = i+1
            model_dataset = pd.concat([model_dataset, segment_model_dataset], verify_integrity=True)

        model_dataset.sort_index(inplace=True)

        return model_dataset

    def get_model_formula(self, segment=None):
        """

        :param segment:
        :return:
        """

        if segment:

            self._check_segment_number(segment)

            model_formula = self._model[segment-1].get_model_formula()

        else:

            model_formula = []

            for segment_model in self._model:

                model_formula.append(segment_model.get_model_formula())

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

    def remove_explanatory_var_transform(self, transform, segment=None):
        """

        :param segment:
        :param transform:
        :return:
        """

        if segment:
            self._check_segment_number(segment)
            self._model[segment-1].remove_explanatory_var_transform(transform)

        else:

            for segment_model in self._model:
                segment_model.remove_explanatory_var_transform(transform)

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

            self._model[segment-1].reset_explanatory_var_transform()

        else:

            for segment_model in self._model:

                segment_model.reset_explanatory_var_transform()

    def set_explanatory_variable(self, explanatory_variable):
        """

        :param explanatory_variable:
        :return:
        """

        self._check_variable_names([explanatory_variable])
        self._explanatory_variables = (explanatory_variable,)

        self.update_model()

    def transform_response_variable(self, transform):
        """

        :param transform:
        :return:
        """

        self._check_transform(transform)

        for segment_model in self._model:
            segment_model.transform_response_variable(transform)

    def predict_response_variable(self, explanatory_data=None, bias_correction=False, prediction_interval=False):
        """

        :param explanatory_data:
        :param bias_correction:
        :param prediction_interval:
        :return:
        """

        predicted = pd.DataFrame()

        if explanatory_data:
            explanatory_df = explanatory_data.get_data()
            explanatory_origin = explanatory_data.get_origin()
        else:
            explanatory_df = self._model_dataset.copy(deep=True)
            explanatory_origin = self._model_data_origin.copy(deep=True)

        explanatory_series = explanatory_df[self.get_explanatory_variable()]

        for i in range(self.get_number_of_segments()):
            lower_bound = self._breakpoints[i]
            upper_bound = self._breakpoints[i+1]
            segment_index = (lower_bound <= explanatory_series) & (explanatory_series < upper_bound)
            predictor_data_manager = data.DataManager(explanatory_df.ix[segment_index, :],
                                                      explanatory_origin)
            segment_predicted = self._model[i].predict_response_variable(explanatory_data=predictor_data_manager,
                                                                         bias_correction=bias_correction,
                                                                         prediction_interval=prediction_interval)
            predicted = pd.concat([predicted, segment_predicted])

        return predicted
