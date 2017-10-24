import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from linearmodel import stats, model as saidmodel


class SurrogateRatingModel:

    def __init__(self, constituent_data, surrogate_data, **kwargs):
        """

        :param constituent_data: 
        :type constituent_data: DataManager
        :param surrogate_data: 
        :type surrogate_data: DataManager
        """

        self._constituent_data = copy.deepcopy(constituent_data)
        self._surrogate_data = copy.deepcopy(surrogate_data)

        surrogate_variables = self._surrogate_data.get_variable_names()
        constituent_variables = self._constituent_data.get_variable_names()

        self._surrogate_variables = kwargs.pop('surrogate_variables', [surrogate_variables[0]])
        self._constituent_variable = kwargs.pop('constituent_variable', constituent_variables[0])

        self._surrogate_transform = dict(zip(surrogate_variables, [(None,)]*len(surrogate_variables)))
        self._constituent_transform = dict(zip(constituent_variables, [None]*len(constituent_variables)))

        match_method = kwargs.pop('match_method', 'nearest')
        self._match_method = dict(zip(surrogate_variables, [match_method]*len(surrogate_variables)))

        match_time = kwargs.pop('match_time', 0)
        self._match_time = dict(zip(surrogate_variables, [match_time]*len(surrogate_variables)))

        self._excluded_observations = pd.DatetimeIndex([])

        self._model = self._create_model()

    def _create_model(self):
        """
        
        :return: 
        """

        model_data = self._get_model_data()

        if len(self._surrogate_variables) > 1:

            model = saidmodel.MultipleLinearOLSModel(model_data,
                                                     response_variable=self._constituent_variable,
                                                     explanatory_variables=self._surrogate_variables)

            for variable in self._surrogate_variables:
                model.transform_explanatory_variable(variable, self._surrogate_transform[variable[0]])

        else:

            surrogate_variable = self._surrogate_variables[0]
            surrogate_variable_transform = self._surrogate_transform[surrogate_variable]

            if len(surrogate_variable_transform) > 1:
                model = saidmodel.ComplexOLSModel(model_data,
                                                  response_variable=self._constituent_variable,
                                                  explanatory_variable=surrogate_variable)
                for transform in surrogate_variable_transform:
                    model.add_explanatory_var_transform(transform)
            else:
                model = saidmodel.SimpleLinearOLSModel(model_data,
                                                       response_variable=self._constituent_variable,
                                                       explanatory_variable=surrogate_variable)
                model.transform_explanatory_variable(surrogate_variable_transform[0])

        model.transform_response_variable(self._constituent_transform[self._constituent_variable])
        model.exclude_observation(self._excluded_observations)

        return model

    def _get_model_data(self):
        """
        
        :return: 
        """

        model_data = copy.deepcopy(self._constituent_data)

        for variable in self._surrogate_variables:
            if variable in model_data.get_variable_names():
                continue
            time_window_width = self._match_time[variable]
            match_method = self._match_method[variable]
            model_data = model_data.match_data(self._surrogate_data, variable, time_window_width, match_method)

        return model_data

    def _plot_constituent_time_series(self, ax):
        """

        :param ax:
        :return:
        """

        model_dataset = self._model.get_model_dataset()
        response_variable = self._model.get_response_variable()

        constituent_variable_series = model_dataset[response_variable]

        missing_observation_index = model_dataset['Missing']
        excluded_observation_index = model_dataset['Excluded']
        included_observation_index = ~(missing_observation_index | excluded_observation_index)

        if excluded_observation_index.any():
            ax.plot(constituent_variable_series.index[excluded_observation_index],
                    constituent_variable_series[excluded_observation_index].as_matrix(),
                    marker='s', color='red', linestyle='None', label='Observations excluded from model')
        if missing_observation_index.any():
            ax.plot(constituent_variable_series.index[missing_observation_index],
                    constituent_variable_series[missing_observation_index].as_matrix(),
                    marker='d', color='black', linestyle='None', label='Observations missing from model')
        if included_observation_index.any():
            ax.plot(constituent_variable_series.index[included_observation_index],
                    constituent_variable_series[included_observation_index].as_matrix(),
                    marker='o', markerfacecolor='yellow', markeredgecolor='black',
                    linestyle='None', label='Observations included in model')

    def _plot_model_time_series(self, ax):
        """

        :return:
        """

        # plot the predicted time series
        self._plot_predicted_time_series(ax)
        self._plot_constituent_time_series(ax)

        # set the y scale to logarithmic if the response variable is log transformed
        response_variable_name = self._model.get_response_variable()
        response_transform = self._model.get_variable_transform(response_variable_name)
        if (response_transform is 'log') or (response_transform is 'log10'):
            ax.set_yscale('log')

        ax.set_ylabel(response_variable_name)
        ax.legend(loc='best', numpoints=1)

    def _plot_predicted_time_series(self, ax):
        """

        :param ax:
        :return:
        """

        # get predicted data to plot
        predicted_data = self._model.predict_response_variable(
            explanatory_data=self._surrogate_data, bias_correction=True, prediction_interval=True)

        # mean response
        response_variable_name = self._model.get_response_variable()
        ax.plot(predicted_data.index, predicted_data[response_variable_name].as_matrix(), color='blue',
                linestyle='None', marker='.', markersize=5,
                label='Predicted ' + response_variable_name)

        # lower prediction interval
        lower_interval_name = response_variable_name + '_L90.0'

        # upper prediction interval
        upper_interval_name = response_variable_name + '_U90.0'

        ax.fill_between(predicted_data.index,
                        predicted_data[lower_interval_name].as_matrix(),
                        predicted_data[upper_interval_name].as_matrix(),
                        facecolor='gray', edgecolor='gray', alpha=0.5, label='90% Prediction interval')

    def _plot_observation_quantile(self, surrogate_variable_name, ax):
        """

        :param surrogate_variable_name:
        :param ax:
        :return:
        """

        # get the surrogate variable plotting positions
        surrogate_variable = self._surrogate_data.get_variable(surrogate_variable_name)
        surrogate_variable = surrogate_variable.dropna()
        surrogate_variable = np.squeeze(surrogate_variable)
        surrogate_pp = stats.calc_plotting_position(surrogate_variable)

        # plot the surrogate plotting positions
        ax.plot(surrogate_variable, surrogate_pp, marker='.', markersize=5,
                linestyle='None', label='Surrogate observation')

        # find the information to plot model observations dates included in the model
        model_dataset = self._model.get_model_dataset()
        missing_or_excluded_index = model_dataset['Missing'] | model_dataset['Excluded']
        included_dates = model_dataset.index[~missing_or_excluded_index]

        # find the surrogate values (and plotting position) closest-in-time to model observations
        model_obs_surrogate_value = np.empty(included_dates.shape)
        model_obs_pp = np.empty(included_dates.shape)
        for i in range(len(included_dates)):

            # find the minimum absolute time difference
            abs_time_diff = np.abs(included_dates[i] - surrogate_variable.index)
            min_abs_time_diff = np.min(abs_time_diff)

            # get the closest observation index
            closest_obs_index = abs_time_diff == min_abs_time_diff

            # set the surrogate value and plotting position to the earliest occurrence
            model_obs_surrogate_value[i] = surrogate_variable.ix[closest_obs_index][0]
            model_obs_pp[i] = surrogate_pp[closest_obs_index][0]

        # plot model observation occurrences
        ax.plot(model_obs_surrogate_value, model_obs_pp,
                marker='o', markerfacecolor='yellow', markeredgecolor='black',
                linestyle='None', label='Model observation')

        ax.set_xlabel(surrogate_variable_name)
        ax.set_ylabel('Cumulative frequency')

        ax.legend(loc='best', numpoints=1)

    def add_surrogate_transform(self, surrogate_variable, surrogate_transform):
        """

        :param surrogate_variable:
        :param surrogate_transform: 
        :return: 
        """

        saidmodel.RatingModel.check_transform(surrogate_transform)

        surrogate_variables = self._surrogate_data.get_variable_names()
        if surrogate_variable not in surrogate_variables:
            raise ValueError("Invalid surrogate variable name: {}".format(surrogate_variable))

        self._surrogate_transform[surrogate_variable] = self._surrogate_transform[surrogate_variable] + \
                                                        (surrogate_transform,)

    def exclude_observations(self, observations):
        """

        :param observations: 
        :return: 
        """

        self._excluded_observations = observations
        self._model = self._create_model()

    def get_constituent_transform(self):
        """

        :return: 
        """

        return self._constituent_transform[self._constituent_variable]

    def get_constituent_variable(self):
        """

        :return: 
        """

        return self._constituent_variable

    def get_constituent_variable_names(self):
        """
        
        :return: 
        """

        return self._constituent_data.get_variable_names()

    def get_surrogate_transform(self):
        """

        :return: 
        """

        return {var: self._surrogate_transform[var] for var in self._surrogate_variables}

    def get_surrogate_variable_names(self):
        """
        
        :return: 
        """

        return self._surrogate_data.get_variable_names()

    def get_surrogate_variables(self):
        """

        :return: 
        """

        return copy.deepcopy(self._surrogate_variables)

    def get_model_dataset(self):
        """

        :return:
        """
        return self._model.get_model_dataset()

    def get_model_report(self):
        """

        :return: 
        """

        return self._model.get_model_report()

    def plot(self, plot_type='model_scatter', ax=None, **kwargs):
        """

        :param plot_type:
        :param ax:
        :return:
        """

        # create an axis if one isn't passed
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        if plot_type == 'time series':
            self._plot_model_time_series(ax)
        elif plot_type == 'quantile':
            # TODO: figure out how combinations of kwargs can be enforced
            surrogate_variable = kwargs.pop('surrogate_variable', self._surrogate_variables[0])
            self._plot_observation_quantile(surrogate_variable, ax)
        else:
            self._model.plot(plot_type, ax)

        return ax

    def set_constituent_transform(self, constituent_transform):
        """

        :param constituent_transform: 
        :return: 
        """

        saidmodel.RatingModel.check_transform(constituent_transform)
        self._constituent_transform[self._constituent_variable] = constituent_transform
        self._model = self._create_model()

    def set_constituent_variable(self, constituent_variable):
        """
        
        :param constituent_variable: 
        :return: 
        """

        constituent_variable_names = self.get_constituent_variable_names()
        if constituent_variable not in constituent_variable_names:
            raise ValueError("Invalid constituent variable name: {}".format(constituent_variable))

    def set_observation_match_method(self, method, time):
        """

        :param method: 
        :param time: 
        :return: 
        """

        if method != 'nearest' or method != 'mean':
            raise ValueError("Invalid match method: {}".format(method))

        if time < 0:
            raise ValueError("time must be greater than or equal to zero.")

        match_method = dict(zip(self._surrogate_variables, [method]*len(self._surrogate_variables)))
        match_time = dict(zip(self._surrogate_variables, [time]*len(self._surrogate_variables)))

        self._match_method.update(match_method)
        self._match_time.update(match_time)

        self._model = self._create_model()

    def set_surrogate_transform(self, surrogate_transform, surrogate_variable=None):
        """

        :param surrogate_variable:
        :param surrogate_transform:
        :return:
        """

        saidmodel.RatingModel.check_transform(surrogate_transform)
        if surrogate_variable is None:
            surrogate_variable = self._surrogate_data.get_variable_names()[0]
        elif surrogate_variable not in self._surrogate_data.get_variable_names():
            raise ValueError("Invalid variable name: {}".format(surrogate_variable))

        self._surrogate_transform[surrogate_variable] = surrogate_transform
        self._model = self._create_model()

    def set_surrogate_variables(self, surrogate_variables):
        """
        
        :param surrogate_variables: 
        :return: 
        """

        for variable in surrogate_variables:
            if variable not in self._surrogate_data.get_variable_names():
                raise ValueError("Invalid variable name: {}".format(variable))

        self._surrogate_variables = copy.deepcopy(list(surrogate_variables))
        self._model = self._create_model()
