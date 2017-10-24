from datetime import timedelta

import matplotlib.colors
import matplotlib.lines
import matplotlib.pyplot as plt
import numpy as np

import datamanager
from linearmodel import stats


class LineStyleGenerator:

    def __init__(self):

        self._line_color_iterator = self._create_line_color_iterator()
        self._line_marker_iterator = self._create_line_marker_iterator()
        self._line_style_iterator = self._create_line_style_iterator()

    @staticmethod
    def _create_line_color_iterator():

        base_color_keys = matplotlib.colors.BASE_COLORS.keys()
        base_color_iterator = iter(base_color_keys)

        return base_color_iterator

    @staticmethod
    def _create_line_marker_iterator():

        line_marker_keys = matplotlib.lines.lineMarkers.keys()
        line_marker_iterator = iter(line_marker_keys)

        return line_marker_iterator

    @staticmethod
    def _create_line_style_iterator():

        line_style_keys = matplotlib.lines.lineStyles.keys()
        line_style_iterator = iter(line_style_keys)

        return line_style_iterator

    def get_line_color(self):

        try:
            line_color = next(self._line_color_iterator)
        except StopIteration:
            self._line_color_iterator = self._create_line_color_iterator()
            line_color = self.get_line_color()

        return line_color

    def get_line_properties(self):

        line_color = self.get_line_color()
        line_marker = self.get_marker()
        line_style = self.get_line_style()

        return line_color, line_style, line_marker

    def get_line_style(self, draw_nothing=True):

        try:
            line_style = next(self._line_style_iterator)
        except StopIteration:
            self._line_style_iterator = self._create_line_style_iterator()
            line_style = self.get_line_style()

        line_style_description = matplotlib.lines.lineStyles[line_style]

        if (not draw_nothing) and line_style_description == '_draw_nothing':

            line_style = self.get_line_style(draw_nothing)

        return line_style

    def get_line_style_string(self):

        line_color = self.get_line_color()
        line_style = self.get_line_style()
        line_marker = self.get_marker()

        line_style_description = matplotlib.lines.lineStyles[line_style]
        line_marker_description = matplotlib.lines.lineMarkers[line_marker]

        while line_style_description == '_draw_nothing' and line_marker_description == 'nothing':
            line_style = self.get_line_style()
            line_style_description = matplotlib.lines.lineStyles[line_style]

        line_style_string = line_color + line_style + line_marker
        line_style_string = line_style_string.replace('None', '')

        return line_style_string

    def get_marker(self):

        try:
            line_marker = next(self._line_marker_iterator)
        except StopIteration:
            self._line_marker_iterator = self._create_line_marker_iterator()
            line_marker = self.get_marker()

        return line_marker


class AcousticProfilePlotCreator:
    """Plot acoustic profiles from """

    _line_styles = ['-', '--', '-.', ':']
    _colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    def __init__(self, constituent_data_manager):
        """

        :param constituent_data_manager:
        """

        self._constituent_data_manager = constituent_data_manager

    def plot_backscatter_profiles(self, acoustic_parameter='MeanSCB', constituent_observation_times=None):
        """

        :param acoustic_parameter:
        :param constituent_observation_times:
        :return:
        """

        data_index = self._constituent_data_manager.get_data().index

        if constituent_observation_times is None:
            constituent_observation_times = data_index

        observation_index = data_index.isin(constituent_observation_times)

        observation_times = data_index[observation_index]

        advm_data = self._constituent_data_manager.get_surrogate_data_manager()

        cell_range = advm_data.get_cell_range()
        measured_backscatter = advm_data.get_mb()
        water_corrected_backscatter = advm_data.get_wcb()
        sediment_corrected_backscatter = advm_data.get_scb()

        surrogate_match_method = self._constituent_data_manager.get_surrogate_match_method(acoustic_parameter)

        if surrogate_match_method == 'average':
            time_window_width = self._constituent_data_manager.get_surrogate_avg_window(acoustic_parameter)
            time_window_width = timedelta(minutes=time_window_width)
        elif surrogate_match_method == 'closest':
            time_window_width = self._constituent_data_manager.get_surrogate_max_abs_time_diff(acoustic_parameter)

        fig, (scb_ax, wcb_ax, mb_ax) = plt.subplots(nrows=3, sharex=True)

        for obs_time in observation_times:

            if surrogate_match_method == 'average':
                beginning_time = obs_time - time_window_width
                ending_time = obs_time + time_window_width
                time_window = (beginning_time < cell_range.index) & (cell_range.index <= ending_time)

            elif surrogate_match_method == 'closest':
                surrogate_obs = advm_data.get_closest_variable_observation(acoustic_parameter, obs_time)
                time_window = surrogate_obs.ix[0].name == cell_range.index

            obs_cell_range = cell_range.ix[time_window, :].as_matrix()
            obs_mb = measured_backscatter.ix[time_window, :].as_matrix()
            obs_wcb = water_corrected_backscatter.ix[time_window, :].as_matrix()
            obs_scb = sediment_corrected_backscatter.ix[time_window, :].as_matrix()

            obs_number = np.nonzero(obs_time==data_index)[0]

            color_index = int(obs_number % len(self._colors))
            obs_color = self._colors[color_index]

            line_style_index = int(obs_number//len(self._line_styles) % len(self._line_styles))
            obs_line_style = self._line_styles[line_style_index]

            mb_ax.plot(obs_cell_range.transpose(), obs_mb.transpose(), ls=obs_line_style, color=obs_color, marker='.')
            mb_ax.set_xlabel('Cell range, in meters')
            mb_ax.set_ylabel('Measured backscatter,\nin decibels')

            wcb_ax.plot(obs_cell_range.transpose(), obs_wcb.transpose(), ls=obs_line_style, color=obs_color, marker='.')
            wcb_ax.set_ylabel('Water corrected\nbackscatter, in decibels')

            scb_ax.plot(obs_cell_range.transpose(), obs_scb.transpose(), ls=obs_line_style, color=obs_color, marker='.')
            scb_ax.set_ylabel('Sediment corrected\nbackscatter, in decibels')

        return fig


class ConstituentDataSetPlotCreator:

    def __init__(self, constituent_data_manager):
        """

        :param constituent_data_manager:
        """

        self._constituent_data_manager = constituent_data_manager
        self._surrogate_data_manager = constituent_data_manager.get_surrogate_data_manager()

    def plot_scatter(self, constituent_variable, surrogate_variable, x_log=False, y_log=False, ax=None):

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        constituent_series = self._constituent_data_manager.get_variable(constituent_variable)
        surrogate_series = self._constituent_data_manager.get_variable(surrogate_variable)

        ax.plot(surrogate_series, constituent_series, '.')
        ax.set_xlabel(surrogate_variable)
        ax.set_ylabel(constituent_variable)

        if x_log:
            ax.set_xscale('log')
        if y_log:
            ax.set_yscale('log')

        return ax

    def plot_time_series(self, constituent_variable, surrogate_variable, left_log=False, right_log=False, left_ax=None):

        if left_ax is None:
            fig = plt.figure()
            left_ax = fig.add_subplot(111)

        right_ax = plt.twinx(left_ax)

        constituent_time_series = self._constituent_data_manager.get_variable(constituent_variable)

        left_line, = left_ax.plot(constituent_time_series.index, constituent_time_series.as_matrix(),
                                  marker='o', markerfacecolor='yellow', markeredgecolor='black', markersize=5,
                                  linestyle='None', label='Constituent observations')
        left_ax.set_ylabel(constituent_variable)
        if left_log:
            left_ax.set_yscale('log')

        surrogate_time_series = self._surrogate_data_manager.get_variable(surrogate_variable)
        right_line, = right_ax.plot(surrogate_time_series.index, surrogate_time_series.as_matrix(),
                                    marker='.', color='blue', markersize=2,
                                    linestyle='None', label='Surrogate observations')
        right_ax.set_ylabel(surrogate_variable)
        if right_log:
            right_ax.set_yscale('log')

        left_ax.legend(handles=[left_line, right_line], loc='best', numpoints=1)

        return [left_ax, right_ax]


class SurrogateModelPlotCreator:

    def __init__(self, surrogate_rating_model):
        """

        :param surrogate_rating_model:
        """

        self._surrogate_rating_model = surrogate_rating_model

        # determine if the rating model is a constituent-surrogate model
        self._constituent_data_manager = model_data_manager = surrogate_rating_model.get_data_manager()

        self._surrogate_data_manager = model_data_manager.get_surrogate_data_manager()

        if not isinstance(self._surrogate_data_manager, datamanager.SurrogateData):
            raise TypeError("Rating model data manager must have an associated surrogate data manager.")

    def _plot_constituent_time_series(self, ax):
        """

        :param ax:
        :return:
        """

        model_dataset = self._surrogate_rating_model.get_model_dataset()
        response_variable = self._surrogate_rating_model.get_response_variable()

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

    def _plot_predicted_time_series(self, ax):
        """

        :param ax:
        :return:
        """

        # get predicted data to plot
        predicted_data = self._surrogate_rating_model.predict_response_variable(
            explanatory_data=self._surrogate_data_manager, bias_correction=True, prediction_interval=True)

        # mean response
        response_variable_name = self._surrogate_rating_model.get_response_variable()
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

    def plot_model_time_series(self, ax=None):
        """

        :return:
        """
        # create an axis if one isn't passed
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        # plot the predicted time series
        self._plot_predicted_time_series(ax)
        self._plot_constituent_time_series(ax)

        # set the y scale to logarithmic if the response variable is log transformed
        response_variable_name = self._surrogate_rating_model.get_response_variable()
        response_transform = self._surrogate_rating_model.get_variable_transform(response_variable_name)
        if (response_transform is 'log') or (response_transform is 'log10'):
            ax.set_yscale('log')

        ax.set_ylabel(response_variable_name)
        ax.legend(loc='best', numpoints=1)

        return ax

    def plot_observation_quantile(self, surrogate_variable_name, ax=None):
        """

        :param surrogate_variable_name:
        :param ax:
        :return:
        """

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        # get the surrogate variable plotting positions
        surrogate_variable = self._surrogate_data_manager.get_variable(surrogate_variable_name)
        surrogate_variable = surrogate_variable.dropna()
        surrogate_variable = np.squeeze(surrogate_variable)
        surrogate_pp = stats.calc_plotting_position(surrogate_variable)

        # plot the surrogate plotting positions
        ax.plot(surrogate_variable, surrogate_pp, marker='.', markersize=5,
                linestyle='None', label='Surrogate observation')

        # find the information to plot model observations
        # dates included in the model
        model_dataset = self._surrogate_rating_model.get_model_dataset()
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

        return ax
