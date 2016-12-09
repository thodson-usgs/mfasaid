import matplotlib.pyplot as plt
import numpy as np

from datetime import timedelta

import data


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

        avg_window = self._constituent_data_manager.get_surrogate_avg_window(acoustic_parameter)
        avg_window = timedelta(minutes=avg_window)

        fig, (scb_ax, wcb_ax, mb_ax) = plt.subplots(nrows=3, sharex=True)

        for obs_time in observation_times:

            beginning_time = obs_time - avg_window
            ending_time = obs_time + avg_window

            time_window = (beginning_time < cell_range.index) & (cell_range.index <= ending_time)

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


class SurrogateDataPlotCreator:

    def __init__(self, constituent_data_manager):
        """

        :param constituent_data_manager:
        """

        self._constituent_data_manager = constituent_data_manager
        self._surrogate_data_manager = constituent_data_manager.get_surrogate_data_manager()

    def plot_scatter(self, constituent_variable, surrogate_variable, x_log=False, y_log=False, ax=None):

        if ax is None:
            ax = plt.axes()

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
            left_ax = plt.axes()

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

        # determine if the rating model has a constituent data manager
        model_data_manager = surrogate_rating_model.get_data_manager()

        if not isinstance(model_data_manager, data.ConstituentData):
            raise TypeError("Rating model must have a constituent data manager type.")

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

        if included_observation_index.any():
            ax.plot(constituent_variable_series.index[included_observation_index],
                    constituent_variable_series[included_observation_index].as_matrix(),
                    marker='o', markerfacecolor='yellow', markeredgecolor='black',
                    linestyle='None', label='Observations included in model')
        if excluded_observation_index.any():
            ax.plot(constituent_variable_series.index[excluded_observation_index],
                    constituent_variable_series[excluded_observation_index].as_matrix(),
                    marker='s', color='red', linestyle='None', label='Observations excluded from model')
        if missing_observation_index.any():
            ax.plot(constituent_variable_series.index[missing_observation_index],
                    constituent_variable_series[missing_observation_index].as_matrix(),
                    marker='d', color='black', linestyle='None', label='Observations missing from model')

    def _plot_predicted_time_series(self, ax):
        """

        :param ax:
        :return:
        """

        # get predicted data to plot
        constituent_data_manager = self._surrogate_rating_model.get_data_manager()
        surrogate_data_manager = constituent_data_manager.get_surrogate_data_manager()
        predicted_data = self._surrogate_rating_model.predict_response_variable(explanatory_data=surrogate_data_manager,
                                                                                bias_correction=True,
                                                                                prediction_interval=True)

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
            ax = plt.axes()

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
