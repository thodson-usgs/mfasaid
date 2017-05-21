from datetime import timedelta

import pandas as pd
import numpy as np

from datamanager import DataManager


class ConstituentData(DataManager):
    """Data manager class for constituent data"""

    def __init__(self, data, data_origin):
        """

        :param data:
        :param data_origin:
        """
        super().__init__(data, data_origin)

        self._constituent_data = data

    def add_surrogate_data(self, surrogate_data, variable_name=None, time_window_width=0, match_method='nearest'):
        """

        :param surrogate_data: 
        :param variable_name: 
        :param time_window_width: 
        :param match_method: 
        :return: 
        """

        # initialize data for a DataManager
        matched_surrogate_data = pd.DataFrame(index=self._data.index)
        surrogate_variable_origin_data = []

        if variable_name is None:
            variable_names = surrogate_data.get_variable_names()
        else:
            variable_names = [variable_name]

        for variable in variable_names:

            # skip adding the variable if it's in the constituent data set
            if variable in self.get_variable_names():
                continue

            # iterate through all rows and add the matched surrogate observation
            variable_series = pd.Series(index=matched_surrogate_data.index, name=variable)
            for index, _ in matched_surrogate_data.iterrows():
                observation_value = surrogate_data.get_variable_observation(variable, index,
                                                                            time_window_width=time_window_width,
                                                                            match_method=match_method)
                variable_series[index] = observation_value

            # add the origins of the variable to the origin data list
            for origin in surrogate_data.get_variable_origin(variable):
                surrogate_variable_origin_data.append([variable, origin])

            # add the matched variable series to the dataframe
            matched_surrogate_data[variable] = variable_series

        # create a data manager
        surrogate_variable_origin = pd.DataFrame(data=surrogate_variable_origin_data, columns=['variable', 'origin'])
        matched_surrogate_data_manager = DataManager(matched_surrogate_data, surrogate_variable_origin)

        # add the matched surrogate data manager to the constituent data manager
        return self.add_data(matched_surrogate_data_manager)


class SurrogateData(DataManager):
    """Data manager class for surrogate data"""

    def __init__(self, data, data_origin):
        """

        :param data:
        :param data_origin:
        """

        super().__init__(data, data_origin)

        variable_names = self.get_variable_names()

        # initialize matching times to zero
        matching_times = [0] * len(variable_names)
        self._variable_matching_times = dict(zip(variable_names, matching_times))

        # initialize matching methods to nearest
        matching_methods = ['nearest'] * len(variable_names)
        self._variable_matching_method = dict(zip(variable_names, matching_methods))

    def get_variable_observation(self, variable_name, time, time_window_width=0, match_method='nearest'):
        """

        :param variable_name: 
        :param time: 
        :param time_window_width: 
        :param match_method: 
        :return: 
        """

        self._check_variable_name(variable_name)

        # if the default values, use superclass behavior
        if time_window_width == 0 and match_method == 'nearest':
            variable_observation = super().get_variable_observation(variable_name, time)

        else:

            variable = self.get_variable(variable_name)

            # get the subset of times with the variable
            time_diff = timedelta(minutes=time_window_width / 2.)
            beginning_time = time - time_diff
            ending_time = time + time_diff
            time_window = (beginning_time < variable.index) & (variable.index <= ending_time)
            variable_near_time = variable.ix[time_window]

            # match the nearest-in-time observation
            if match_method == 'nearest':
                absolute_time_difference = np.abs(variable_near_time.index - time)
                min_abs_time_diff_index = absolute_time_difference.min() == absolute_time_difference
                nearest_observation = variable_near_time.ix[min_abs_time_diff_index]
                variable_observation = nearest_observation.as_matrix()[0]

            # get the mean observation
            elif match_method == 'mean':
                variable_observation = variable_near_time.mean()

            else:
                msg = 'Unrecognized keyword value for match_method: {}'.format(match_method)
                raise ValueError(msg)

        return variable_observation
