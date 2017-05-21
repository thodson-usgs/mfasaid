import copy
from datetime import timedelta

import pandas as pd
import numpy as np

from datamanager import DataManager
import model as saidmodel


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


class SurrogateRatingModel:

    def __init__(self, constituent_data, surrogate_data):
        """

        :param constituent_data: 
        :type constituent_data: ConstituentData
        :param surrogate_data: 
        :type surrogate_data: SurrogateData
        """

        self._constituent_data = copy.deepcopy(constituent_data)
        self._surrogate_data = copy.deepcopy(surrogate_data)

        surrogate_variables = self._surrogate_data.get_variable_names()
        constituent_variables = self._constituent_data.get_variable_names()

        self._surrogate_variables = [surrogate_variables[0]]
        self._constituent_variable = constituent_variables[0]

        self._surrogate_transform = dict(zip(surrogate_variables, [(None,)]*len(surrogate_variables)))
        self._constituent_transform = dict(zip(constituent_variables, [None]*len(constituent_variables)))

        self._match_method = dict(zip(surrogate_variables, ['nearest']*len(surrogate_variables)))
        self._match_time = dict(zip(surrogate_variables, [0]*len(surrogate_variables)))

        self._excluded_observations = pd.DatetimeIndex([])

        self._model = self._create_model()

    def _create_model(self):
        """
        
        :return: 
        """

        model_data = self._get_model_data()

        if len(self._surrogate_variables) > 1:

            model = saidmodel.MultipleLinearRatingModel(model_data,
                                                        response_variable=self._constituent_variable,
                                                        explanatory_variables=self._surrogate_variables)

            for variable in self._surrogate_variables:
                model.transform_explanatory_variable(variable, self._surrogate_transform[variable[0]])

        else:

            surrogate_variable = self._surrogate_variables[0]
            surrogate_variable_transform = self._surrogate_transform[surrogate_variable]

            if len(surrogate_variable_transform) > 1:
                model = saidmodel.ComplexRatingModel(model_data,
                                                     response_variable=self._constituent_variable,
                                                     explanatory_variable=surrogate_variable)
                for transform in surrogate_variable_transform:
                    model.add_explanatory_var_transform(transform)
            else:
                model = saidmodel.SimpleLinearRatingModel(model_data,
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
            time_window_width = self._match_time[variable]
            match_method = self._match_method[variable]
            model_data = model_data.add_surrogate_data(self._surrogate_data, variable, time_window_width, match_method)

        return model_data

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

    def get_model_report(self):
        """

        :return: 
        """

        return self._model.get_model_report()

    def plot(self, plot_type):
        """

        :param plot_type: 
        :return: 
        """

    def set_constituent_transform(self, constituent_transform):
        """

        :param constituent_transform: 
        :return: 
        """

        saidmodel.RatingModel.check_transform(constituent_transform)
        self._constituent_transform = constituent_transform
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

    def set_surrogate_transform(self, surrogate_transform):
        """

        :param surrogate_transform: 
        :return: 
        """

    def set_surrogate_variables(self, surrogate_variables):
        """
        
        :param surrogate_variables: 
        :return: 
        """
