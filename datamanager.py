import copy
from datetime import timedelta

import pandas as pd
import numpy as np


class DataException(Exception):
    """Base class for all exceptions in the data module"""
    pass


class DataOriginError(DataException):
    """An error if origin information is inconsistent with the data at Data subclass initialization"""
    pass


class ConcurrentObservationError(DataException):
    """An error if concurrent observations exist for a single variable."""
    pass


class DataManager:
    """Base class for data subclasses.

    This class provides methods for data management subclasses.
    """

    def __init__(self, data, data_origin=None):
        """Initialize a Data object.

        data_origin must be a DataFrame that describes the origin of all columns in the data parameter. At least one
        row per variable. The column names of data_origin must be 'variable' and 'origin.' A brief example follows.

            data_origin example:

                 variable             origin
            0           Q   Q_ILR_WY2016.txt
            1           Q   Q_ILR_WY2017.txt
            2          GH   Q_ILR_WY2017.txt
            3   Turbidity   TurbILR.txt

        :param data: Pandas DataFrame with time DatetimeIndex index type.
        :type data: pd.DataFrame
        :param data_origin: Pandas DataFrame containing variable origin information.
        :type data_origin: pd.DataFrame
        """

        if data_origin is None:
            data_origin = self._create_empty_origin(data)

        self._check_origin(data, data_origin)

        self._data = data.copy(deep=True)
        self._data_origin = data_origin.copy(deep=True)

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v, in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result

    def _check_for_concurrent_obs(self, other):
        """Check other DataManager for concurrent observations of a variable. Raise ConcurrentObservationError if
        concurrent observations exist.

        :param other:
        :type other: DataManager
        :return:
        """

        # check for concurrent observations between self and other DataManager
        self_variables = self.get_variable_names()

        other_variables = other.get_variable_names()

        for variable in other_variables:

            if variable in self_variables:

                current_variable = self.get_variable(variable)
                new_variable = other.get_variable(variable)

                if np.any(new_variable.index.isin(current_variable.index)):

                    # raise exception if concurrent observations exist
                    raise ConcurrentObservationError("Concurrent observations exist for variable {}".format(variable))

    @staticmethod
    def _check_origin(data, origin):
        """

        :param origin:
        :return:
        """

        if not isinstance(origin, pd.DataFrame):
            raise TypeError("Origin must be type pandas.DataFrame")

        correct_origin_columns = {'variable', 'origin'}

        origin_columns_difference = correct_origin_columns.difference(origin.keys())

        if len(origin_columns_difference) != 0:
            raise DataOriginError("Origin DataFrame does not have the correct column names")

        variables_grouped = origin.groupby('variable')
        origin_variable_set = set(list(variables_grouped.groups))

        data_variable_set = set(list(data.keys()))

        if not (origin_variable_set.intersection(data_variable_set) == origin_variable_set.union(data_variable_set)):
            raise DataOriginError("Origin and data variables do not match")

    @staticmethod
    def _check_timestamp(value):
        """

        :param value:
        :return:
        """
        if not isinstance(value, pd.tslib.Timestamp):
            raise TypeError('Expected type pandas.tslib.Timestamp, received {}'.format(type(value)), value)

    def _check_variable_name(self, variable_name):
        """

        :param variable_name:
        :type variable_name: str
        :return:
        """

        if variable_name not in self.get_variable_names():
            raise ValueError('{} is not a valid variable name'.format(variable_name), variable_name)

    @classmethod
    def _create_empty_origin(cls, data):
        """
        
        :return: 
        """

        origin = cls.create_data_origin(data, np.NaN)

        return origin

    @staticmethod
    def _load_tab_delimited_data(file_path):
        """

        :param file_path:
        :return:
        """

        # Read TAB-delimited txt file into a DataFrame.
        tab_delimited_df = pd.read_table(file_path, sep='\t')

        # Check the formatting of the date/time columns. If one of the correct formats is used, reformat
        # those date/time columns into a new timestamp column. If none of the correct formats are used,
        # return an invalid file format error to the user.
        if 'y' and 'm' and 'd' and 'H' and 'M' and 'S' in tab_delimited_df.columns:
            tab_delimited_df.rename(columns={"y": "year", "m": "month", "d": "day"}, inplace=True)
            tab_delimited_df.rename(columns={"H": "hour", "M": "minute", "S": "second"}, inplace=True)
            tab_delimited_df["year"] = pd.to_datetime(tab_delimited_df[["year", "month", "day", "hour",
                                                                        "minute", "second"]], errors="coerce")
            tab_delimited_df.rename(columns={"year": "DateTime"}, inplace=True)
            tab_delimited_df.drop(["month", "day", "hour", "minute", "second"], axis=1, inplace=True)
        elif 'Date' and 'Time' in tab_delimited_df.columns:
            tab_delimited_df["Date"] = pd.to_datetime(tab_delimited_df["Date"] + " " + tab_delimited_df["Time"],
                                                      errors="coerce")
            tab_delimited_df.rename(columns={"Date": "DateTime"}, inplace=True)
            tab_delimited_df.drop(["Time"], axis=1, inplace=True)
        elif 'DateTime' in tab_delimited_df.columns:
            # tab_delimited_df.rename(columns={"DateTime": "Timestamp"}, inplace=True)
            tab_delimited_df["DateTime"] = pd.to_datetime(tab_delimited_df["DateTime"], errors="coerce")
        else:
            raise ValueError("Date and time information is incorrectly formatted.", file_path)

        tab_delimited_df.set_index("DateTime", drop=True, inplace=True)
        tab_delimited_df = tab_delimited_df.apply(pd.to_numeric, args=('coerce', ))

        return tab_delimited_df

    def add_data(self, other, keep_curr_obs=None):
        """Add data from other DataManager subclass.

        This method adds the data and data origin information from other DataManager objects. An exception will be
        raised if keep_curr_obs=None and concurrent observations exist for variables.

        :param other: Other DataManager object.
        :type other: DataManager
        :param keep_curr_obs: Indicate whether or not to keep the current observations.
        :type keep_curr_obs: {None, True, False}
        :return:
        """

        if keep_curr_obs is None:
            self._check_for_concurrent_obs(other)

        # get the DataFrames to combine
        new_df = other.get_data()
        old_df = self.get_data()

        # initialize an empty DataFrame containing all columns and indices
        new_columns = set(new_df.keys())
        old_columns = set(old_df.keys())
        columns = new_columns.union(old_columns)
        index = old_df.index.union(new_df.index)
        combined_df = pd.DataFrame(index=index, columns=columns)

        # combine the DataFrames
        for variable in columns:

            if variable in old_df.keys() and variable in new_df.keys():

                old_index = old_df[variable].index
                new_index = new_df[variable].index

                # fill the empty DataFrame with rows that are in the old DataFrame but not the new
                old_index_diff = old_index.difference(new_index)
                combined_df.ix[old_index_diff, variable] = old_df.ix[old_index_diff, variable]

                # fill the empty DataFrame with rows that are in the new DataFrame but not the old
                new_index_diff = new_index.difference(old_index)
                combined_df.ix[new_index_diff, variable] = new_df.ix[new_index_diff, variable]

                # handle the row intersection
                index_intersect = old_index.intersection(new_index)
                if keep_curr_obs:
                    combined_df.ix[index_intersect, variable] = old_df.ix[index_intersect, variable]
                else:
                    combined_df.ix[index_intersect, variable] = new_df.ix[index_intersect, variable]

            elif variable in old_df.keys():

                combined_df.ix[old_df.index, variable] = old_df[variable]

            elif variable in new_df.keys():

                combined_df.ix[new_df.index, variable] = new_df[variable]

        combined_df = combined_df.apply(pd.to_numeric, args=('ignore', ))

        data_origin = self._data_origin.copy(deep=True)
        combined_data_origin = data_origin.append(other._data_origin)
        combined_data_origin.drop_duplicates(inplace=True)
        combined_data_origin.reset_index(drop=True, inplace=True)

        return type(self)(combined_df, combined_data_origin)

    @staticmethod
    def create_data_origin(data_df, data_path):
        """

        :param data_df: 
        :param data_path: 
        :return: 
        """

        acoustic_variables = list(data_df)
        data = [[variable, data_path] for variable in acoustic_variables]
        data_origin = pd.DataFrame(data=data, columns=['variable', 'origin'])
        return data_origin

    def drop_variables(self, variable_names):
        """

        :param variable_names: list-like parameter containing names of variables to drop
        :return:
        """

        # drop the columns containing the variables
        data = self._data.copy(deep=True)
        data.drop(variable_names, axis=1, errors='ignore', inplace=True)

        data_origin = self._data_origin.copy(deep=True)

        # drop the variable origin information
        for variable in variable_names:

            variable_row = data_origin['variable'] == variable
            data_origin = data_origin[~variable_row]

        return type(self)(data, data_origin)

    def get_data(self):
        """Return a copy of the time series data contained within the manager.

        :return: Copy of the data being managed in a DataFrame
        """
        return self._data.copy(deep=True)

    def get_origin(self):
        """Return a DataFrame containing the variable origins.

        :return:
        """

        return self._data_origin.copy(deep=True)

    def get_variable(self, variable_name):
        """Return the time series of the valid observations of the variable described by variable_name.

        Any NaN observations will not be returned.

        :param variable_name: Name of variable to return time series
        :type variable_name: str
        :return:
        """

        self._check_variable_name(variable_name)

        # return pd.DataFrame(self._data.ix[:, variable_name]).dropna()

        return pd.DataFrame(self._data.ix[:, variable_name])

    def get_variable_names(self):
        """Return a list of variable names.

        :return: List of variable names.
        """
        return list(self._data.keys())

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
            try:
                variable_observation = self._data.ix[time, variable_name]
            except KeyError as err:
                if err.args[0] == time:
                    variable_observation = None
                else:
                    raise err

        else:

            variable = self.get_variable(variable_name)

            # get the subset of times with the variable
            time_diff = timedelta(minutes=time_window_width / 2.)

            # match the nearest-in-time observation
            if match_method == 'nearest':
                nearest_index = variable.index.get_loc(time, method='nearest', tolerance=time_diff)
                nearest_observation = variable.ix[nearest_index]
                variable_observation = nearest_observation.as_matrix()[0]

            # get the mean observation
            elif match_method == 'mean':
                beginning_time = time - time_diff
                ending_time = time + time_diff
                time_window = (beginning_time < variable.index) & (variable.index <= ending_time)
                variable_near_time = variable.ix[time_window]
                variable_observation = variable_near_time.mean()

            else:
                msg = 'Unrecognized keyword value for match_method: {}'.format(match_method)
                raise ValueError(msg)

        return variable_observation

    def get_variable_origin(self, variable_name):
        """Get a list of the origin(s) for the given variable name.

        :param variable_name: Name of variable
        :type variable_name: str
        :return: List containing the origin(s) of the given variable.
        :return type: list
        """

        self._check_variable_name(variable_name)

        grouped = self._data_origin.groupby('variable')
        variable_group = grouped.get_group(variable_name)
        variable_origin = list(variable_group['origin'])

        return variable_origin

    def match_data(self, surrogate_data, variable_name=None, time_window_width=0, match_method='nearest'):
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

    @classmethod
    def read_tab_delimited_data(cls, file_path):
        """Read a tab-delimited file containing a time series and return a DataManager instance.

        :param file_path: File path containing the TAB-delimited ASCII data file
        :param params: None.
        :return: DataManager object containing the data information
        """

        tab_delimited_df = cls._load_tab_delimited_data(file_path)

        origin = []

        for variable in tab_delimited_df.keys():
            origin.append([variable, file_path])

        data_origin = pd.DataFrame(data=origin, columns=['variable', 'origin'])

        return cls(tab_delimited_df, data_origin)
