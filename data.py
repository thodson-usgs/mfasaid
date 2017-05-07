from datetime import timedelta
from numbers import Number

import pandas as pd
import numpy as np

"""Module containing classes to handle data within the SAID library.


Exceptions:

DataException: Base class for exceptions within this module.

ADVMDataIncompatibleError: An error if ADVMData instance are incompatible

DataOriginError: An error if origin information is inconsistent with the data at Data subclass initialization

ConcurrentObservationError: An error if concurrent observations exist for a single variable.


Classes:

DataManager: Base class for data subclasses.

ConstituentData: Base class to manage constituent data.

SurrogateData: Data manager class for surrogate data.

"""


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
        :param data_path: Pandas DataFrame containing variable origin information.
        :type data_path: pd.DataFrame
        """

        if data_origin is None:
            data_origin = self._create_empty_origin(data)

        self._check_origin(data, data_origin)

        self._data = data.copy(deep=True)
        self._data_origin = data_origin.copy(deep=True)

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

    @staticmethod
    def _create_empty_origin(data):
        """
        
        :return: 
        """
        variables = list(data.keys())

        data = []

        for var in variables:
            data.append([var, np.nan])

        origin_columns = ['variable', 'origin']

        origin = pd.DataFrame(data=data, columns=origin_columns)

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

        self._data = combined_df.apply(pd.to_numeric, args=('coerce', ))

        # combine the variable origin data
        self._data_origin = self._data_origin.append(other._data_origin)
        self._data_origin.drop_duplicates(inplace=True)
        self._data_origin.reset_index(drop=True, inplace=True)

    def drop_variable(self, variable_names):
        """

        :param variable_names: list-like parameter containing names of variables to drop
        :return:
        """

        # drop the columns containing the variables
        self._data.drop(variable_names, axis=1, errors='ignore', inplace=True)

        # drop the variable origin information
        for variable in variable_names:

            variable_row = self._data_origin['variable'] == variable
            self._data_origin = self._data_origin[~variable_row]

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

    def get_variable_observation(self, variable_name, time):
        """Return the value for the observation value of the given variable at the given time.

        :param variable_name: Name of variable.
        :type variable_name: str
        :param time: Time
        :type time: pandas.tslib.Timestamp
        :return: variable_observation
        :return type: numpy.float64
        """

        self._check_variable_name(variable_name)

        try:
            variable_observation = self._data.ix[time, variable_name]
        except KeyError as err:
            if err.args[0] == time:
                variable_observation = None
            else:
                raise err

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


class ConstituentData(DataManager):
    """Data manager class for constituent data"""
    # TODO: Make ConstituentData immutable
    def __init__(self, data, data_origin, surrogate_data=None):
        """

        :param data:
        :param data_origin:
        """
        super().__init__(data, data_origin)

        self._constituent_data = data

        self._surrogate_variable_avg_window = {}
        self._surrogate_variable_match_method = {}
        self._surrogate_max_abs_time_diff = {}
        self._surrogate_data = None

        if surrogate_data:
            self.add_surrogate_data(surrogate_data)

    def _check_surrogate_variable_name(self, variable_name):
        """

        :param variable_name:
        :return:
        """

        if variable_name not in self._surrogate_data.get_variable_names():
            raise ValueError("{} is an invalid surrogate variable name".format(variable_name))

    def add_surrogate_data(self, surrogate_data, keep_curr_obs=None):
        """Add surrogate data observations.

        :param surrogate_data: Surrogate data manager
        :type surrogate_data: SurrogateData
        :param keep_curr_obs:
        :return:
        """

        # surrogate_data must be a subclass of SurrogateData
        if not isinstance(surrogate_data, SurrogateData):
            raise TypeError("surrogate_data must be a subclass of data.SurrogateData")

        # add the surrogate data
        if self._surrogate_data:
            self._surrogate_data.add_data(surrogate_data, keep_curr_obs)
        else:
            self._surrogate_data = surrogate_data

        # remove variables from the surrogate data set that are in the constituent data set
        self._surrogate_data.drop_variable(self._constituent_data.keys())

        # update the surrogate data averaging and max windows
        for variable in self._surrogate_data.get_variable_names():
            if variable not in self._surrogate_variable_avg_window.keys():
                self._surrogate_variable_match_method[variable] = 'average'
                self._surrogate_variable_avg_window[variable] = 0
                self._surrogate_max_abs_time_diff[variable] = 0

        # update the dataset
        self.update_data()

    def get_surrogate_avg_window(self, surrogate_variable_name):
        """

        :param surrogate_variable_name:
        :return:
        """

        return self._surrogate_variable_avg_window[surrogate_variable_name]

    def get_surrogate_data_manager(self):
        """Returns the surrogate data manager instance.

        :return:
        """

        return self._surrogate_data

    def get_surrogate_match_method(self, surrogate_variable_name):
        """

        :param surrogate_variable_name:
        :return:
        """

        self._check_surrogate_variable_name(surrogate_variable_name)

        return self._surrogate_variable_match_method[surrogate_variable_name]

    def get_surrogate_max_abs_time_diff(self, surrogate_variable_name):
        """

        :param surrogate_variable_name:
        :return:
        """

        self._check_surrogate_variable_name(surrogate_variable_name)

        return self._surrogate_max_abs_time_diff[surrogate_variable_name]

    def set_surrogate_avg_window(self, surrogate_variable_name, avg_window):
        """Set the surrogate variable averaging window.

        :param surrogate_variable_name:
        :param avg_window:
        :return:
        """

        if not isinstance(avg_window, Number):
            raise TypeError("avg_window must be type Number")

        self._check_surrogate_variable_name(surrogate_variable_name)
        self._surrogate_variable_avg_window[surrogate_variable_name] = avg_window
        self.update_data()

    def set_surrogate_match_method(self, surrogate_variable_name, method):
        """

        :param surrogate_variable_name:
        :param method: 'average' or 'closest'
        :return:
        """

        if method != 'average' and method != 'closest':
            raise ValueError('method must be average or closest')

        self._check_surrogate_variable_name(surrogate_variable_name)
        self._surrogate_variable_match_method[surrogate_variable_name] = method
        self.update_data()

    def set_surrogate_max_abs_time_diff(self, surrogate_variable_name, time_window):
        """

        :param surrogate_variable_name:
        :param time_window:
        :return:
        """

        if not isinstance(time_window, Number):
            raise TypeError("time_window must be type Number")

        self._check_surrogate_variable_name(surrogate_variable_name)
        self._surrogate_max_abs_time_diff[surrogate_variable_name] = time_window
        self.update_data()

    def update_data(self):
        """Update the data set.

        Call when changes to the surrogate data set have been made.

        :return: None
        """

        if self._surrogate_data:

            # initialize data for a DataManager
            matched_surrogate_data = \
                pd.DataFrame(index=self._data.index, columns=self._surrogate_data.get_variable_names())
            surrogate_variable_origin_data = []

            # iterate over all variables
            for variable in self._surrogate_data.get_variable_names():

                match_method = self._surrogate_variable_match_method[variable]

                # iterate over all times in constituent data set
                for index, _ in matched_surrogate_data.iterrows():

                    # average matching
                    if match_method == 'average':
                        avg_window = self._surrogate_variable_avg_window[variable]
                        surrogate_value = self._surrogate_data.get_avg_variable_observation(variable, index, avg_window)

                    # closest-in-time matching
                    else:

                        # get the closest-in-time surrogate observation
                        closest_surrogate_obs = \
                            self._surrogate_data.get_closest_variable_observation(variable, index)
                        closest_surrogate_obs = closest_surrogate_obs.ix[0]
                        max_time = timedelta(minutes=self._surrogate_max_abs_time_diff[variable])

                        # match the surrogate observation if the time is within the window
                        abs_time_diff = np.abs(closest_surrogate_obs.name - index)
                        if abs_time_diff < max_time:
                            surrogate_value = closest_surrogate_obs.as_matrix()[0]
                        else:
                            surrogate_value = np.NaN

                    matched_surrogate_data.ix[index, variable] = surrogate_value

                for origin in self._surrogate_data.get_variable_origin(variable):
                    surrogate_variable_origin_data.append([variable, origin])

            # create origin DataFrame
            surrogate_variable_origin = \
                pd.DataFrame(data=surrogate_variable_origin_data, columns=['variable', 'origin'])

            # create data manager for new data
            data_manager = DataManager(matched_surrogate_data, surrogate_variable_origin)

            self.add_data(data_manager, keep_curr_obs=False)


class SurrogateData(DataManager):
    """Data manager class for surrogate data"""

    def __init__(self, data, data_origin):
        """

        :param data:
        :param data_origin:
        """

        super().__init__(data, data_origin)

    def get_avg_variable_observation(self, variable_name, time, avg_window):
        """For a given variable, get an average value from observations around a given time within a given window.

        :param variable_name: Variable name
        :type variable_name: str
        :param time: Center of averaging window
        :type time: pandas.tslib.Timestamp
        :param avg_window: Width of half of averaging window, in minutes
        :type avg_window: Number
        :return: Averaged value
        :return type: numpy.float
        """

        self._check_variable_name(variable_name)

        variable = self.get_variable(variable_name)

        time_diff = timedelta(minutes=avg_window)

        beginning_time = time - time_diff
        ending_time = time + time_diff

        time_window = (beginning_time < variable.index) & (variable.index <= ending_time)

        variable_observation = np.float(variable.ix[time_window].mean())

        return variable_observation

    def get_closest_variable_observation(self, variable_name, time):
        """

        :param variable_name:
        :param time:
        :return:
        """

        self._check_variable_name(variable_name)

        variable = self.get_variable(variable_name)

        absolute_time_difference = np.abs(variable.index - time)

        min_abs_time_diff_index = absolute_time_difference.min() == absolute_time_difference

        return variable.ix[min_abs_time_diff_index]
