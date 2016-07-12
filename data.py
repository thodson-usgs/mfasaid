import re

import pandas as pd
import numpy as np
import copy


class ADVMProcParam:
    """
    Stores ADVM Processing parameters.

    """

    def __init__(self, num_cells):
        """

        :param num_cells: Number of cells reported by the ADVM configuration parameters

        Note: The number of cells is required because it is used for a boundary check
            when setting the 'Minimum Number of Cells' value.
        """

        self._number_of_cells = num_cells
        self._proc_dict = {
            "Beam": 2,
            "Moving Average Span": 1,
            "Backscatter Values": "SNR",
            "Minimum Cell Mid-Point Distance": -np.inf,
            "Maximum Cell Mid-Point Distance": np.inf,
            "Minimum Number of Cells": 1,
            "Minimum Vbeam": -np.inf,
            "Near Field Correction": True,
            "WCB Profile Adjustment": True
        }

    def __getitem__(self, key):
        """
        Return the requested processing parameter value.

        :param key: Processing parameter key
        :return: Processing parameter corresponding to the given key
        """

        self._check_proc_key(key)
        return self._proc_dict[key]

    def __setitem__(self, key, value):
        """
        Store the requested processing parameter value.

        :param key: Processing parameter key
        :param value: Processing value to be stored
        :return: Nothing
        """

        self._check_proc_key(key)
        self._check_proc_value(key, value)
        self._proc_dict[key] = value
        return

    def __delitem__(self, key):
        """
        Delete the requested processing parameter.

        :param key: Processing parameter key
        :return: Nothing
        """

        self._proc_dict.pop(key, None)
        return

    def _check_proc_key(self, key):
        """
        Check if the provided key exists in the proc_dict. Raise an exception if not.

        :param key: User-provided processing dictionary key
        :return: Nothing
        """

        if key not in self._proc_dict.keys():
            raise KeyError(key)
        return

    def _check_proc_value(self, key, value):
        """
        Check if the value provided is valid. Raise an exception if not.

        :param key: User-provided processing dictionary key
        :param value: User-provided processing dictionary value
        :return: Nothing
        """

        if key == "Beam" and (value in range(1, 3) or value == 'Avg'):
            return
        elif key == "Moving Average Span" and (0 <= value <= 1):
            return
        elif key == "Backscatter Values" and (value == "SNR" or value == "Amp"):
            return
        elif key == "Minimum Cell Mid-Point Distance":
            np.float(value)
            return
        elif key == "Maximum Cell Mid-Point Distance":
            np.float(value)
            return
        elif key == "Minimum Number of Cells" and (1 <= value <= self._number_of_cells):
            return
        elif key == "Minimum Vbeam":
            np.float(value)
            return
        elif key == "Near Field Correction" and isinstance(value, bool):
            return
        elif key == "WCB Profile Adjustment" and isinstance(value, bool):
            return
        else:
            raise ValueError(value)


class ADVMConfigParam:
    """
    Stores ADVM Configuration parameters.

    """

    def __init__(self, config_dict):
        """

        :param config_dict: Dictionary containing ADVM configuration data required to process ADVM data.
        """

        self._config_dict = copy.deepcopy(config_dict)

    def __getitem__(self, key):
        """
        Return the requested configuration parameter value.

        :param key: Configuration parameter key
        :return: Configuration parameter corresponding to the given key
        """

        self._check_config_key(key)
        return self._config_dict[key]

    def __setitem__(self, key, value):
        """
        Store the requested configuration parameter value.

        :param key: Configuration parameter key
        :param value: Configuration value to be stored
        :return: Nothing
        """

        self._check_config_key(key)
        self._config_dict[key] = value
        return

    def __delitem__(self, key):
        """
        Delete the requested configuration parameter.

        :param key: Configuration parameter key
        :return: Nothing
        """

        self._config_dict.pop(key, None)
        return

    def _check_config_key(self, key):
        """
        Check if the provided key exists in the config_dict. Raise an exception if not.

        :param key: Configuration dictionary key
        :return: Nothing
        """

        if key not in self._config_dict.keys():
            raise KeyError(key)
        return


class ADVMData:
    """
    Stores ADVM data and parameters.

    """

    def __init__(self, config_dict, acoustic_df):
        """
        Initializes ADVMData instance. Creates default processing configuration data attribute.

        :param config_dict: Dictionary containing ADVM configuration data required to process ADVM data
        :param acoustic_df: DataFrame containing acoustic data
        """

        self._acoustic_df = copy.deepcopy(acoustic_df)
        self._config_dict = ADVMConfigParam(config_dict)
        self._proc_dict = ADVMProcParam(config_dict["Number of Cells"])

    def set_proc_params(self, proc_params):
        """
        Sets the processing parameters based on user input.

        :param proc_params: Dictionary containing configuration parameters
        :return: Nothing
        """

        for key in proc_params.keys():
            self._proc_dict[key] = proc_params[key]

        return

    def get_config_params(self, config_params):
        """
        Find and return configuration parameters. Throw ValueError if config_params contains invalid key.

        :param config_params: List containing configuration parameters (keys)
        :return: Dictionary containing configuration parameters
        """

        requested_params = {}

        # Set return values for all matching keys.
        for key in config_params:
            requested_params[key] = self._config_dict[key]

        return requested_params

    def get_proc_params(self, proc_params):
        """
        Find and return processing parameters. Throws ValueError if proc_params contains invalid key.

        :param proc_params: List containing processing parameters (keys)
        :return: Dictionary containing processing parameters
        """

        requested_params = {}

        # Set return values for all matching keys.
        for key in proc_params:
            requested_params[key] = self._proc_dict[key]

        return requested_params

    def get_meanSCB(self):
        """
        Return mean sediment corrected backscatter. Throw exception if all required variables have not been provided.

        :return: Mean sediment corrected backscatter for all observations contained in acoustic_df
        """

        return pd.Series(index=self._acoustic_df.index.values)


    def get_cell_range(self):
        """
        Calculate range of cells along a single beam.

        :return: Range of cells along a single beam
        """

        return pd.Series(index=self._acoustic_df.index.values)

    def get_MB(self):
        """Return measured backscatter"""

        return

    def get_SAC(self):
        """
        Calculate sediment attenuation coefficient. Throw exception if all required variables have not been provided.

        :return: Sediment attenuation coefficient for all observations in acoustic_df
        """

        return pd.Series(index=self._acoustic_df.index.values)

    def get_SCB(self):
        """
        Calculate sediment corrected backscatter. Throw exception if all required variables have not been provided.

        :return: Sediment corrected backscatter for all cells in the acoustic time series
        """

        return pd.Series(index=self._acoustic_df.index.values)

    def get_WCB(self):
        """
        Calculate water corrected backscatter. Throw exception if all required variables have not been provided

        :return: Water corrected backscatter for all cells in the acoustic time series
        """

        return pd.Series(index=self._acoustic_df.index.values)

    def merge(self, other):
        """
        Merges self and other ADVMData objects. Throws exception if other ADVMData object is incompatible with self.

        :param other: ADVMData object to be merged with self
        :return: Merged ADVMData object
        """

        return pd.Series(index=self._acoustic_df.index.values)


class RawAcousticDataContainer:
    """Container for raw acoustic data type. The unique
    identifier for an acoustic data type object is the frequency
    of the instrument.
    """

    def __init__(self):
        """Initialize AcousticDataContainer object."""

        # initialize _acoustic_data as empty dictionary
        self._acoustic_data = {}

    def add_data(self, new_advm_data, **kwargs):
        """ Add acoustic data to container."""

        # get the frequency of the ADVM data set
        frequency = new_advm_data.get_config_params()['Frequency']

        # if a data set with the frequency is already loaded,
        if frequency in self._acoustic_data.keys():
            self._acoustic_data[frequency].merge(new_advm_data, **kwargs)

        else:
            self._acoustic_data[frequency] = new_advm_data

    def get_cell_range(self, frequency):
        """Return the cell range for the acoustic data set described
        by frequency.
        """

        # return the cell range
        return self._acoustic_data[frequency].get_cell_range()

    def get_config_params(self, frequency, config_params):
        """Return the configuration parameters of an acoustic data
        structure.
        """

        # return the configuration parameters described by config_params
        return self._acoustic_data[frequency].get_config_params(config_params)

    def get_freqs(self):
        """Return the acoustic frequencies of the contained data."""

        return self._acoustic_data.keys()

    def get_meanSCB(self, frequencies='All'):
        """Return the mean sediment corrected backscatter (SCB)
        time series.
        """

        # get all contained frequencies if requested
        if frequencies == 'All':
            frequencies = self.get_freqs()

        # create empty DataFrame
        meanSCB = pd.DataFrame()

        # assert that frequencies is a list
        assert (isinstance(frequencies, list))

        for freq in frequencies:
            advm_data = self._acoustic_data[freq]
            tmp_meanSCB = advm_data.get_meanSCB()
            tmp_meanSCB.columns = ['meanSCB' + freq]
            meanSCB = meanSCB.join(tmp_meanSCB, how='outer', sort=True)

        return meanSCB

    def get_proc_params(self, frequency):
        """Return the processing parameters for for the acoustic data
        set described by frequency.
        """

        return self._acoustic_data[frequency].get_proc_params()

    def get_SAC(self, frequencies):
        """Return the sediment attenuation coefficient (SAC) time series."""

        # get all contained frequencies if requested
        if frequencies == 'All':
            frequencies = self.get_freqs()

        # create empty DataFrame
        sac = pd.DataFrame()

        # assert that frequencies is a list
        assert (isinstance(frequencies, list))

        for freq in frequencies:
            advm_data = self._acoustic_data[freq]
            tmp_sac = advm_data.get_SAC()
            tmp_sac.columns = ['meanSCB' + freq]
            sac = sac.join(tmp_sac, how='outer', sort=True)

        return sac

    def get_SCB(self, frequency):
        """Return the sediment corrected backscatter (SCB) time series."""
        return self._acoustic_data[frequency].get_SCB()

    def get_WCB(self, frequency):
        """Return the water corrected backscatter (WCB) time series."""
        return self._acoustic_data[frequency].get_WCB()


class RawAcousticDataConverter:
    """An object to convert data contained within a pandas.DataFrame object to an ADVMData type.

    """

    def __init__(self):

        self._conf_params = ADVMConfParams()

    def get_conf_params(self):
        """Returns ADVM configuration parameters"""

        return self._conf_params.get_dict()

    def set_conf_params(self, advm_conf_params):
        """Sets the configuration parameters that are used to generate
        ADVMData objects.
        """

        self._conf_params.update(advm_conf_params)

    def convert_raw_data(self, advm_df):
        """Convert raw acoustic data to ADVM data type"""

        # get the columns of the data frame
        df_cols = list(advm_df)

        # the pattern of the columns to search for
        col_pattern = r'(Temp|Vbeam|(^Cell\d{2}(Amp|SNR)\d{1}))$'

        # create an empty list to hold raw data column names
        raw_data_cols = []

        # for each column in the column names
        for col in df_cols:

            # search for the regular expression
            m = re.search(col_pattern, col)

            # if the search string was found
            if m is not None:

                # append the column name to the raw data columns
                raw_data_cols.append(m.group())

        # if the column list contains column names
        if len(raw_data_cols) > 0:

            # get a data frame containing the columns of the raw
            # acoustic data
            raw_data_df = advm_df.loc[:, raw_data_cols]

            # drop the columns from the input data frame
            raw_data_df.drop(raw_data_cols, inplace=True)

            # create and return an ADVMData instance
            advm_data = ADVMData(self._conf_params, raw_data_df)
            return advm_data

        # otherwise, return None
        else:
            return None


class SurrogateDataContainer:
    """Container for surrogate data. """

    def __init__(self):
        self._surrogate_data = pd.DataFrame()

    def add_data(self, new_surrogate_data):

        # merge the old data with the new data
        # self._surrogate_data.merge(surrogate_data, inplace=True)
        concated = pd.concat([self._surrogate_data, new_surrogate_data])

        # get the grouped data frame
        grouped = concated.groupby(level=0)

        # drop the new rows
        self._surrogate_data = grouped.last()

        # sort the undated data frame
        self._surrogate_data.sort(inplace=True)

    def get_surrogate_sample(self, surrogate_name, time, time_delta):
        """Return the surrogate samples within a 2 X time_delta time window
        centered on time
        """

        # assert proper types
        assert (isinstance(surrogate_name, str))
        assert (isinstance(time, pd.Timestamp))
        assert (isinstance(time_delta, pd.Timedelta))

        # get the beginning and ending time of the time period requested
        begin_time_period = time - time_delta
        end_time_period = time + time_delta

        # get a series instance of the surrogate data
        surrogate_series = self._surrogate_data[surrogate_name]

        # get an index of the times within the given window
        surrogate_sample_index = begin_time_period \
            <= surrogate_series.index <= end_time_period

        # get a series of the samples within the window
        surrogate_samples = surrogate_series[surrogate_sample_index]

        # return the samples
        return surrogate_samples

    def get_mean_surrogate_sample(self, surrogate_name, time, time_delta):
        """Return the mean of the surrogate samples within a
        2 X time_delta time window centered on time
        """

        # get the surrogate samples within the requested window
        surrogate_samples = self.get_surrogate_sample(surrogate_name,
                                                      time, time_delta)

        # get a mean of the samples
        surrogate_mean = surrogate_samples.mean()

        # create a Series instance with the mean of the sample
        # at the given time
        mean_surrogate_sample = pd.Series(data=surrogate_mean,
                                          index=[time], name=surrogate_name)

        # return the mean surrogate series
        return mean_surrogate_sample
