import re
import copy
import abc

import pandas as pd
import numpy as np


class ADVMParam(abc.ABC):
    """Base class for ADVM parameter classes"""

    @abc.abstractmethod
    def __init__(self, param_dict):
        self._dict = param_dict

    def __deepcopy__(self, memo):
        """Provide method for copy.deepcopy()"""

        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v, in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result

    def __getitem__(self, key):
        """Return the requested parameter value.

        :param key: Parameter key
        :return: Parameter corresponding to the given key
        """
        return self._dict[key]

    def __setitem__(self, key, value):
        """Set the requested parameter value.
        :param key: Parameter key
        :param value: Value to be stored
        :return: Nothing
        """

        self._check_value(key, value)
        self._dict[key] = value

    @abc.abstractmethod
    def _check_value(self, key, value):
        pass

    def _check_key(self, key):
        """Check if the provided key exists in the _dict. Raise an exception if not.

        :param key: Parameter key to check
        :return: Nothing
        """

        if key not in self._dict.keys():
            raise KeyError(key)
        return

    def get_dict(self):
        """Return a dictionary containing the processing parameters.

        :return: Dictionary containing the processing parameters
        """

        return copy.deepcopy(self._dict)

    def items(self):
        """Return a set-like object providing a view on the contained parameters.

        :return: Set-like object providing a view on the contained parameters
        """

        return self._dict.items()

    def keys(self):
        """Return the parameter keys.

        :return: A set-like object providing a view on the parameter keys.
        """

        return self._dict.keys()

    def update(self, update_values):
        """Update the  parameters.

        :param update_values: Item containing key/value processing parameters
        return: Nothing
        """

        for key, value in update_values.items():
            self._check_value(key, value)
            self._dict[key] = value


class ADVMProcParam(ADVMParam):
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
        proc_dict = {
            "Beam": 2,
            # "Moving Average Span": 1,
            "Backscatter Values": "SNR",
            "Minimum Cell Mid-Point Distance": -np.inf,
            "Maximum Cell Mid-Point Distance": np.inf,
            "Minimum Number of Cells": 2,
            "Minimum Vbeam": -np.inf,
            "Near Field Correction": True,
            "WCB Profile Adjustment": True
        }

        super().__init__(proc_dict)

    def _check_value(self, key, value):
        """
        Check if the value provided is valid. Raise an exception if not.

        :param key: User-provided processing dictionary key
        :param value: User-provided processing dictionary value
        :return: Nothing
        """

        self._check_key(key)

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


class ADVMConfigParam(ADVMParam):
    """
    Stores ADVM Configuration parameters.

    """

    def __init__(self):
        """
        """

        # the valid for accessing information in the configuration parameters
        valid_keys = ['Frequency', 'Effective Transducer Diameter', 'Beam Orientation', 'Slant Angle',
                            'Blanking Distance', 'Cell Size', 'Number of Cells']

        # initial values for the configuration parameters
        init_values = np.tile(np.nan, (len(valid_keys),))

        config_dict = dict(zip(valid_keys, init_values))

        super().__init__(config_dict)

    def _check_value(self, key, value):
        """Check if the provided value is valid for the given key. Raise an exception if not.

        :param key: Keyword for configuration item
        :param value: Value for corresponding key
        :return: Nothing
        """

        self._check_key(key)

        other_keys = ['Frequency', 'Effective Transducer Diameter',  'Slant Angle', 'Blanking Distance', 'Cell Size']

        if key == "Beam Orientation" and (value == "Horizontal" or value == "Vertical"):
            return
        elif key == "Number of Cells" and (1 <= value and isinstance(value, int)):
            return
        elif key in other_keys and 0 <= value and isinstance(value, (int, float)):
            return
        else:
            raise ValueError(value)


class ADVMData:
    """
    Stores ADVM data and parameters.

    """

    def __init__(self, config_params, acoustic_df):
        """
        Initializes ADVMData instance. Creates default processing configuration data attribute.

        :param config_params: ADVM configuration data required to process ADVM data
        :param acoustic_df: DataFrame containing acoustic data
        """

        # self._acoustic_df = copy.deepcopy(acoustic_df)
        self._acoustic_df = acoustic_df.copy(deep=True)
        self._config_param = ADVMConfigParam()
        self._config_param.update(config_params)
        self._proc_param = ADVMProcParam(self._config_param["Number of Cells"])

    def set_proc_params(self, proc_params):
        """
        Sets the processing parameters based on user input.

        :param proc_params: Dictionary containing configuration parameters
        :return: Nothing
        """

        for key in proc_params.keys():
            self._proc_param[key] = proc_params[key]

        return

    def get_config_params(self):
        """
        Return dictionary containing configuration parameters.

        :return: Dictionary containing configuration parameters
        """

        return self._config_param.get_dict()

    def get_proc_params(self):
        """
        Return dictionary containing processing parameters.

        :return: Dictionary containing processing parameters
        """

        return self._proc_param.get_dict()

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

    def get_config_params(self, frequency):
        """Return the configuration parameters of an acoustic data
        structure.
        """

        # return the configuration parameters described by config_params
        return self._acoustic_data[frequency].get_config_params()

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

        self._conf_params = ADVMConfigParam()

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
