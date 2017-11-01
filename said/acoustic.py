import abc
import copy
from datetime import timedelta
import linecache
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from linearmodel import datamanager
import said.surrogatemodel as surrogatemodel
from said.plotting import LineStyleGenerator


class AcousticException(Exception):
    """Base class for all exceptions in the acoustic module"""
    pass


class ADVMDataIncompatibleError(AcousticException):
    """An error if ADVMData instances are incompatible"""
    pass


class ADVMParam(abc.ABC):
    """Base class for ADVM parameter classes"""

    @abc.abstractmethod
    def __init__(self, param_dict):
        self._dict = param_dict

    def __deepcopy__(self, memo):
        """
        Provide method for copy.deepcopy().

        :param memo:
        :return:
        """

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

    def __repr__(self):
        return self._dict.__repr__()

    def __setitem__(self, key, value):
        """Set the requested parameter value.

        :param key: Parameter key
        :param value: Value to be stored
        :return: Nothing
        """

        self._check_value(key, value)
        self._dict[key] = value

    def __str__(self):
        return self._dict.__str__()

    @abc.abstractmethod
    def _check_value(self, key, value):
        raise NotImplementedError

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
        :return: Nothing
        """

        for key, value in update_values.items():
            self._check_value(key, value)
            self._dict[key] = value


class ADVMProcParam(ADVMParam):
    """Stores ADVM Processing parameters."""

    def __init__(self, num_cells):
        """

        :param num_cells: Number of cells reported by the ADVM configuration parameters

        Note: The number of cells is required because it is used for a boundary check
            when setting the 'Minimum Number of Cells' value.
        """

        self._number_of_cells = num_cells
        proc_dict = {
            "Beam": 1,
            # "Moving Average Span": 1,
            "Backscatter Values": "SNR",
            "Intensity Scale Factor": 0.43,
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
        elif key == "Intensity Scale Factor" and 0 < value:
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
    """Stores ADVM Configuration parameters."""

    def __init__(self):
        """
        """

        # the valid for accessing information in the configuration parameters
        valid_keys = ['Frequency', 'Effective Transducer Diameter', 'Beam Orientation', 'Slant Angle',
                      'Blanking Distance', 'Cell Size', 'Number of Cells', 'Number of Beams']

        # initial values for the configuration parameters
        init_values = np.tile(np.nan, (len(valid_keys),))

        config_dict = dict(zip(valid_keys, init_values))

        super().__init__(config_dict)

    def is_compatible(self, other):
        """Checks compatibility of ADVMConfigParam instances

        :param other: Other instance of ADVMConfigParam
        :return:
        """

        keys_to_check = ['Frequency', 'Slant Angle', 'Blanking Distance', 'Cell Size', 'Number of Cells']

        compatible_configs = True

        for key in keys_to_check:
            if not self[key] == other[key]:
                compatible_configs = False
                break

        return compatible_configs

    def _check_value(self, key, value):
        """Check if the provided value is valid for the given key. Raise an exception if not.

        :param key: Keyword for configuration item
        :param value: Value for corresponding key
        :return: Nothing
        """

        self._check_key(key)

        other_keys = ['Frequency', 'Effective Transducer Diameter', 'Slant Angle', 'Blanking Distance', 'Cell Size']

        if key == "Beam Orientation" and (value == "Horizontal" or value == "Vertical"):
            return
        elif key == "Number of Cells" and (1 <= value and isinstance(value, int)):
            return
        elif key == "Number of Beams" and (0 <= value and isinstance(value, int)):
            return
        elif key in other_keys and 0 <= value and isinstance(value, (int, float)):
            return
        else:
            raise ValueError(value, key)


class ADVMData:

    @abc.abstractmethod
    def __init__(self):

        self._data_manager = None
        self._configuration_parameters = None

    def __deepcopy__(self, memo):
        """
        Provide method for copy.deepcopy().

        :param memo:
        :return:
        """

        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v, in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result

    @staticmethod
    def _calc_speed_of_sound(temperature):
        """Calculate the speed of sound in water (in meters per second) based on Marczak, 1997

        :param temperature: Array of temperature values, in degrees Celsius
        :return: speed_of_sound: Speed of sound in meters per second
        """

        speed_of_sound = 1.402385 * 10 ** 3 + 5.038813 * temperature - \
                         (5.799136 * 10 ** -2) * temperature ** 2 + \
                         (3.287156 * 10 ** -4) * temperature ** 3 - \
                         (1.398845 * 10 ** -6) * temperature ** 4 + \
                         (2.787860 * 10 ** -9) * temperature ** 5

        return speed_of_sound

    @staticmethod
    def _calc_wavelength(speed_of_sound, frequency):
        """Calculate the wavelength of an acoustic signal.

        :param speed_of_sound: Array containing the speed of sound in meters per second
        :param frequency: Scalar containing acoustic frequency in kHz
        :return:
        """

        wavelength = speed_of_sound / (frequency * 1e3)

        return wavelength

    def _create_origin_from_data_frame(self, acoustic_df):

        data_origin = self._data_manager.get_origin()
        data_sources = set(data_origin['origin'])

        new_data_origin = pd.DataFrame(columns=['variable', 'origin'])

        for source in data_sources:
            tmp_origin = datamanager.DataManager.create_data_origin(acoustic_df, source)
            new_data_origin = new_data_origin.append(tmp_origin)

        new_data_origin.reset_index(drop=True, inplace=True)

        return new_data_origin

    def get_configuration_parameters(self):
        """

        :return: 
        """

        return copy.deepcopy(self._configuration_parameters)

    def get_data(self):
        """

        :return: 
        """

        return self._data_manager.get_data()

    def get_data_manager(self):
        """

        :return: 
        """

        return copy.deepcopy(self._data_manager)

    def get_origin(self):
        """

        :return: 
        """

        return self._data_manager.get_origin()

    def get_variable(self, variable_name):
        """

        :param variable_name: 
        :return: 
        """

        return self._data_manager.get_variable(variable_name)

    def get_variable_names(self):
        """

        :return: 
        """

        return self._data_manager.get_variable_names()

    def get_variable_observation(self, variable_name, time, time_window_width=0, match_method='nearest'):
        """

        :param variable_name: 
        :param time: 
        :param time_window_width: 
        :param match_method: 
        :return: 
        """

        return self._data_manager.get_variable_observation(variable_name, time, time_window_width, match_method)

    def get_variable_origin(self, variable_name):
        """
        
        :param variable_name: 
        :return: 
        """

        data_origin = self.get_origin()
        grouped = data_origin.groupby('variable')
        variable_group = grouped.get_group(variable_name)
        variable_origin = list(variable_group['origin'])

        return variable_origin


class ProcessedData(ADVMData):

    def __init__(self, data_manager, configuration_parameters, processing_parameters):
        """

        :param data_manager: DataManager
        :param configuration_parameters: ADVMConfigParam
        :param processing_parameters: ADVMProcParam
        """

        assert isinstance(data_manager, datamanager.DataManager)
        assert isinstance(configuration_parameters, ADVMConfigParam)
        assert isinstance(processing_parameters, ADVMProcParam)

        super().__init__()

        self._data_manager = copy.deepcopy(data_manager)
        self._configuration_parameters = copy.deepcopy(configuration_parameters)
        self._processing_parameters = copy.deepcopy(processing_parameters)

    def add_data(self, other, keep_curr_obs=None):
        """

        :param other: ProcessedADVMSedimentData
        :param keep_curr_obs: 
        :return: 
        """

        if not self._configuration_parameters.is_compatible(other.get_configuration_parameters()) and \
                self._processing_parameters.is_compatible(other.get_processing_parameters()) and \
                isinstance(other, type(self)):
            raise ADVMDataIncompatibleError("ADVM data sets are incompatible")

        other_data_manager = other.get_data_manager()

        combined_data_manager = self._data_manager.add_data(other_data_manager, keep_curr_obs=keep_curr_obs)

        return type(self)(combined_data_manager, self._configuration_parameters, self._processing_parameters)

    def get_processing_parameters(self):
        """

        :return: 
        """

        return copy.deepcopy(self._processing_parameters)


class BackscatterData(ADVMData):

    @abc.abstractmethod
    def __init__(self):

        super().__init__()

    def _calc_cell_range(self):
        """Calculate range of cells along a single beam.

        :return: Range of cells along a single beam
        """

        acoustic_data = self._data_manager.get_data()

        blanking_distance = self._configuration_parameters['Blanking Distance']
        cell_size = self._configuration_parameters['Cell Size']
        number_of_cells = self._configuration_parameters['Number of Cells']

        first_cell_mid_point = blanking_distance + cell_size / 2
        last_cell_mid_point = first_cell_mid_point + (number_of_cells - 1) * cell_size

        slant_angle = self._configuration_parameters['Slant Angle']

        cell_range = np.linspace(first_cell_mid_point,
                                 last_cell_mid_point,
                                 num=number_of_cells) / np.cos(np.radians(slant_angle))

        cell_range = np.tile(cell_range, (acoustic_data.shape[0], 1))

        col_names = ['R{:03}'.format(cell) for cell in range(1, number_of_cells+1)]

        cell_range_df = pd.DataFrame(data=cell_range, index=acoustic_data.index, columns=col_names)

        return cell_range_df

    def get_cell_range(self):
        """Get a DataFrame containing the range of cells along a single beam.

        :return: Range of cells along a single beam
        """

        return self._calc_cell_range()


class ProcessedBackscatterData(ProcessedData, BackscatterData):

    @abc.abstractmethod
    def __init__(self, data_manager, configuration_parameters, processing_parameters):

        self._backscatter_name = None

        super().__init__(data_manager, configuration_parameters, processing_parameters)

    def plot(self, index, ax=None):

        data = self.get_data()

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        line_style_creator = LineStyleGenerator()

        cell_range = self.get_cell_range()

        for i in index:

            line_style = line_style_creator.get_line_style(False)
            marker = line_style_creator.get_marker()
            color = line_style_creator.get_line_color()

            ax.plot(cell_range.ix[i], data.ix[i], ls=line_style, marker=marker, color=color)

        ax.set_xlabel('Range, in meters')
        ax.set_ylabel(self._backscatter_name + ',\nin decibels')

        return ax


class RawBackscatterData(BackscatterData):
    """Class for managing raw backscatter data"""

    # regex string to find acoustic backscatter columns
    _abs_columns_regex = r'^(Cell\d{2}(Amp|SNR)\d{1})$'

    # regex string to find ADVM data columns
    _advm_columns_regex = r'^(Temp|Vbeam|Cell\d{2}(Amp|SNR)\d{1})$'

    def __init__(self, data_manager, configuration_parameters):
        """
        
        :param data_manager: 
        :type data_manager: datamanager.DataManager
        :param configuration_parameters:
        :type configuration_parameters: ADVMConfigParam
        """

        # initialize a configuration parameter object

        assert isinstance(data_manager, datamanager.DataManager)
        assert isinstance(configuration_parameters, ADVMConfigParam)

        super().__init__()

        self._configuration_parameters = copy.deepcopy(configuration_parameters)

        # get only the ADVM data from the passed data manager
        acoustic_data = data_manager.get_data()
        acoustic_df = acoustic_data.filter(regex=self._advm_columns_regex)

        data_origin = data_manager.get_origin()
        data_variable_list = list(data_origin['variable'])
        acoustic_variable_idx = [i for i, item in enumerate(data_variable_list)
                                 if re.search(self._advm_columns_regex, item)]
        acoustic_data_origin = data_origin.ix[acoustic_variable_idx]

        self._data_manager = datamanager.DataManager(acoustic_df, acoustic_data_origin)

    def _calc_measured_backscatter(self, processing_parameters):
        """Calculate measured backscatter values based on processing parameters.

        :return:
        """

        # get the backscatter value to return
        backscatter_values = processing_parameters["Backscatter Values"]

        # get the beam number from the processing parameters
        beam_number = processing_parameters["Beam"]

        # if the beam number is average, calculate the average among the beams
        if beam_number == 'Avg':

            # initialize empty list to hold backscatter data frames
            backscatter_list = []

            number_of_beams = self._configuration_parameters['Number of Beams']

            for beam_number in range(1, number_of_beams + 1):
                beam_backscatter_df = self._get_mb_array(backscatter_values, beam_number)

                backscatter_list.append(beam_backscatter_df)

            # cast to keep PyCharm from complaining
            df_concat = pd.DataFrame(pd.concat(backscatter_list))

            by_row_index = df_concat.groupby(df_concat.index)

            backscatter_df = by_row_index.mean()

        # otherwise, get the backscatter from the single beam
        else:

            backscatter_df = self._get_mb_array(backscatter_values, beam_number)

        # if Amp is selected, apply the intensity scale factor to the backscatter values
        if backscatter_values == 'Amp':
            scale_factor = processing_parameters['Intensity Scale Factor']
            backscatter_df = scale_factor * backscatter_df

        return backscatter_df

    def _get_mb_array(self, backscatter_values, beam_number):
        """Extract measured backscatter values from the acoustic data frame.

        :param backscatter_values: Backscatter values to return. 'SNR' or 'Amp'
        :param beam_number: Beam number to extract
        :return: Dataframe containing backscatter from the requested beam
        """

        acoustic_df = self._data_manager.get_data()

        number_of_cells = int(self._configuration_parameters['Number of Cells'])
        number_of_obs = acoustic_df.shape[0]

        # create an number_of_obs by number_of_cells array of NaNs
        backscatter_array = np.tile(np.nan, (number_of_obs, number_of_cells))

        # create a data frame with the nan values as the data
        mb_column_names = ['MB{:03}'.format(cell) for cell in range(1, number_of_cells + 1)]
        mb_df = pd.DataFrame(index=acoustic_df.index, data=backscatter_array, columns=mb_column_names)

        # go through each cell and fill the data from the acoustic data frame
        # skip the cell and leave the values nan if the cell hasn't been loaded
        for cell in range(1, number_of_cells + 1):

            # get the column names for the backscatter in each data frame
            arg_col_name = r'Cell{:02}{}{:1}'.format(cell, backscatter_values, beam_number)
            mb_col_name = r'MB{:03}'.format(cell)

            # try to fill the columns
            # if fail, continue and leave the values nan
            try:
                mb_df.ix[:, mb_col_name] = acoustic_df.ix[:, arg_col_name]
            except KeyError as err:
                if err.args[0] == arg_col_name:
                    continue
                else:
                    raise err

        return mb_df

    @staticmethod
    def _read_argonaut_ctl_file(arg_ctl_filepath):
        """
        Read the Argonaut '.ctl' file into a configuration dictionary.

        :param arg_ctl_filepath: Filepath containing the Argonaut '.dat' file
        :return: Dictionary containing specific configuration parameters
        """

        # Read specific configuration values from the Argonaut '.ctl' file into a dictionary.
        # The fixed formatting of the '.ctl' file is leveraged to extract values from foreknown file lines.
        config_dict = {}
        line = linecache.getline(arg_ctl_filepath, 10).strip()
        arg_type = line.split("ArgType ------------------- ")[-1:]

        if arg_type == "SL":
            config_dict['Beam Orientation'] = "Horizontal"
        else:
            config_dict['Beam Orientation'] = "Vertical"

        line = linecache.getline(arg_ctl_filepath, 12).strip()
        frequency = line.split("Frequency ------- (kHz) --- ")[-1:]
        config_dict['Frequency'] = float(frequency[0])

        # calculate transducer radius (m)
        if float(frequency[0]) == 3000:
            config_dict['Effective Transducer Diameter'] = 0.015
        elif float(frequency[0]) == 1500:
            config_dict['Effective Transducer Diameter'] = 0.030
        elif float(frequency[0]) == 500:
            config_dict['Effective Transducer Diameter'] = 0.090
        elif np.isnan(float(frequency[0])):
            config_dict['Effective Transducer Diameter'] = "NaN"

        config_dict['Number of Beams'] = int(2)  # always 2; no need to check file for value

        line = linecache.getline(arg_ctl_filepath, 16).strip()
        slant_angle = line.split("SlantAngle ------ (deg) --- ")[-1:]
        config_dict['Slant Angle'] = float(slant_angle[0])

        line = linecache.getline(arg_ctl_filepath, 44).strip()
        slant_angle = line.split("BlankDistance---- (m) ------ ")[-1:]
        config_dict['Blanking Distance'] = float(slant_angle[0])

        line = linecache.getline(arg_ctl_filepath, 45).strip()
        cell_size = line.split("CellSize -------- (m) ------ ")[-1:]
        config_dict['Cell Size'] = float(cell_size[0])

        line = linecache.getline(arg_ctl_filepath, 46).strip()
        number_cells = line.split("Number of Cells ------------ ")[-1:]
        config_dict['Number of Cells'] = int(number_cells[0])

        return config_dict

    @staticmethod
    def _read_argonaut_dat_file(arg_dat_filepath):
        """
        Read the Argonaut '.dat' file into a DataFrame.

        :param arg_dat_filepath: Filepath containing the Argonaut '.dat' file
        :return: Timestamp formatted DataFrame containing '.dat' file contents
        """

        # Read the Argonaut '.dat' file into a DataFrame
        dat_df = pd.read_table(arg_dat_filepath, sep='\s+')

        # rename the relevant columns to the standard/expected names
        dat_df.rename(columns={"Temperature": "Temp", "Level": "Vbeam"}, inplace=True)

        # set dataframe index by using date/time information
        date_time_columns = ["Year", "Month", "Day", "Hour", "Minute", "Second"]
        datetime_index = pd.to_datetime(dat_df[date_time_columns])
        dat_df.set_index(datetime_index, inplace=True)

        # remove non-relevant columns
        relevant_columns = ['Temp', 'Vbeam']
        dat_df = dat_df.filter(regex=r'(' + '|'.join(relevant_columns) + r')$')

        dat_df = dat_df.apply(pd.to_numeric, args=('coerce', ))
        dat_df = dat_df.astype(np.float)

        return dat_df

    @staticmethod
    def _read_argonaut_snr_file(arg_snr_filepath):
        """
        Read the Argonaut '.dat' file into a DataFrame.

        :param arg_snr_filepath: Filepath containing the Argonaut '.dat' file
        :return: Timestamp formatted DataFrame containing '.snr' file contents
        """

        # Read the Argonaut '.snr' file into a DataFrame, combine first two rows to make column headers,
        # and remove unused datetime columns from the DataFrame.
        snr_df = pd.read_table(arg_snr_filepath, sep='\s+', header=None)
        header = snr_df.ix[0] + snr_df.ix[1]
        snr_df.columns = header.str.replace(r"\(.*\)", "")  # remove parentheses and everything inside them from headers
        snr_df = snr_df.ix[2:]

        # rename columns to recognizable date/time elements
        column_names = list(snr_df.columns)
        column_names[1] = 'Year'
        column_names[2] = 'Month'
        column_names[3] = 'Day'
        column_names[4] = 'Hour'
        column_names[5] = 'Minute'
        column_names[6] = 'Second'
        snr_df.columns = column_names

        # create a datetime index and set the dataframe index
        datetime_index = pd.to_datetime(snr_df.ix[:, 'Year':'Second'])
        snr_df.set_index(datetime_index, inplace=True)

        # remove non-relevant columns
        snr_df = snr_df.filter(regex=r'(^Cell\d{2}(Amp|SNR)\d{1})$')

        snr_df = snr_df.apply(pd.to_numeric, args=('coerce', ))
        snr_df = snr_df.astype(np.float)

        return snr_df

    def add_data(self, other, keep_curr_obs=None):
        """Adds other RawADMVSedimentAcoustic data instance to self.

        Throws exception if other ADVMData object is incompatible with self. An exception will be raised if
        keep_curr_obs=None and concurrent observations exist for variables.

        :param other: RawADMVSedimentAcousticData object to be added
        :type other: RawADMVSedimentAcousticData
        :param keep_curr_obs: {None, True, False} Flag to indicate whether or not to keep current observations.
        :return: Merged RawADMVSedimentAcoustic object
        """

        # test compatibility of other data set
        if not self._configuration_parameters.is_compatible(other.get_configuration_parameters()) and \
                isinstance(other, type(self)):

            raise ADVMDataIncompatibleError("ADVM data sets are incompatible")

        other_data = other.get_data()
        other_origin = other.get_origin()

        other_data_manager = datamanager.DataManager(other_data, other_origin)

        combined_data_manager = self._data_manager.add_data(other_data_manager, keep_curr_obs=keep_curr_obs)

        return type(self)(combined_data_manager, self._configuration_parameters)

    def calculate_measured_backscatter(self, processing_parameters):
        """
        
        :param processing_parameters: 
        :return: 
        """

        assert isinstance(processing_parameters, ADVMProcParam)

        measured_backscatter_data = self._calc_measured_backscatter(processing_parameters)
        temperature_data = self._data_manager.get_variable('Temp')
        vertical_beam_data = self._data_manager.get_variable('Vbeam')

        measured_backscatter_data = pd.concat([temperature_data, vertical_beam_data, measured_backscatter_data], axis=1)

        measured_backscatter_origin = self._create_origin_from_data_frame(measured_backscatter_data)

        measured_backscatter_manager = datamanager.DataManager(measured_backscatter_data,
                                                               measured_backscatter_origin)

        return MeasuredBackscatterData(measured_backscatter_manager,
                                       self._configuration_parameters,
                                       processing_parameters)

    @classmethod
    def find_advm_variable_names(cls, df):
        """Finds and return a list of ADVM variables contained within a dataframe.

        :param df:
        :return:
        """

        # compile regular expression pattern
        advm_columns_pattern = re.compile(cls._advm_columns_regex)

        # create empty list to hold column names
        advm_columns = []

        # find the acoustic backscatter column names within the DataFrame
        for column in list(df.keys()):
            abs_match = advm_columns_pattern.fullmatch(column)
            if abs_match is not None:
                advm_columns.append(abs_match.string)

        if len(advm_columns) == 0:
            return None
        else:
            return advm_columns

    @classmethod
    def read_argonaut_data(cls, data_directory, filename):
        """Loads an Argonaut data set into an ADVMData class object.

        The DAT, SNR, and CTL ASCII files that are exported (with headers) from ViewArgonaut must be present.

        :param data_directory: file path containing the Argonaut data files
        :type data_directory: str
        :param filename: root filename for the 3 Argonaut files
        :type filename: str
        :return: ADVMData object containing the Argonaut data set information
        """

        dataset_path = os.path.join(data_directory, filename)

        # Read the Argonaut '.dat' file into a DataFrame
        arg_dat_file = dataset_path + ".dat"
        dat_df = cls._read_argonaut_dat_file(arg_dat_file)

        # Read the Argonaut '.snr' file into a DataFrame
        arg_snr_file = dataset_path + ".snr"
        snr_df = cls._read_argonaut_snr_file(arg_snr_file)

        # Read specific configuration values from the Argonaut '.ctl' file into a dictionary.
        arg_ctl_file = dataset_path + ".ctl"
        config_dict = cls._read_argonaut_ctl_file(arg_ctl_file)
        configuration_parameters = ADVMConfigParam()
        configuration_parameters.update(config_dict)

        # Combine the '.snr' and '.dat.' DataFrames into a single acoustic DataFrame, make the timestamp
        # the index, and return an ADVMData object
        acoustic_df = pd.concat([dat_df, snr_df], axis=1)
        data_origin = datamanager.DataManager.create_data_origin(acoustic_df, dataset_path + "(Arg)")

        data_manager = datamanager.DataManager(acoustic_df, data_origin)

        return cls(data_manager, configuration_parameters)

    @classmethod
    def read_tab_delimited_data(cls, file_path, configuration_parameters):
        """Create an ADVMData object from a tab-delimited text file that contains raw acoustic variables.

        ADVM configuration parameters must be provided as an argument.

        :param file_path: Path to tab-delimited file
        :type file_path: str
        :param configuration_parameters: ADVMConfigParam containing necessary ADVM configuration parameters
        :type configuration_parameters: ADVMConfigParam

        :return: ADVMData object
        """

        data_manager = datamanager.DataManager.read_tab_delimited_data(file_path)

        return cls(data_manager, configuration_parameters)


class MeasuredBackscatterData(ProcessedBackscatterData):

    def __init__(self, data_manager, configuration_parameters, processing_parameters):

        super().__init__(data_manager, configuration_parameters, processing_parameters)

        self._backscatter_name = 'Measured backscatter'

    @staticmethod
    def _apply_cell_range_filter(water_corrected_backscatter, cell_range, min_cell_range, max_cell_range):
        """

        :param water_corrected_backscatter:
        :return: Water corrected backscatter with cell range filter applied
        """

        cells_outside_set_range = (max_cell_range < cell_range) | (cell_range < min_cell_range)

        water_corrected_backscatter[cells_outside_set_range] = np.nan

        return water_corrected_backscatter

    def _apply_minwcb_correction(self, water_corrected_backscatter):
        """Remove the values of the cells including and beyond the cell with the minimum water corrected
        backscatter value.

        :param water_corrected_backscatter: Water corrected backscatter array
        :return:
        """

        number_of_cells = self._configuration_parameters['Number of Cells']

        # get the column index of the minimum value in each row
        min_wcb_index = np.argmin(water_corrected_backscatter, axis=1)

        # set the index back one to include cell with min wcb for samples that have a wcb with more than one valid cell
        valid_index = (np.sum(~np.isnan(water_corrected_backscatter), axis=1) > 1) & (min_wcb_index > 0)
        min_wcb_index[valid_index] -= 1

        # get the flat index of the minimum values
        index_array = np.array([np.arange(water_corrected_backscatter.shape[0]), min_wcb_index])
        flat_index = np.ravel_multi_index(index_array, water_corrected_backscatter.shape)

        # get a flat matrix of the cell ranges
        cell_range_df = self.get_cell_range()
        cell_range_mat = cell_range_df.as_matrix()
        cell_range_flat = cell_range_mat.flatten()

        # create an nobs x ncell array of the range of the minimum values
        # where nobs is the number of observations and ncell is the number of cells
        min_wcb_cell_range = cell_range_flat[flat_index]
        min_wcb_cell_range = np.tile(min_wcb_cell_range.reshape((min_wcb_cell_range.shape[0], 1)),
                                     (1, number_of_cells))

        # get the index of the cells that are beyond the cell with the minimum wcb
        wcb_gt_min_index = cell_range_mat > min_wcb_cell_range

        # find the number of bad cells
        number_of_bad_cells = np.sum(wcb_gt_min_index, 1)

        # set the index of the observations with one bad cell to false
        wcb_gt_min_index[number_of_bad_cells == 1, :] = False

        # set the cells that are further away from the adjusted range to nan
        water_corrected_backscatter[wcb_gt_min_index] = np.nan

        return water_corrected_backscatter

    @staticmethod
    def _calc_alpha_w(temperature, frequency):
        """Calculate alpha_w - the water-absorption coefficient (WAC) in dB/m.

        :return: alpha_w
        """

        f_T = 21.9 * 10 ** (6 - 1520 / (temperature + 273))  # temperature-dependent relaxation frequency
        alpha_w = 8.686 * 3.38e-6 * (frequency ** 2) / f_T  # water attenuation coefficient

        return alpha_w

    @staticmethod
    def _calc_rcrit(wavelength, trans_rad):
        """
        Calculate the critical distance from the transducer

        :param wavelength: Array containing wavelength, in meters
        :param trans_rad: Scalar radius of transducer, in meters
        :return:
        """

        r_crit = (np.pi * (trans_rad ** 2)) / wavelength

        return r_crit

    @staticmethod
    def _calc_psi(r_crit, cell_range):
        """Calculate psi - the near field correction coefficient.

        "Function which accounts for the departure of the backscatter signal from spherical spreading
        in the near field of the transducer" Downing (1995)

        :param r_crit: Array containing critical range
        :param cell_range: Array containing the mid-point distances of all of the cells
        :return: psi
        """

        number_of_cells = cell_range.shape[1]

        try:
            Zz = cell_range / np.tile(r_crit, (1, number_of_cells))
        except ValueError:
            r_crit = np.expand_dims(r_crit, axis=1)
            Zz = cell_range / np.tile(r_crit, (1, number_of_cells))

        psi = (1 + 1.35 * Zz + (2.5 * Zz) ** 3.2) / (1.35 * Zz + (2.5 * Zz) ** 3.2)

        return psi

    @classmethod
    def _calc_geometric_loss(cls, cell_range, **kwargs):
        """Calculate the geometric two-way transmission loss due to spherical beam spreading

        :param cell_range: Array of range of cells, in meters
        :param **kwargs
            See below

        :Keyword Arguments:
            * *temperature* --
                Temperature, in Celsius
            * *frequency* --
                Frequency, in kilohertz
            * *trans_rad* --
                Transducer radius, in meters

        :return:
        """

        nearfield_corr = kwargs.pop('nearfield_corr', False)

        if nearfield_corr:

            temperature = kwargs['temperature']
            frequency = np.float(kwargs['frequency'])
            trans_rad = np.float(kwargs['trans_rad'])

            speed_of_sound = cls._calc_speed_of_sound(temperature)
            wavelength = cls._calc_wavelength(speed_of_sound, frequency)
            r_crit = cls._calc_rcrit(wavelength, trans_rad)
            psi = cls._calc_psi(r_crit, cell_range)

            geometric_loss = 20 * np.log10(psi * cell_range)

        else:

            geometric_loss = 20 * np.log10(cell_range)

        return geometric_loss

    @classmethod
    def _calc_water_absorption_loss(cls, temperature, frequency, cell_range):
        """

        :param temperature:
        :param frequency:
        :param cell_range:
        :return:
        """

        alpha_w = np.array(cls._calc_alpha_w(temperature, frequency))

        number_of_cells = cell_range.shape[1]

        try:
            water_absorption_loss = 2 * np.tile(alpha_w, (1, number_of_cells)) * cell_range
        except ValueError:
            alpha_w = np.expand_dims(alpha_w, axis=1)
            water_absorption_loss = 2 * np.tile(alpha_w, (1, number_of_cells)) * cell_range

        return water_absorption_loss

    def _calc_water_corrected_backscatter(self):
        """Calculate water corrected backscatter (WCB).

        :return:
        """

        acoustic_data = self._data_manager.get_data()
        cell_range = self.get_cell_range()
        # calculate the water corrected backscatter
        geometric_loss = self._calc_geometric_loss(cell_range.as_matrix(),
                                                   temperature=acoustic_data['Temp'].as_matrix(),
                                                   frequency=self._configuration_parameters['Frequency'],
                                                   trans_rad=self._configuration_parameters[
                                                                 'Effective Transducer Diameter']/2,
                                                   nearfield_corr=self._processing_parameters['Near Field Correction'])
        water_absorption_loss = self._calc_water_absorption_loss(acoustic_data['Temp'].as_matrix(),
                                                                 self._configuration_parameters['Frequency'],
                                                                 cell_range.as_matrix())
        two_way_transmission_loss = geometric_loss + water_absorption_loss

        measured_backscatter = self.get_measured_backscatter()
        water_corrected_backscatter = measured_backscatter.as_matrix() + two_way_transmission_loss

        # adjust the water corrected backscatter profile
        if self._processing_parameters['WCB Profile Adjustment']:

            water_corrected_backscatter = self._apply_minwcb_correction(water_corrected_backscatter)

        water_corrected_backscatter = self._remove_min_vbeam(water_corrected_backscatter,
                                                             self._processing_parameters['Minimum Vbeam'])

        min_cell_distance = self._processing_parameters['Minimum Cell Mid-Point Distance']
        max_cell_distance = self._processing_parameters['Maximum Cell Mid-Point Distance']
        water_corrected_backscatter = self._apply_cell_range_filter(water_corrected_backscatter,
                                                                    cell_range.as_matrix(),
                                                                    min_cell_distance,
                                                                    max_cell_distance)

        # create a data frame of the water corrected backscatter observations
        index = acoustic_data.index
        wcb_columns = ['WCB{:03}'.format(cell) for cell in
                       range(1, self._configuration_parameters['Number of Cells']+1)]
        water_corrected_backscatter_df = pd.DataFrame(index=index,
                                                      data=water_corrected_backscatter,
                                                      columns=wcb_columns)

        return water_corrected_backscatter_df

    def _remove_min_vbeam(self, water_corrected_backscatter, minimum_vbeam):
        """Remove observations that have a vertical beam value that are below the set threshold.

        :param water_corrected_backscatter: Water corrected backscatter array
        :return:
        """

        acoustic_df = self._data_manager.get_data()

        vbeam = acoustic_df['Vbeam'].as_matrix()

        index_below_min_vbeam = (vbeam < minimum_vbeam)

        water_corrected_backscatter[index_below_min_vbeam, :] = np.nan

        return water_corrected_backscatter

    def calculate_water_corrected_backscatter(self):
        """
        
        :return: 
        """

        water_corrected_backscatter_df = self._calc_water_corrected_backscatter()
        water_corrected_backscatter_origin = self._create_origin_from_data_frame(water_corrected_backscatter_df)
        water_corrected_backscatter_data_manager = datamanager.DataManager(water_corrected_backscatter_df,
                                                                           water_corrected_backscatter_origin)
        return WaterCorrectedBackscatterData(water_corrected_backscatter_data_manager,
                                             self.get_configuration_parameters(),
                                             self.get_processing_parameters())

    def get_data(self):
        """
        
        :return: 
        """

        return self.get_measured_backscatter()

    def get_measured_backscatter(self):
        """
        
        :return: 
        """

        data = self._data_manager.get_data()

        return data.filter(regex='^MB\d{3}$')


class WaterCorrectedBackscatterData(ProcessedBackscatterData):

    def __init__(self, data_manager, configuration_parameters, processing_parameters):

        super().__init__(data_manager, configuration_parameters, processing_parameters)

        self._backscatter_name = 'Water corrected backscatter'

    @staticmethod
    def _calc_sediment_attenuation_coefficient(water_corrected_backscatter, cell_range):
        """Calculate the sediment attenuation coefficient (SAC) in dB/m.

        :return: sac array
        """

        wcb = water_corrected_backscatter.as_matrix()
        cell_range = cell_range.as_matrix()

        # make sure both matrices are the same shape
        assert wcb.shape == cell_range.shape

        # create an empty nan matrix for the SAC
        sac_arr = np.tile(np.nan, (wcb.shape[0],))

        # iterate over all rows
        for row_index in np.arange(wcb.shape[0]):

            # find the column indices that are valid
            fit_index = ~np.isnan(wcb[row_index, :])

            # if there are more than one valid cells, calculate the slope
            if np.sum(fit_index) > 1:

                # find the slope of the WCB with respect to the cell range
                x = cell_range[row_index, fit_index]
                y = wcb[row_index, fit_index]
                xy_mean = np.mean(x * y)
                x_mean = np.mean(x)
                y_mean = np.mean(y)
                cov_xy = xy_mean - x_mean * y_mean
                var_x = np.var(x)
                slope = cov_xy / var_x

            else:
                continue

            # calculate the SAC
            sac_arr[row_index] = -0.5 * slope

        sac_series = pd.DataFrame(data=sac_arr, index=water_corrected_backscatter.index, columns=['SAC'])

        return sac_series

    def _calc_sediment_corrected_backscatter(self):
        """

        :return:
        """

        water_corrected_backscatter = self.get_water_corrected_backscatter()
        wcb = water_corrected_backscatter.as_matrix()
        sediment_attenuation_coefficient_manager = self.calculate_sediment_attenuation_coefficient()
        sac_df = sediment_attenuation_coefficient_manager.get_data()
        sac = sac_df.as_matrix()
        cell_range = self.get_cell_range().as_matrix()

        try:
            scb = wcb + 2 * np.tile(sac, (1, cell_range.shape[1])) * cell_range
        except ValueError:
            sac = np.expand_dims(sac, axis=1)
            scb = wcb + 2 * np.tile(sac, (1, cell_range.shape[1])) * cell_range

        scb = self._single_cell_wcb_correction(wcb, scb)

        # create DateFrame to return
        index = water_corrected_backscatter.index
        scb_columns = ['SCB{:03}'.format(cell) for cell
                       in range(1, self._configuration_parameters['Number of Cells'] + 1)]
        scb_df = pd.DataFrame(index=index, data=scb, columns=scb_columns)

        return scb_df

    @staticmethod
    def _single_cell_wcb_correction(water_corrected_backscatter, sediment_corrected_backscatter):
        """

        :param water_corrected_backscatter:
        :param sediment_corrected_backscatter:
        :return:
        """

        # row index of all single-cell wcb observations
        single_cell_index = np.sum(~np.isnan(water_corrected_backscatter), axis=1) == 1

        # replace NaN in single-cell observations with the WCB value
        sediment_corrected_backscatter[single_cell_index, :] = water_corrected_backscatter[single_cell_index, :]

        return sediment_corrected_backscatter

    def calculate_sediment_attenuation_coefficient(self):
        """
        
        :return: 
        """

        water_corrected_backscatter = self.get_water_corrected_backscatter()
        cell_range = self.get_cell_range()

        sediment_attenuation_coefficient = self._calc_sediment_attenuation_coefficient(water_corrected_backscatter,
                                                                                       cell_range)
        data_origin = self._create_origin_from_data_frame(sediment_attenuation_coefficient)
        data_manager = datamanager.DataManager(sediment_attenuation_coefficient, data_origin)

        return ProcessedData(data_manager, self.get_configuration_parameters(), self.get_processing_parameters())

    def calculate_sediment_corrected_backscatter(self):
        """
        
        :return: 
        """

        sediment_corrected_backscatter_df = self._calc_sediment_corrected_backscatter()
        sediment_corrected_backscatter_origin = self._create_origin_from_data_frame(sediment_corrected_backscatter_df)

        sediment_corrected_backscatter_manager = datamanager.DataManager(sediment_corrected_backscatter_df,
                                                                         sediment_corrected_backscatter_origin)

        return SedimentCorrectedBackscatterData(sediment_corrected_backscatter_manager,
                                                self.get_configuration_parameters(),
                                                self.get_processing_parameters())

    def get_data(self):
        """
        
        :return: 
        """

        return self.get_water_corrected_backscatter()

    def get_water_corrected_backscatter(self):

        data = self._data_manager.get_data()
        return data.filter(regex='^WCB\d{3}$')


class SedimentCorrectedBackscatterData(ProcessedBackscatterData):

    def __init__(self, data_manager, configuration_parameters, processing_parameters):

        super().__init__(data_manager, configuration_parameters, processing_parameters)

        self._backscatter_name = 'Sediment corrected backscatter'

    def calculate_mean_sediment_corrected_backscatter(self):
        """
        
        :return: 
        """

        sediment_corrected_backscatter = self.get_data()

        mean_sediment_corrected_backscatter = pd.DataFrame(sediment_corrected_backscatter.mean(axis=1),
                                                           columns=['MeanSCB'], dtype=np.float)
        data_origin = self._create_origin_from_data_frame(mean_sediment_corrected_backscatter)
        data_manager = datamanager.DataManager(mean_sediment_corrected_backscatter, data_origin)

        return ProcessedData(data_manager, self.get_configuration_parameters(), self.get_processing_parameters())


class ADVMBackscatterDataProcessor:

    def __init__(self, raw_advm_backscatter_data, processing_parameters=None):
        """
        
        :param raw_advm_backscatter_data: 
        :type raw_advm_backscatter_data: RawBackscatterData
        """

        self._raw_advm_backscatter_data = copy.deepcopy(raw_advm_backscatter_data)

        self._measured_backscatter_data = None
        self._water_corrected_backscatter_data = None
        self._sediment_corrected_backscatter_data = None

        if processing_parameters is not None:

            self._acoustic_parameters = self.calculate_acoustic_parameters(processing_parameters)

        else:

            self._acoustic_parameters = None

    def calculate_acoustic_parameters(self, processing_parameters):
        """

        :param processing_parameters: 
        :return: 
        """

        self._measured_backscatter_data = \
            self._raw_advm_backscatter_data.calculate_measured_backscatter(processing_parameters)
        self._water_corrected_backscatter_data = self._measured_backscatter_data.calculate_water_corrected_backscatter()
        self._sediment_corrected_backscatter_data = \
            self._water_corrected_backscatter_data.calculate_sediment_corrected_backscatter()

        sediment_attenuation_coefficient = \
            self._water_corrected_backscatter_data.calculate_sediment_attenuation_coefficient()
        mean_sediment_corrected_backscatter = \
            self._sediment_corrected_backscatter_data.calculate_mean_sediment_corrected_backscatter()

        self._acoustic_parameters = sediment_attenuation_coefficient.add_data(mean_sediment_corrected_backscatter)

        return copy.deepcopy(self._acoustic_parameters)

    def get_acoustic_parameters(self):

        return copy.deepcopy(self._acoustic_parameters)

    def get_configuration_parameters(self):

        return self._raw_advm_backscatter_data.get_configuration_parameters()

    def get_data(self):
        """

        :return:
        """

        return self._acoustic_parameters.get_data()

    def get_processing_parameters(self):

        if self._measured_backscatter_data is None:

            processing_parameters = None

        else:

            processing_parameters = self._measured_backscatter_data.get_processing_parameters()

        return processing_parameters

    def get_variable(self, variable_name):
        """

        :param variable_name:
        :return:
        """

        return self._acoustic_parameters.get_variable(variable_name)

    def get_variable_names(self):
        """

        :return:
        """

        if self._acoustic_parameters is None:
            return None
        else:
            return self._acoustic_parameters.get_variable_names()

    def get_variable_observation(self, variable_name, time, time_window_width=0, match_method='nearest'):
        """

        :param variable_name:
        :param time:
        :param time_window_width:
        :param match_method:
        :return:
        """

        if self._acoustic_parameters is None:
            return None
        else:
            return self._acoustic_parameters.get_variable_observation(variable_name, time,
                                                                      time_window_width, match_method)

    def get_variable_origin(self, variable_name):
        """

        :return:
        """

        if self._acoustic_parameters is None:
            return None
        else:
            return self._acoustic_parameters.get_variable_origin(variable_name)

    def plot(self, index):
        """
        
        :param index: 
        :return: 
        """

        if self._acoustic_parameters is not None:

            fig, (scb_ax, wcb_ax, mb_ax) = plt.subplots(nrows=3, sharex=True)

            self._measured_backscatter_data.plot(index, ax=mb_ax)
            self._water_corrected_backscatter_data.plot(index, ax=wcb_ax)
            self._sediment_corrected_backscatter_data.plot(index, ax=scb_ax)

            scb_ax.set_xlabel('')
            wcb_ax.set_xlabel('')

        else:

            fig = None

        return fig

    @classmethod
    def read_argonaut_data(cls, data_directory, filename):
        """
        
        :param data_directory: 
        :param filename: 
        :return: 
        """

        raw_advm_backscatter_data = RawBackscatterData.read_argonaut_data(data_directory, filename)

        return cls(raw_advm_backscatter_data)


class BackscatterRatingModel(surrogatemodel.SurrogateRatingModel):

    def plot_backscatter_profiles(self, observation_numbers=None):
        """

        :param observation_numbers:
        :return:
        """

        model_dataset = self._model.get_model_dataset()

        if observation_numbers is None:

            index = model_dataset.index

        else:

            included_observations = ~(model_dataset['Missing'] | model_dataset['Excluded'])
            index = model_dataset.index[observation_numbers]
            index = index[included_observations]

        acoustic_data_index = self._surrogate_data.get_data().index
        match_time = self._match_time[self._surrogate_variables[0]]
        time_diff = timedelta(minutes=match_time/2.)

        plot_index = pd.DatetimeIndex([])

        for datetime in index:

            nearest_ix = acoustic_data_index.get_loc(datetime, method='nearest', tolerance=time_diff)
            nearest_time = acoustic_data_index[nearest_ix]
            plot_index = plot_index.append((pd.DatetimeIndex([nearest_time])))

        return self._surrogate_data.plot(plot_index)
