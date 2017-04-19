import abc
import copy
import linecache
import os
import re

import pandas as pd
import numpy as np

from data import SurrogateData


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


class ADVMSedimentAcousticData(SurrogateData):
    """Data manager class for ADVM sediment acoustic data"""

    # regex string to find acoustic backscatter columns
    _abs_columns_regex = r'^(Cell\d{2}(Amp|SNR)\d{1})$'

    # regex string to find ADVM data columns
    _advm_columns_regex = r'^(Temp|Vbeam|Cell\d{2}(Amp|SNR)\d{1})$'

    def __init__(self, config_params, acoustic_data, data_origin):
        """
        Initializes ADVMData instance. Creates default processing configuration data attribute.

        :param config_params: ADVM configuration data required to process ADVM data
        :param acoustic_df: DataFrame containing acoustic data
        """

        # get only the ADVM data from the passed DataFrame
        self._acoustic_df = acoustic_data.filter(regex=self._advm_columns_regex)

        # rename the index column
        # self._acoustic_df.index.name = 'DateTime'

        # initialize a configuration parameter object
        self._config_param = ADVMConfigParam()

        # update the configuration parameters
        self._config_param.update(config_params)

        # initialize a processing parameter object
        self._proc_param = ADVMProcParam(self._config_param["Number of Cells"])

        self._cell_range = pd.DataFrame()
        self._mb = pd.DataFrame()
        self._wcb = pd.DataFrame()
        self._scb = pd.DataFrame()

        acoustic_variables = self._update_acoustic_data()

        super().__init__(acoustic_variables, data_origin)

    def _apply_cell_range_filter(self, water_corrected_backscatter):
        """

        :param water_corrected_backscatter:
        :return: Water corrected backscatter with cell range filter applied
        """

        max_cell_range = self._proc_param['Maximum Cell Mid-Point Distance']
        min_cell_range = self._proc_param['Minimum Cell Mid-Point Distance']

        cell_range = self.get_cell_range().as_matrix()

        cells_outside_set_range = (max_cell_range < cell_range) | (cell_range < min_cell_range)

        water_corrected_backscatter[cells_outside_set_range] = np.nan

        return water_corrected_backscatter

    def _apply_minwcb_correction(self, water_corrected_backscatter):
        """Remove the values of the cells including and beyond the cell with the minimum water corrected
        backscatter value.

        :param water_corrected_backscatter: Water corrected backscatter array
        :return:
        """

        number_of_cells = self._config_param['Number of Cells']

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

        assert isinstance(temperature, np.ndarray) and isinstance(frequency, float)

        f_T = 21.9 * 10 ** (6 - 1520 / (temperature + 273))  # temperature-dependent relaxation frequency
        alpha_w = 8.686 * 3.38e-6 * (frequency ** 2) / f_T  # water attenuation coefficient

        return alpha_w

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
            frequency = kwargs['frequency']
            trans_rad = kwargs['trans_rad']

            speed_of_sound = cls._calc_speed_of_sound(temperature)
            wavelength = cls._calc_wavelength(speed_of_sound, frequency)
            r_crit = cls._calc_rcrit(wavelength, trans_rad)
            psi = cls._calc_psi(r_crit, cell_range)

            geometric_loss = 20 * np.log10(psi * cell_range)

        else:

            geometric_loss = 20 * np.log10(cell_range)

        return geometric_loss

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

        return snr_df

    @staticmethod
    def _calc_sac(wcb, cell_range):
        """Calculate the sediment attenuation coefficient (SAC) in dB/m.

        :return: sac array
        """

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

        return sac_arr

    @classmethod
    def _calc_scb(cls, wcb, sac, cell_range):
        """

        :param wcb:
        :param sac:
        :return:
        """
        try:
            scb = wcb + 2 * np.tile(sac, (1, cell_range.shape[1])) * cell_range
        except ValueError:
            sac = np.expand_dims(sac, axis=1)
            scb = wcb + 2 * np.tile(sac, (1, cell_range.shape[1])) * cell_range

        scb = cls._single_cell_wcb_correction(wcb, scb)

        return scb

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

    @staticmethod
    def _calc_wavelength(speed_of_sound, frequency):
        """Calculate the wavelength of an acoustic signal.

        :param speed_of_sound: Array containing the speed of sound in meters per second
        :param frequency: Scalar containing acoustic frequency in kHz
        :return:
        """

        wavelength = speed_of_sound / (frequency * 1e3)

        return wavelength

    @classmethod
    def _calc_wcb(cls, mb, temperature, frequency, cell_range, trans_rad, nearfield_corr):
        """
        Calculate the water corrected backscatter. Include the geometric loss due to spherical spreading.

        :param mb: Measured backscatter, in decibels
        :param temperature: Temperature, in Celsius
        :param frequency: Frequency, in kilohertz
        :param cell_range: Mid-point cell range, in meters
        :param trans_rad: Transducer radius, in meters
        :param nearfield_corr: Flag to use nearfield_correction
        :return:
        """

        geometric_loss = cls._calc_geometric_loss(cell_range,
                                                  temperature=temperature,
                                                  frequency=frequency,
                                                  trans_rad=trans_rad,
                                                  nearfield_corr=nearfield_corr)

        water_absorption_loss = cls._calc_water_absorption_loss(temperature, frequency, cell_range)

        two_way_transmission_loss = geometric_loss + water_absorption_loss

        wcb = mb + two_way_transmission_loss

        return wcb

    @staticmethod
    def _get_acoustic_data_origin(data_path):
        """

        :param data_path:
        :return:
        """
        data = [['MeanSCB', data_path], ['SAC', data_path]]
        data_origin = pd.DataFrame(data=data, columns=['variable', 'origin'])
        return data_origin

    def _get_mb_array(self, backscatter_values, beam_number):
        """Extract measured backscatter values from the acoustic data frame.

        :param backscatter_values: Backscatter values to return. 'SNR' or 'Amp'
        :param beam_number: Beam number to extract
        :return: Dataframe containing backscatter from the requested beam
        """

        assert type(backscatter_values) is str and type(beam_number) is int

        number_of_cells = int(self._config_param['Number of Cells'])
        number_of_obs = self._acoustic_df.shape[0]

        # create an number_of_obs by number_of_cells array of NaNs
        backscatter_array = np.tile(np.nan, (number_of_obs, number_of_cells))

        # create a data frame with the nan values as the data
        mb_column_names = ['MB{:03}'.format(cell) for cell in range(1, number_of_cells + 1)]
        mb_df = pd.DataFrame(index=self._acoustic_df.index, data=backscatter_array, columns=mb_column_names)

        # go through each cell and fill the data from the acoustic data frame
        # skip the cell and leave the values nan if the cell hasn't been loaded
        for cell in range(1, number_of_cells + 1):

            # the patern of the columns to search for
            # col_pattern = r'(^Cell\d{2}(' + backscatter_values + str(beam_number) + r'))$'
            # backscatter_df = self._acoustic_df.filter(regex=col_pattern)

            # get the column names for the backscatter in each dataframe
            arg_col_name = r'Cell{:02}{}{:1}'.format(cell, backscatter_values, beam_number)
            mb_col_name = r'MB{:03}'.format(cell)

            # try to fill the columns
            # if fail, continue and leave the values nan
            try:
                mb_df.ix[:, mb_col_name] = self._acoustic_df.ix[:, arg_col_name]
            except KeyError as err:
                if err.args[0] == arg_col_name:
                    continue
                else:
                    raise err

        return mb_df

    def _remove_min_vbeam(self, water_corrected_backscatter):
        """Remove observations that have a vertical beam value that are below the set threshold.

        :param water_corrected_backscatter: Water corrected backscatter array
        :return:
        """

        vbeam = self._acoustic_df['Vbeam'].as_matrix()

        min_vbeam = self._proc_param['Minimum Vbeam']

        index_below_min_vbeam = (vbeam < min_vbeam)

        water_corrected_backscatter[index_below_min_vbeam, :] = np.nan

        return water_corrected_backscatter

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

    def _update_acoustic_data(self):
        """

        :return:
        """
        self._update_cell_range()
        self._update_measured_backscatter()
        self._update_water_corrected_backscatter()
        sac = self._calc_sac(self._wcb.as_matrix(), self._cell_range.as_matrix())
        self._update_sediment_corrected_backscatter(sac)

        mean_scb = pd.Series(self._scb.mean(axis=1), name='MeanSCB', dtype=np.float)
        # mean_scb.name = 'MeanSCB'
        sac_df = pd.Series(data=sac, index=mean_scb.index, name='SAC', dtype=np.float)

        acoustic_data = pd.concat([mean_scb, sac_df], axis=1)

        return acoustic_data

    def _update_cell_range(self):
        """Calculate range of cells along a single beam.

        :return: Range of cells along a single beam
        """

        # TODO: Add feature to calculate nominal vs. measurement cell range
        blanking_distance = self._config_param['Blanking Distance']
        cell_size = self._config_param['Cell Size']
        number_of_cells = self._config_param['Number of Cells']

        first_cell_mid_point = blanking_distance + cell_size / 2
        last_cell_mid_point = first_cell_mid_point + (number_of_cells - 1) * cell_size

        slant_angle = self._config_param['Slant Angle']

        cell_range = np.linspace(first_cell_mid_point,
                                 last_cell_mid_point,
                                 num=number_of_cells) / np.cos(np.radians(slant_angle))

        cell_range = np.tile(cell_range, (self._acoustic_df.shape[0], 1))

        col_names = ['R{:03}'.format(cell) for cell in range(1, number_of_cells+1)]

        cell_range_df = pd.DataFrame(data=cell_range, index=self._acoustic_df.index, columns=col_names)

        self._cell_range = cell_range_df

    def _update_measured_backscatter(self):
        """Calculate measured backscatter values based on processing parameters.

        :return:
        """

        # get the backscatter value to return
        backscatter_values = self._proc_param["Backscatter Values"]

        # get the beam number from the processing parameters
        beam = self._proc_param["Beam"]

        # if the beam number is average, calculate the average among the beams
        if beam == 'Avg':

            # initialize empty list to hold backscatter dataframes
            backscatter_list = []

            number_of_beams = self._config_param['Number of Beams']

            for beam in range(1, number_of_beams + 1):
                beam_backscatter_df = self._get_mb_array(backscatter_values, beam)

                backscatter_list.append(beam_backscatter_df)

            # cast to keep PyCharm from complaining
            df_concat = pd.DataFrame(pd.concat(backscatter_list))

            by_row_index = df_concat.groupby(df_concat.index)

            backscatter_df = by_row_index.mean()

        # otherwise, get the backscatter from the single beam
        else:

            backscatter_df = self._get_mb_array(backscatter_values, beam)

        # if Amp is selected, apply the intensity scale factor to the backscatter values
        if backscatter_values == 'Amp':
            scale_factor = self._proc_param['Intensity Scale Factor']
            backscatter_df = scale_factor * backscatter_df

        self._mb = backscatter_df

    def _update_sediment_corrected_backscatter(self, sac):
        """

        :return:
        """
        # get the cell range
        cell_range = self._cell_range.as_matrix()

        # get the water corrected backscatter, with corrections specific to this instance of ADVMData
        wcb = self._wcb.as_matrix()

        # calculate sediment attenuation coefficient and sediment corrected backscatter
        # sac = self._calc_sac(wcb, cell_range)
        # sac = self.get_variable('SAC').as_matrix()

        scb = self._calc_scb(wcb, sac, cell_range)

        # create DateFrame to return
        index = self._acoustic_df.index
        scb_columns = ['SCB{:03}'.format(cell) for cell in range(1, self._config_param['Number of Cells'] + 1)]
        scb_df = pd.DataFrame(index=index, data=scb, columns=scb_columns)

        self._scb = scb_df

    def _update_water_corrected_backscatter(self):
        """Calculate water corrected backscatter (WCB).

        :return:
        """
        cell_range = self._cell_range.as_matrix()
        temperature = self._acoustic_df['Temp'].as_matrix()
        frequency = self._config_param['Frequency']
        trans_rad = self._config_param['Effective Transducer Diameter']/2
        nearfield_corr = self._proc_param['Near Field Correction']

        measured_backscatter = self._mb.as_matrix()  # measured backscatter values

        water_corrected_backscatter = self._calc_wcb(measured_backscatter, temperature, frequency, cell_range,
                                                     trans_rad, nearfield_corr)

        # adjust the water corrected backscatter profile
        if self._proc_param['WCB Profile Adjustment']:

            water_corrected_backscatter = self._apply_minwcb_correction(water_corrected_backscatter)

        water_corrected_backscatter = self._remove_min_vbeam(water_corrected_backscatter)
        water_corrected_backscatter = self._apply_cell_range_filter(water_corrected_backscatter)

        # create a dataframe of the water corrected backscatter observations
        index = self._acoustic_df.index
        wcb_columns = ['WCB{:03}'.format(cell) for cell in range(1, self._config_param['Number of Cells']+1)]
        wcb = pd.DataFrame(index=index, data=water_corrected_backscatter, columns=wcb_columns)

        self._wcb = wcb

    def add_data(self, other, keep_curr_obs=None):
        """Adds and another DataManager object containing acoustic data to the current object.

        Throws exception if other ADVMData object is incompatible with self. An exception will be raised if
        keep_curr_obs=None and concurrent observations exist for variables.

        :param other: ADVMData object to be added
        :type other: ADVMData
        :param keep_curr_obs: {None, True, False} Flag to indicate whether or not to keep current observations.
        :return: Merged ADVMData object
        """

        # check type of other
        # if not isinstance(other, ADVMData):
        #     raise TypeError('other must be of type data.ADVMData')

        # check type of keep_curr_obs
        # if (keep_curr_obs is not None) and not (isinstance(keep_curr_obs, bool)):
        #     raise TypeError('keep_curr_obs type must be None or bool')

        # test compatibility of other data set
        if self._config_param.is_compatible(other.get_config_params()):

            other.set_proc_params(self.get_proc_params())

            self._acoustic_df = pd.concat([self._acoustic_df, other._acoustic_df])

            grouped = self._acoustic_df.groupby(level=0)

            if keep_curr_obs:
                self._acoustic_df = grouped.first()
            else:
                self._acoustic_df = grouped.last()

        else:

            raise ADVMDataIncompatibleError("ADVM data sets are incompatible")

        # self._data = self._compute_acoustic_data()
        super().add_data(other, keep_curr_obs=keep_curr_obs)

    @classmethod
    def drop_abs_variables(cls, surrogate_data):
        """Drop the acoustic backscatter (ABS) variables from a DataManager object.

        :param surrogate_data: DataManager object
        :type surrogate_data: DataManager
        :return:
        """

        # compile regular expression pattern
        abs_columns_pattern = re.compile(cls._abs_columns_regex)

        # create empty list to hold column names
        abs_column_names = []

        # find the acoustic backscatter column names within the DataFrame
        for column in list(surrogate_data._data.keys()):
            abs_match = abs_columns_pattern.fullmatch(column)
            if abs_match is not None:
                abs_column_names.append(abs_match.string)

        # return a copy of the DataFrame with the abs columns dropped
        return surrogate_data._data.drop(abs_column_names, axis=1)

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
    def from_surrogate_data(cls, surrogate_data, config_params):
        """Convert the variables within a DataManager object into an ADVMData object.

        ADVM configuration parameters must be provided as an argument.

        :param surrogate_data: DataManager object containing acoustic variables
        :type surrogate_data: DataManager
        :param config_params: Dictionary containing necessary ADVM configuration parameters
        :type config_params: dict
        :return: ADVMData object
        """

        acoustic_df = surrogate_data._data.filter(regex=cls._advm_columns_regex)

        data_origin = pd.DataFrame(columns=['variable', 'origin'])
        origin_group = surrogate_data._data_origin.groupby('origin')

        for origin in list(origin_group.groups):
            tmp_data_origin = cls._get_acoustic_data_origin(origin)
            data_origin = data_origin.append(tmp_data_origin)

        return cls(config_params, acoustic_df, data_origin)

    def get_cell_range(self):
        """Get a DataFrame containing the range of cells along a single beam.

        :return: Range of cells along a single beam
        """

        return self._cell_range.copy(deep=True)

    def get_config_params(self):
        """Return a dictionary containing configuration parameters.

        :return: Dictionary containing configuration parameters
        """

        return self._config_param.get_dict()

    def get_mb(self):
        """Get a DataFrame containing the measured backscatter.

        :return: DataFrame containing measured backscatter values
        """

        return self._mb.copy(deep=True)

    def get_proc_params(self):
        """Return dictionary containing processing parameters.

        :return: Dictionary containing processing parameters
        """

        return self._proc_param.get_dict()

    def get_scb(self):
        """Get a DataFrame containing sediment corrected backscatter.

        :return: Sediment corrected backscatter for all cells in the acoustic time series.
        """
        return self._scb.copy(deep=True)

    def get_wcb(self):
        """Get a DataFrame containing water corrected backscatter.

        :return: Water corrected backscatter for all cells in the acoustic time series
        """

        return self._wcb.copy(deep=True)

    @classmethod
    def read_argonaut_data(cls, data_path, filename):
        """Loads an Argonaut data set into an ADVMData class object.

        The DAT, SNR, and CTL ASCII files that are exported (with headers) from ViewArgonaut must be present.

        :param data_path: file path containing the Argonaut data files
        :type data_path: str
        :param filename: root filename for the 3 Argonaut files
        :type file: str
        :return: ADVMData object containing the Argonaut data set information
        """

        dataset_path = os.path.join(data_path, filename)

        # Read the Argonaut '.dat' file into a DataFrame
        arg_dat_file = dataset_path + ".dat"
        dat_df = cls._read_argonaut_dat_file(arg_dat_file)

        # Read the Argonaut '.snr' file into a DataFrame
        arg_snr_file = dataset_path + ".snr"
        snr_df = cls._read_argonaut_snr_file(arg_snr_file)

        # Read specific configuration values from the Argonaut '.ctl' file into a dictionary.
        arg_ctl_file = dataset_path + ".ctl"
        config_dict = cls._read_argonaut_ctl_file(arg_ctl_file)

        # Combine the '.snr' and '.dat.' DataFrames into a single acoustic DataFrame, make the timestamp
        # the index, and return an instantiated ADVMData object
        # acoustic_df = pd.DataFrame(index=dat_df.index, data=(pd.concat([snr_df, dat_df], axis=1)))
        # acoustic_df.set_index('year', drop=True, inplace=True)
        # acoustic_df.index.names = ['Timestamp']
        acoustic_df = pd.concat([dat_df, snr_df], axis=1)

        data_origin = cls._get_acoustic_data_origin(dataset_path + "(Arg)")

        return cls(config_dict, acoustic_df, data_origin)

    @classmethod
    def read_tab_delimited_data(cls, file_path, params=None):
        """Create an ADVMData object from a tab-delimited text file that contains raw acoustic variables.

        ADVM configuration parameters must be provided as an argument.

        :param file_path: Path to tab-delimited file
        :type file_path: str
        :param params: Dictionary containing necessary ADVM configuration parameters
        :type params: dict

        :return: ADVMData object
        """
        if not isinstance(params, ADVMConfigParam):
            raise TypeError('params must be type data.ADVMConfigParam', params)

        tab_delimited_data = cls._load_tab_delimited_data(file_path)
        acoustic_df = tab_delimited_data.filter(regex=cls._advm_columns_regex)

        data_origin = cls._get_acoustic_data_origin(file_path)

        return cls(params, acoustic_df, data_origin)

    def set_proc_params(self, proc_params):
        """Sets the processing parameters based on user input.

        An example processing parameter dictionary that contains all of the valid keys follows.

            {'Backscatter Values': 'Amp',
             'Beam': 2,
             'Intensity Scale Factor': 0.43,
             'Maximum Cell Mid-Point Distance': inf,
             'Minimum Cell Mid-Point Distance': -inf,
             'Minimum Number of Cells': 2,
             'Minimum Vbeam': -inf,
             'Near Field Correction': True,
             'WCB Profile Adjustment': True}

        :param proc_params: Dictionary containing key, value pairs of processing parameters
        :type proc_params: dict
        :return: None
        """

        if not isinstance(proc_params, dict):
            raise TypeError('proc_params must be type dict')

        for key in proc_params.keys():
            self._proc_param[key] = proc_params[key]

        acoustic_data = self._update_acoustic_data()

        self._data = acoustic_data


