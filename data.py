import re
import copy
import abc

import pandas as pd
import numpy as np


class DataException(Exception):
    """Base class for all exceptions in the data module"""
    pass


class ADVMDataIncompatibleError(DataException):
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
        :return: Nothing
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
    """
    Stores ADVM Configuration parameters.

    """

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

    def get_num_observations(self):

        num_observations = self._acoustic_df.shape[0]

        return num_observations

    def set_proc_params(self, proc_params):
        """Sets the processing parameters based on user input.

        :param proc_params: Dictionary containing configuration parameters
        :return: Nothing
        """

        for key in proc_params.keys():
            self._proc_param[key] = proc_params[key]

        return

    def get_config_params(self):
        """Return dictionary containing configuration parameters.

        :return: Dictionary containing configuration parameters
        """

        return self._config_param.get_dict()

    def get_proc_params(self):
        """Return dictionary containing processing parameters.

        :return: Dictionary containing processing parameters
        """

        return self._proc_param.get_dict()

    def get_mean_scb(self):
        """Return mean sediment corrected backscatter. Throw exception if all required variables have not been provided.

        :return: Mean sediment corrected backscatter for all observations contained in acoustic_df
        """

        scb = self.get_scb()

        return pd.DataFrame(scb.mean(axis=1), columns=['MeanSCB'])

    def get_cell_range(self):
        """Calculate range of cells along a single beam.

        :return: Range of cells along a single beam
        """

        blanking_distance = self._config_param['Blanking Distance']
        cell_size = self._config_param['Cell Size']
        number_of_cells = self._config_param['Number of Cells']

        first_cell_mid_point = blanking_distance + cell_size / 2
        last_cell_mid_point = first_cell_mid_point + (number_of_cells - 1) * cell_size

        slant_angle = self._config_param['Slant Angle']

        cell_range = np.linspace(first_cell_mid_point,
                                 last_cell_mid_point,
                                 num=number_of_cells) / np.cos(np.radians(slant_angle))

        cell_range = np.tile(cell_range, (self.get_num_observations(), 1))

        col_names = ['R{:03}'.format(cell) for cell in range(1, number_of_cells+1)]

        cell_range_df = pd.DataFrame(data=cell_range, index=self._acoustic_df.index, columns=col_names)

        return cell_range_df

    def get_mb(self):
        """Calculate measured backscatter values based on processing parameters.

        :return: DataFrame containing measured backscatter values
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

            for beam in range(1, number_of_beams+1):

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
            backscatter_df = scale_factor*backscatter_df

        return backscatter_df

    def get_sac(self):
        """Calculate sediment attenuation coefficient. Throw exception if all required variables have not been provided.

        :return: Sediment attenuation coefficient for all observations in acoustic_df
        """

        wcb = self.get_wcb()
        cell_range = self.get_cell_range()

        wcb_arr = wcb.as_matrix()
        cell_range_arr = cell_range.as_matrix()

        sac_arr = self._calculate_sac(wcb_arr, cell_range_arr)

        sac = pd.DataFrame(index=wcb.index, data=sac_arr, columns=['SAC'])

        return sac

    def get_scb(self):
        """Calculate sediment corrected backscatter. Throw exception if all required variables have not been provided.

        :return: Sediment corrected backscatter for all cells in the acoustic time series
        """

        wcb = self.get_wcb()
        wcb_arr = wcb.as_matrix()

        cell_range = self.get_cell_range().as_matrix()

        sac = self.get_sac().as_matrix()
        sac = np.tile(sac, (1, cell_range.shape[1]))

        scb_arr = wcb_arr + 2*sac*cell_range

        scb_columns = ['SCB{:03}'.format(cell) for cell in range(1, self._config_param['Number of Cells'] + 1)]

        scb_df = pd.DataFrame(index=wcb.index, data=scb_arr, columns=scb_columns)

        return scb_df

    def get_wcb(self):
        """Calculate water corrected backscatter (WCB). Throw exception if all required variables have not been provided

        :return: Water corrected backscatter for all cells in the acoustic time series
        """

        spreading_loss = self._calculate_spreading_loss()

        water_absorption_loss = self._calculate_water_absorption_loss()

        two_way_transmission_loss = spreading_loss + water_absorption_loss  # two-way transmission loss
        measured_backscatter_df = self.get_mb()  # measured backscatter values

        water_corrected_backscatter = np.array(measured_backscatter_df.as_matrix() + two_way_transmission_loss)

        # adjust the water corrected backscatter profile
        if self._proc_param['WCB Profile Adjustment']:

            water_corrected_backscatter = self._apply_minwcb_correction(water_corrected_backscatter)

        water_corrected_backscatter = self._remove_min_vbeam(water_corrected_backscatter)
        water_corrected_backscatter = self._apply_cell_range_filter(water_corrected_backscatter)

        # create a dataframe of the water corrected backscatter observations
        wcb_columns = ['WCB{:03}'.format(cell) for cell in range(1, self._config_param['Number of Cells']+1)]
        wcb = pd.DataFrame(index=measured_backscatter_df.index, data=water_corrected_backscatter, columns=wcb_columns)

        return wcb

    def merge(self, other):
        """Merges self and other ADVMData objects. Throws exception if other ADVMData object is incompatible with self.

        :param other: ADVMData object to be merged with self
        :return: Merged ADVMData object
        """

        if self._config_param.is_compatible(other.get_config_params()):

            # cast as data frame to keep pycharm from complaining
            new_acoustic_df = pd.DataFrame(pd.concat([self._acoustic_df, other._acoustic_df]))

            new_advm_data = ADVMData(self._config_param, new_acoustic_df)
            new_advm_data.set_proc_params(self.get_proc_params())

            return new_advm_data

        else:

            raise ADVMDataIncompatibleError("ADVM data sets are incompatible")

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
        for cell in range(1, number_of_cells+1):

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

    def _apply_cell_range_filter(self, water_corrected_backscatter):
        """

        :param water_corrected_backscatter:
        :return:
        """

        max_cell_range = self._proc_param['Maximum Cell Mid-Point Distance']
        min_cell_range = self._proc_param['Minimum Cell Mid-Point Distance']

        cell_range = self.get_cell_range().as_matrix()

        cells_outside_set_range = (max_cell_range < cell_range) | (cell_range < min_cell_range)

        water_corrected_backscatter[cells_outside_set_range] = np.nan

        return water_corrected_backscatter

# TODO refactor so helper methods are static

    def _calculate_spreading_loss(self):
        """Calculate the two-way transmission loss due to spherical beam spreading

        :return: DataFrame containing the time series for the spreading loss
        """

        cell_range = self.get_cell_range()

        if self._proc_param['Near Field Correction']:

            psi = self._calculate_psi()

            spreading_loss = 20 * np.log10(psi.as_matrix() * cell_range.as_matrix())

        else:

            spreading_loss = 20 * np.log10(cell_range.as_matrix())

        return spreading_loss

    def _calculate_water_absorption_loss(self):
        """Calculate the two-way acoustic attenuation due to water absorption for all cells along the beam

        :return: DataFrame containing the time series for water absorption
        """

        tmp_alpha_w = self._calculate_alpha_w()
        tmp_alpha_w = tmp_alpha_w.reshape((self.get_num_observations(), 1))

        num_cells = self._config_param['Number of Cells']

        alpha_w = np.tile(tmp_alpha_w, (1, num_cells))

        water_absorption_loss = 2*alpha_w*self.get_cell_range()

        return water_absorption_loss.as_matrix()

    def _calculate_alpha_w(self):
        """Calculate alpha_w - the water-absorption coefficient (WAC) in dB/m.

        :return: alpha_w
        """

        ADVMTemp = self._acoustic_df['Temp']
        frequency = self._config_param['Frequency']

        f_T = 21.9 * 10 ** (6 - 1520 / (ADVMTemp + 273))  # temperature-dependent relaxation frequency
        alpha_w = 8.686 * 3.38e-6 * (frequency ** 2) / f_T  # water attenuation coefficient

        return alpha_w.rename('alpha_w')

    @staticmethod
    def _calculate_sac(wcb_arr, range_arr):
        """Calculate the sediment attenuation coefficient (SAC) in dB/m.

        :return: sac array
        """

        # make sure both matrices are the same shape
        assert wcb_arr.shape == range_arr.shape

        # create an empty nan matrix for the SAC
        sac_arr = np.tile(np.nan, (wcb_arr.shape[0],))

        # iterate over all rows
        for row_index in np.arange(wcb_arr.shape[0]):

            # find the column indices that are valid
            fit_index = ~np.isnan(wcb_arr[row_index, :])

            # if there are less than one valid cell, skip this observation
            if np.sum(fit_index) < 1:
                continue

            # find the slope of the WCB with respect to the cell range
            x = range_arr[row_index, fit_index]
            y = wcb_arr[row_index, fit_index]
            xy_mean = np.mean(x * y)
            x_mean = np.mean(x)
            y_mean = np.mean(y)
            cov_xy = xy_mean - x_mean * y_mean
            var_x = np.var(x)
            slope = cov_xy/var_x

            # calculate the SAC
            sac_arr[row_index] = -0.5 * slope

        return sac_arr

    def _calculate_rcrit(self):
        """Calculate Rcrit - the critical range (aka near zone distance), in meters.

        :return: Rcrit
        """

        ADVMTemp = self._acoustic_df['Temp']

        # speed of sound in water (m/s) (Marczak 1997)
        c = 1.402385 * 10 ** 3 + 5.038813 * ADVMTemp - \
            (5.799136 * 10 ** -2) * ADVMTemp ** 2 + \
            (3.287156 * 10 ** -4) * ADVMTemp ** 3 - \
            (1.398845 * 10 ** -6) * ADVMTemp ** 4 + \
            (2.787860 * 10 ** -9) * ADVMTemp ** 5

        wavelength = c / (self._config_param['Frequency'] * 1e3)  # in meters (m)
        at = self._config_param['Effective Transducer Diameter'] / 2

        # cast to keep PyCharm from complaining
        # Rcrit = pd.DataFrame((np.pi * (at ** 2)) / wavelength)
        Rcrit = (np.pi * (at ** 2)) / wavelength

        return Rcrit.rename('Rcrit')

    def _calculate_psi(self):
        """Calculate psi - the near field correction coefficient.

        :return: psi
        """

        # "function which accounts for the departure of the backscatter signal from spherical spreading
        #  in the near field of the transducer" Downing (1995)

        critical_range = self._calculate_rcrit()
        cell_range = self.get_cell_range()

        def div_r_by_rcrit(r):
            return r/critical_range

        # Zz = np.outer(1/critical_range, cell_range)  # normalized range dependence
        Zz = cell_range.apply(div_r_by_rcrit)
        # cast to keep PyCharm from complaining
        psi = pd.DataFrame((1 + 1.35 * Zz + (2.5 * Zz) ** 3.2) / (1.35 * Zz + (2.5 * Zz) ** 3.2))

        number_of_cells = self._config_param['Number of Cells']

        col_names = ['Psi{:03}'.format(cell) for cell in range(1, number_of_cells + 1)]

        return psi.rename(columns=dict(zip(psi.keys(), col_names)))


class RawAcousticDataContainer:
    """Container for raw acoustic data type. The unique
    identifier for an acoustic data type object is the frequency
    of the instrument.
    """

    def __init__(self):
        """Initialize AcousticDataContainer object."""

        # initialize _acoustic_data as empty dictionary
        self._acoustic_data = {}

    def add_data(self, new_advm_data):
        """Add acoustic data to container."""

        # get the frequency of the ADVM data set
        frequency = new_advm_data.get_config_params()['Frequency']

        # if a data set with the frequency is already loaded,
        if frequency in self._acoustic_data.keys():
            self._acoustic_data[frequency].merge(new_advm_data)

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
