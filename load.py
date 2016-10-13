import pandas as pd
import linecache
import math
import os
from data import ADVMData


def load_argonaut_data(data_path, filename):
    """
    Loads a Argonaut File into an ADVMData class object.

    :param data_path: file path containing the Argonaut data files
    :param filename: root filename for the 3 Argonaut files
    :return: ADVMData object containing the Argonaut dataset information
    """

    # Read the Argonaut '.dat' file into a DataFrame
    arg_dat_file = os.path.join(data_path, filename + ".dat")
    dat_df = _read_argonaut_dat_file(arg_dat_file)

    # Read the Argonaut '.snr' file into a DataFrame
    arg_snr_file = os.path.join(data_path, filename + ".snr")
    snr_df = _read_argonaut_snr_file(arg_snr_file)

    # Read specific configuration values from the Argonaut '.ctl' file into a dictionary.
    arg_ctl_file = os.path.join(data_path, filename + ".ctl")
    config_dict = _read_argonaut_ctl_file(arg_ctl_file)

    # Combine the '.snr' and '.dat.' DataFrames into a single acoustic DataFrame, make the timestamp
    # the index, and return an instantiated ADVMData object
    # acoustic_df = pd.DataFrame(index=dat_df.index, data=(pd.concat([snr_df, dat_df], axis=1)))
    # acoustic_df.set_index('year', drop=True, inplace=True)
    # acoustic_df.index.names = ['Timestamp']
    acoustic_df = pd.concat([dat_df, snr_df], axis=1)

    return ADVMData(config_dict, acoustic_df)


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

    dat_df.apply(pd.to_numeric)

    return dat_df


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

    snr_df = snr_df.apply(pd.to_numeric)

    return snr_df


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
    elif math.isnan(float(frequency[0])):
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


def load_tab_delimited_data(data_path, filename):
    """
    Loads a TAB-delimited ASCII File into an ADVMData class object.

    :param data_path: file path containing the TAB-delimited ASCII data file
    :param filename: root filename for the TAB-delimited ASCII file
    :return: DataFrame object containing the ASCII file dataset information
    """

    # Read TAB-delimited txt file into a DataFrame.
    tab_delimited_file = os.path.join(data_path, filename + ".txt")
    tab_delimited_df = pd.read_table(tab_delimited_file, sep='\t')

    # Check the formatting of the date/time columns. If one of the correct formats is used, reformat
    # those date/time columns into a new timestamp column. If none of the correct formats are used,
    # return an invalid file format error to the user.
    if 'y' and 'm' and 'd' and 'H' and 'M' and 'S' in tab_delimited_df.columns:
        tab_delimited_df.rename(columns={"y": "year", "m": "month", "d": "day"}, inplace=True)
        tab_delimited_df.rename(columns={"H": "hour", "M": "minute", "S": "second"}, inplace=True)
        tab_delimited_df["year"] = pd.to_datetime(tab_delimited_df[["year", "month", "day", "hour",
                                                                    "minute", "second"]], errors="coerce")
        tab_delimited_df.rename(columns={"year": "Timestamp"}, inplace=True)
        tab_delimited_df.drop(["month", "day", "hour", "minute", "second"], axis=1, inplace=True)
    elif 'Date' and 'Time' in tab_delimited_df.columns:
        tab_delimited_df["Date"] = pd.to_datetime(tab_delimited_df["Date"] + " " + tab_delimited_df["Time"],
                                                  errors="coerce")
        tab_delimited_df.rename(columns={"Date": "Timestamp"}, inplace=True)
        tab_delimited_df.drop(["Time"], axis=1, inplace=True)
    elif 'DateTime' in tab_delimited_df.columns:
        tab_delimited_df.rename(columns={"DateTime": "Timestamp"}, inplace=True)
        tab_delimited_df["Timestamp"] = pd.to_datetime(tab_delimited_df["Timestamp"], errors="coerce")
    else:
        raise ValueError("Date and time information is incorrectly formatted.", tab_delimited_file)

    tab_delimited_df.set_index('Timestamp', drop=True, inplace=True)

    return tab_delimited_df
