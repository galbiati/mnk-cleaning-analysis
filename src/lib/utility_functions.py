import os
import numpy as np


def get_all_csvs(directory_path):
    """Return a list of all the CSV files in a directory.

    TODO: should replace the below with glob like a sane person.

    Arguments
    ---------
    directory_path : str
        path to directory of CSV files

    Returns
    -------
    list
        list of CSV file paths in directory
    """

    return [os.path.join(directory_path, f)
            for f in os.listdir(directory_path)
            if f[-3:] == 'csv']


def position_string_to_array(position_string):
    """Convert a position string representation to a numpy array."""
    return np.array(list(position_string), dtype=int)


def series_to_array(series):
    """Convert a pandas series of string representations to a numpy array."""
    return np.stack(series.map(position_string_to_array).values, axis=1)
