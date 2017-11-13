import os
import numpy as np

def get_all_csvs(dirpath):
    """
    Return a list of all the CSV files in a directory

    Arguments
    ---------
    :dirpath is the path of the target directory

    Outputs
    ---------
    Returns a list of CSV files in dirpath
    """

    return [os.path.join(dirpath, f) for f in os.listdir(dirpath) if f[-3:] == 'csv']

def position_string_to_array(position_string):
    return np.array(list(position_string), dtype=int)

def series_to_array(series):
    return np.stack(series.map(position_string_to_array).values, axis=1)
