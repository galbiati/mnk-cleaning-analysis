import os

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

    return [os.path.join(dirpath, f) for f in os.listdir(dirname) if f[-3:] == 'csv']
