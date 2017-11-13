import os
import numpy as np
import pandas as pd

from .util import *

def load_file(filepath):
    """
    Loads a single data file into a dataframe and
    appends any necessary identifier columns

    This strips the sequences of moves out from the data,
    along with any mousetracking data. A different file loading function
    will be necessary to use any of those sequential items

    NOTE: this function will not work if the files are not contained in
    appropriately named directories

    Arguments:
    ----------
    :filepath is a complete relative or absolute filepath pointing to a csv file

    Outputs:
    ----------
    :DF[keep] is a pandas DataFrame containing the relevant fields
    """

    assert filepath[-3:] == 'csv'                                               # throw an error if not a csv

    # pretty names for data fields
    col_names = [
        'Index', 'Subject ID', 'Player Color',
        'Game Index', 'Move Index', 'Status',
        'Black Position', 'White Position', 'Action',
        'Response Time', 'Time Stamp',
        'Mouse Timestamps', 'Mouse Position'
    ]

    # final data fields
    keep = [
        'Subject ID', 'Condition', 'Game Index', 'Status',
        'Black Position', 'White Position'
    ]

    DF = pd.read_csv(filepath, names=col_names)                                 # load the file with pandas
    DF = DF.loc[DF['Status'].isin(['reconi', 'reconf'])].reset_index(drop=True) # only keep initial and final board states
    DF['Game Index'] = DF.index // 2                                            # fix game indexes
    DF['Condition'] = 'Trained' if 'Trained' in filepath else 'Naive'           # get condition from filepath

    return DF[keep]


def load_data(filepaths):
    """
    Loads all data into a single dataframe
    and does some additional preprocessing

    TODO: add real/fake identifiers for each position!

    Arguments:
    ----------
    :filepaths is a list of complete relative or absolute filepaths pointing to
               csv files

    Outputs:
    ----------
    :DFi[keep] is a pandas DataFrame with the relevant columns
    """

    # get all data into a single dataframe
    loaded = [load_file(path) for path in filepaths]                            # load all files in filepaths into individual dataframes
    DF = pd.concat(loaded).reset_index(drop=True)                               # glue 'em

    DF['Status'] = DF['Status'].map({'reconi': 'I', 'reconf': 'F'})             # pretty names for status

    # assign an ID to every unique position
    # NOTE: could make this slightly more efficient by melting first
    DF['Position ID'] = DF['Black Position'] + DF['White Position']
    first_subject = DF['Subject ID'] == DF['Subject ID'].unique()[0]
    initial_stim = DF['Status'] == 'I'
    position_index = DF.loc[first_subject & initial_stim, 'Position ID']
    position_map = dict(zip(position_index, np.arange(len(position_index))))
    DF['Position ID'] = DF['Position ID'].map(position_map)

    # "melt" initial and final observations (stimulus and final submission) into single rows
    DF.loc[initial_stim, 'Black Position (final)'] = DF.loc[~initial_stim, 'Black Position'].values
    DF.loc[initial_stim, 'White Position (final)'] = DF.loc[~initial_stim, 'White Position'].values

    # only keep what we're actively using
    keep = [
        'Subject ID', 'Condition', 'Game Index',
        'Position ID', 'Black Position', 'White Position',
        'Black Position (final)', 'White Position (final)'
    ]

    return DF.loc[initial_stim, keep]

def process_data(DF):
    """
    Processes data, adding auxilliary information and counting errors
    of various types

    TODO: make this a class to retain eg bpi, bpf, wpi, wpf
    """

    bpi = series_to_array(DF['Black Position'])
    bpf = series_to_array(DF['Black Position (final)'])
    wpi = series_to_array(DF['White Position'])
    wpf = series_to_array(DF['White Position (final)'])

    black_errors = (bpf != bpi).astype(int)
    white_errors = (wpf != wpi).astype(int)

    type_1b = ((bpf == 1) & (bpi == 0)).astype(int).sum(axis=0)
    type_1w = ((wpf == 1) & (wpi == 0)).astype(int).sum(axis=0)

    type_2b = ((bpf == 0) & (bpi == 1)).astype(int).sum(axis=0)
    type_2w = ((wpf == 0) & (wpi == 1)).astype(int).sum(axis=0)

    type_3b = ((wpf == 1) & (bpi == 1)).astype(int).sum(axis=0)
    type_3w = ((bpf == 1) & (wpi == 1)).astype(int).sum(axis=0)

    DF['Num Black Pieces'] = bpi.sum(axis=0)
    DF['Num White Pieces'] = wpi.sum(axis=0)
    DF['Num Pieces'] = DF['Num Black Pieces'] + DF['Num White Pieces']
    DF['Total Black Errors'] = black_errors.sum(axis=0)
    DF['Total White Errors'] = white_errors.sum(axis=0)
    DF['Total Errors'] = np.ceil((black_errors + white_errors) / 2).sum(axis=0)

    DF['Type I Errors (black)'] = type_1b - type_3w
    DF['Type I Errors (white)'] = type_1w - type_3b
    DF['Type I Errors'] = type_1b + type_1w - type_3b - type_3w

    DF['Type II Errors (black)'] = type_2b - type_3b
    DF['Type II Errors (white)'] = type_2w - type_3w
    DF['Type II Errors'] = type_2b + type_2w - type_3b - type_3w

    DF['Type III Errors (black)'] = type_3b
    DF['Type III Errors (white)'] = type_3w
    DF['Type III Errors'] = type_3b + type_3w
    return DF
