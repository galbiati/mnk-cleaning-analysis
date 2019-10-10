#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Wei Ji Ma Lab
# By: Gianni Galbiati

import pandas as pd

from .utility_functions import *


def load_file(file_path):
    """Loads file and appends any necessary identifier columns

    This strips the sequences of moves out from the data,
    along with any mousetracking data. A different file loading function
    will be necessary to use any of those sequential items

    NOTE: this function will not work if the files are not contained in
    appropriately named directories

    Arguments
    ---------
    file_path : str
        a file path pointing to a csv file containing clean experiment data

    Returns
    -------
    df : pd.DataFrame
         DataFrame containing relevant fields
    """

    # Throw an error if file is not a csv
    assert file_path[-3:] == 'csv'

    initials = file_path.split('/')[-1].split('.')[0].split('_')[-1]

    # Pretty names for data fields
    col_names = ['Index', 'Subject ID', 'Player Color',
                 'Game Index', 'Move Index', 'Status',
                 'Black Position', 'White Position', 'Action',
                 'Mouse Timestamps', 'Mouse Position']

    # Final data fields
    keep = ['Subject ID', 'Initials', 'Condition', 'Game Index', 'Status',
            'Black Position', 'White Position', 'Response Time']

    df = pd.read_csv(file_path, names=col_names)
    
    # Recompute response times from timestamps
    reconi = df['Status'] == 'reconi'
    reconf = df['Status'] == 'reconf'

    trial_starts = df.loc[reconi, 'Time Stamp'].values
    trial_ends = df.loc[reconf, 'Time Stamp'].values
    df.loc[reconi, 'Response Time'] = trial_ends - trial_starts

    # Only keep the initial and final board states
    df = df.loc[df['Status'].isin(['reconi', 'reconf'])]
    df.reset_index(inplace=True, drop=True)

    # Fix game indices
    df['Game Index'] = df.index // 2

    # Extract training condition from file path
    df['Condition'] = 'Trained' if 'Trained' in file_path else 'Naive'

    # Add initials
    df['Initials'] = initials

    # Filter out irrelevant columns
    df = df[keep]

    return df


def load_data(file_path_list):
    """Load all data into a single dataframe and preprocess.

    Arguments
    ---------
    file_path_list : list
        list of paths to data files

    Returns
    -------
    df : pd.DataFrame
        DataFrame containing relevant fields
    """

    # Load all file paths into individual dataframes and concatenate them
    loaded = [load_file(path) for path in file_path_list]
    df = pd.concat(loaded).reset_index(drop=True)

    # Get cleaner names for status field
    df['Status'] = df['Status'].map({'reconi': 'I', 'reconf': 'F'})
    initial_stim = df['Status'] == 'I'

    # Assign an ID to each unique position
    df['Position ID'] = df['Black Position'] + df['White Position']

    position_map = pd.read_csv('../etc/4 Reconstruction/position_map.csv',
                               index_col=0, skiprows=1,
                               names=['Position', 'Is Real', 'Position ID'])

    df['Is Real'] = df['Position ID'].map(position_map['Is Real']).values
    df['Position ID'] = df['Position ID'].map(position_map['Position ID']).values

    # Group stimulus and final submission into single rows
    for color in ['Black', 'White']:
        final_position = df.loc[~initial_stim, f'{color} Position'].values
        df.loc[initial_stim, f'{color} Position (final)'] = final_position

    # Filter out irrelevant columns
    keep = ['Subject ID', 'Initials', 'Condition', 'Game Index',
            'Position ID', 'Is Real', 'Black Position', 'White Position',
            'Black Position (final)', 'White Position (final)',
            'Response Time']

    df = df.loc[initial_stim, keep]

    return df


def process_data(df):
    """Add auxilliary information and count errors."""
    # TODO: make this a class to retain eg bpi, bpf, wpi, wpf?

    bpi, wpi, bpf, wpf = unpack_positions(df)

    black_errors = (bpf != bpi).astype(int)
    white_errors = (wpf != wpi).astype(int)

    type_1b = ((bpf == 1) & (bpi == 0)).astype(int).sum(axis=0)
    type_1w = ((wpf == 1) & (wpi == 0)).astype(int).sum(axis=0)

    type_2b = ((bpf == 0) & (bpi == 1)).astype(int).sum(axis=0)
    type_2w = ((wpf == 0) & (wpi == 1)).astype(int).sum(axis=0)

    type_3b = ((wpf == 1) & (bpi == 1)).astype(int).sum(axis=0)
    type_3w = ((bpf == 1) & (wpi == 1)).astype(int).sum(axis=0)

    df['Num Black Pieces'] = bpi.sum(axis=0)
    df['Num White Pieces'] = wpi.sum(axis=0)
    df['Num Pieces'] = df['Num Black Pieces'] + df['Num White Pieces']
    df['Total Black Errors'] = black_errors.sum(axis=0)
    df['Total White Errors'] = white_errors.sum(axis=0)
    df['Total Errors'] = np.ceil((black_errors + white_errors) / 2).sum(axis=0)

    df['Type I Errors (black)'] = type_1b - type_3w
    df['Type I Errors (white)'] = type_1w - type_3b
    df['Type I Errors'] = type_1b + type_1w - type_3b - type_3w

    df['Type II Errors (black)'] = type_2b - type_3b
    df['Type II Errors (white)'] = type_2w - type_3w
    df['Type II Errors'] = type_2b + type_2w - type_3b - type_3w

    df['Type III Errors (black)'] = type_3b
    df['Type III Errors (white)'] = type_3w
    df['Type III Errors'] = type_3b + type_3w

    return df


def unpack_positions(df):
    """Converts positions for stimuli and responses to numpy arrays
    """

    fields = ['Black Position', 'White Position',
              'Black Position (final)', 'White Position (final)']

    bpi, wpi, bpf, wpf = [series_to_array(df[field]) for field in fields]
    return bpi, wpi, bpf, wpf
