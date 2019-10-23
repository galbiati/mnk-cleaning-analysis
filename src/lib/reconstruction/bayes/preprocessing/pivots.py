#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Wei Ji Ma Lab
# By: Gianni Galbiati

# Standard libraries

# External libraries
import numpy as np
import pandas as pd

# Internal libraries
from ....reconstruction.errors import get_errors_per_location
from ....reconstruction.neighbors import get_adjacency_per_location, get_color_per_location


def load_tidy(tidy_path):
    """Load dataframe with pre-computed metrics in trial-wise tidy format."""

    # Read dataframe
    tidy_df = pd.read_csv(tidy_path, index_col=0)

    # Rename "Naive" values for conditions
    tidy_df['Condition'] = tidy_df['Condition'].map(lambda x: 'Untrained' if x == 'Naive' else x)

    # Force position to be integer
    tidy_df['Position ID'] = tidy_df['Position ID'].map(int)

    # Pivot table at per-board level
    # to efficiently compute and index board properties
    board_values = ['Black Position', 'White Position', 'Is Real', 'Num Pieces']

    board_set = tidy_df.pivot_table(index='Position ID', values=board_values,
                                    aggfunc=lambda x: x.unique()[0])

    board_set = board_set[board_values]

    # Compute adjacency statistics at each location
    adjacency_column_names = [f'adjacency_{v}' for v in ('all', 'same', 'opposite')]

    adjacencies = board_set.apply(get_adjacency_per_location, axis=1)

    adjacency_df = pd.DataFrame(adjacencies.tolist(),
                                index=board_set.index, columns=adjacency_column_names)

    # Expand colors over board locations
    colors = board_set.apply(get_color_per_location, axis=1)
    colors_series = pd.Series(colors.tolist(), index=board_set.index, name='colors')

    # Compute errors at each board location
    for error_type in range(1, 4):
        tidy_df[f'errors_{error_type}'] = tidy_df.apply(lambda x: get_errors_per_location(x, str(error_type)), axis=1)

    # Compute occupancy and condition indicators
    # for each location on each trial
    tidy_df['occupied'] = tidy_df.apply(_get_occupied_mask, axis=1)
    tidy_df['condition_mask'] = tidy_df['Condition'].map(_get_condition_mask)

    # For convenience, pull per-location statistics for each trial
    # into the per-trial dataframe
    tidy_df['adjacency_same'] = tidy_df['Position ID'].map(adjacency_df['adjacency_same'])
    tidy_df['adjacency_opposite'] = tidy_df['Position ID'].map(adjacency_df['adjacency_opposite'])
    tidy_df['colors'] = tidy_df['Position ID'].map(colors_series)

    # Convert the subject UID to an integer index
    subject_ids = tidy_df['Subject ID'].unique()
    subject_index = np.arange(len(subject_ids))
    subject_index_map = dict(zip(subject_ids, subject_index))

    tidy_df['subject_idx'] = tidy_df['Subject ID'].map(subject_index_map)

    # Per-position mean error rates (over subjects)
    g = tidy_df.groupby('Position ID')
    board_set['errors'] = g.apply(_get_error_rates)

    # Get a dummy array of location indices for convenience
    board_set['location_idx'] = np.tile(np.arange(36, dtype=np.uint8),
                                        [len(board_set), 1]).tolist()

    # Get distances to center as a dummy field
    # (same for all positions!)
    blank_board = np.zeros((4, 9))
    center = (blank_board.shape[0] / 2 - .5, blank_board.shape[1] / 2 - .5)

    distances = ((np.argwhere(blank_board == 0) - center) ** 2).sum(axis=1)
    distances = np.sqrt(distances)

    board_set['distance_to_center'] = np.tile(distances, [len(board_set), 1]).tolist()

    return tidy_df, board_set


def compute_extra_tidy(tidy_df):
    """Convert a trial-wise dataframe into a trial-location-wise dataframe."""

    # Extract exogenous variables
    x_same = np.concatenate(tidy_df['adjacency_same'].values)
    x_opposite = np.concatenate(tidy_df['adjacency_opposite'].values)
    x_occupied = np.concatenate(tidy_df['occupied'].values)
    x_condition_mask = np.concatenate(tidy_df['condition_mask'].values)
    x_colors = np.concatenate(tidy_df['colors'].values)

    x_subject = np.concatenate(tidy_df['subject_idx'].map(_expand_indicators).values)
    x_position_type = np.concatenate(tidy_df['Is Real'].map(_expand_indicators).values)
    x_position_id = np.concatenate(tidy_df['Position ID'].map(_expand_indicators).values)

    # Extract endogenous variables
    y1 = np.concatenate(tidy_df['errors_1'].values)
    y2 = np.concatenate(tidy_df['errors_2'].values)
    y3 = np.concatenate(tidy_df['errors_3'].values)

    columns = ['subject', 'condition_mask', 'occupied', 'color',
               'same', 'opposite',
               'position_type', 'position_id',
               'errors_1', 'errors_2', 'errors_3']

    df_data = np.stack((x_subject, x_condition_mask, x_occupied, x_colors,
                        x_same, x_opposite,
                        x_position_type, x_position_id,
                        y1, y2, y3)).T

    extra_tidy_df = pd.DataFrame(data=df_data, columns=columns)

    # Force pandas to use integers.
    for c in extra_tidy_df.columns:
        if c not in ['condition_mask', 'same', 'opposite', 'color']:
            extra_tidy_df[c] = extra_tidy_df[c].astype(int)

    extra_tidy_df['condition_indicator'] = extra_tidy_df.condition_mask.map({'Trained': 1, 'Untrained': 0}).astype(int)

    # Set up condition and position type indicators
    trained_sel = extra_tidy_df['condition_mask'] == 'Trained'
    untrained_sel = extra_tidy_df['condition_mask'] == 'Untrained'
    natural_sel = extra_tidy_df['position_type'] == 1
    synthetic_sel = extra_tidy_df['position_type'] == 0

    # Convert subject indices for columnar format
    # Eg, indexes start from 0 for both trained/untrained groups

    extra_tidy_df.loc[untrained_sel, 'subject'] = extra_tidy_df.loc[untrained_sel, 'subject'] % 19

    # But still cache a unique indicator for each subject
    extra_tidy_df['usubject'] = extra_tidy_df['subject']
    extra_tidy_df['usubject'] += extra_tidy_df['condition_indicator'] * 19

    # Same for real/fake positions
    natural_ids = extra_tidy_df.loc[natural_sel, 'position_id'].astype(int)
    extra_tidy_df.loc[natural_sel, 'position_id'] = natural_ids - natural_ids.min()

    # Get an integer indicator for the *type* of error that was made
    # Where a 0 indicates no error
    error_columns = ['errors_1', 'errors_2', 'errors_3']
    extra_tidy_df['has_error'] = extra_tidy_df[error_columns].astype(int).sum(axis=1)

    error_sel = extra_tidy_df['has_error'] == 1

    error_values = extra_tidy_df.loc[error_sel, error_columns].values
    error_types = np.argmax(error_values, axis=1) + 1

    extra_tidy_df['error_type'] = 0
    extra_tidy_df.loc[error_sel, 'error_type'] = error_types

    return extra_tidy_df


def compute_per_trial_pivot(extra_tidy_df):
    per_trial_pivot = extra_tidy_df.pivot_table(
        index=('usubject', 'position_id'),
        values=('errors_2', 'occupied'),
        aggfunc=np.sum)

    per_trial_pivot['position_type'] = extra_tidy_df.pivot_table(
        index=('usubject', 'position_id'), values='position_type',
        aggfunc=lambda x: x.unique()[0])

    per_trial_pivot['condition_indicator'] = extra_tidy_df.pivot_table(
        index=('usubject', 'position_id'), values='condition_indicator',
        aggfunc=lambda x: x.unique()[0])

    per_trial_pivot['usubject'] = per_trial_pivot.index.get_level_values(
        'usubject')
    per_trial_pivot['position_id'] = per_trial_pivot.index.get_level_values(
        'position_id')

    per_trial_pivot.reset_index(drop=True, inplace=True)

    per_trial_pivot['interaction'] = per_trial_pivot.apply(
        lambda row: str(row['position_type']) + str(
            row['condition_indicator']),
        axis=1
    )

    return per_trial_pivot


def compute_per_subject_pivot(extra_tidy_df):
    piv_idx_ = ('usubject', 'position_type', 'condition_indicator',)
    piv_vals_ = ('errors_2', 'occupied')
    per_subject_pivot = extra_tidy_df.pivot_table(index=piv_idx_, values=piv_vals_, aggfunc=np.sum)

    per_subject_pivot['usubject'] = per_subject_pivot.index.get_level_values('usubject')
    per_subject_pivot['position_type'] = per_subject_pivot.index.get_level_values('position_type')
    per_subject_pivot['condition_indicator'] = per_subject_pivot.index.get_level_values('condition_indicator')
    per_subject_pivot.reset_index(drop=True, inplace=True)

    per_subject_pivot['interaction'] = per_subject_pivot.apply(lambda row: str(row['position_type']) + str(row['condition_indicator']), axis=1)

    return per_subject_pivot


# Utility functions
def _expand_indicators(x):
    return np.stack([int(x), ] * 36)


def _get_error_rates(df):
    """(DEPRECATED?) Return type 2 average error rates in a vector"""
    return np.stack(df['errors_2'], axis=1).mean(axis=1)


def _get_occupied_mask(row):
    """Return indicators for whether a piece was at a location"""
    bp = np.stack([int(i) for i in row['Black Position']])
    wp = np.stack([int(i) for i in row['White Position']])
    p = bp + wp
    return p.tolist()


def _get_condition_mask(condition):
    """Return convenience indicators for condition"""
    return [condition, ] * 36