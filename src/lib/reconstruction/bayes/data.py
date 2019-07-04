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
from ..errors import get_errors_per_location


class BayesDFCompute(object):
    def __init__(self):
        self.bayes_columns = ['subject', 'condition_mask', 'occupied',
                              'same', 'opposite',
                              'position_type', 'position_id',
                              'errors_1', 'errors_2', 'errors_3']

    @staticmethod
    def get_error_rates(df, error_type):
        """Compute mean error rates over a dataframe along location axis"""
        return np.stack(df[f'errors_{error_type}'], axis=1).mean(axis=1)

    @staticmethod
    def get_bayes_data(tidy):
        """Extract per-location vectors from tidy df and stack them"""
        x_same = np.concatenate(tidy['adjacency_same'].values)
        x_opposite = np.concatenate(tidy['adjacency_opposite'].values)
        x_occupied = np.concatenate(tidy['occupied'].values)
        x_condition_mask = np.concatenate(tidy['condition_mask'].values)

        x_subject = np.concatenate(tidy['subject_idx'].map(
            lambda x: np.stack([x, ] * 36)).values)

        x_position_type = np.concatenate(tidy['Is Real'].map(
            lambda x: np.stack([int(x), ] * 36)).values)

        x_position_id = np.concatenate(tidy['Position ID'].map(
            lambda x: np.stack([int(x), ] * 36)).values)

        y1 = np.concatenate(tidy['errors_1'].values)
        y2 = np.concatenate(tidy['errors_2'].values)
        y3 = np.concatenate(tidy['errors_3'].values)

        bayes_data = np.stack((x_subject, x_condition_mask, x_occupied,
                               x_same, x_opposite,
                               x_position_type, x_position_id,
                               y1, y2, y3)).T

        return bayes_data

    def __call__(self, tidy, board_set):
        # Compute error arrays at observation and board levels
        g = tidy.groupby('Position ID')

        for i in range(1, 4):
            tidy[f'errors_{i}'] = tidy.apply(
                lambda x: get_errors_per_location(x, str(i)), axis=1)

            # TODO: board_set thing is probably unnecessary here
            board_set[f'errors_{i}'] = g.apply(
                lambda x: self.get_error_rates(x, i))

        bayes_data = self.get_bayes_data(tidy)
        model_df = pd.DataFrame(data=bayes_data,
                                columns=self.bayes_columns)

        model_df['subject'] = model_df['subject'].astype(int)

        # Get integer representation of condition
        indicator_map = {'Trained': 1, 'Untrained': 0}
        indicator = model_df['condition_mask'].map(indicator_map)
        model_df['condition_indicator'] = indicator.astype(int)

        # Get selectors
        trained_sel = model_df['condition_mask'] == 'Trained'
        untrained_sel = model_df['condition_mask'] == 'Untrained'
        natural_sel = model_df['position_type'] == '1'
        synthetic_sel = model_df['position_type'] == '0'

        # Mod subject IDs for use with PyMC3
        untrained_subjects = model_df.loc[untrained_sel, 'subject'] % 19
        model_df.loc[untrained_sel, 'subject'] = untrained_subjects

        # Mod position IDs for use with PyMC3
        natural_ids = model_df.loc[natural_sel, 'position_id'].astype(int)
        natural_ids -= natural_ids.min()
        model_df.loc[natural_sel, 'position_id'] = natural_ids

        position_ids_ = model_df.loc[synthetic_sel, 'position_id'].astype(int)
        unique_positions = np.unique(position_ids_.astype(int).values)
        position_index = np.arange(len(position_ids_.unique()))
        synthetic_id_map = dict(zip(unique_positions, position_index))

        position_ids = position_ids_.map(synthetic_id_map)
        model_df.loc[synthetic_sel, 'position_id'] = position_ids

        return model_df
