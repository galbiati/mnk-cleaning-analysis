#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Wei Ji Ma Lab
# By: Gianni Galbiati

# Standard libraries
import sys

from argparse import ArgumentParser

# External libraries
import numpy as np
import pandas as pd
import patsy as pt
import pymc3 as pm

from theano import tensor as T

# Internal libraries
from src.lib.reconstruction.bayes.preprocessing.pivots import (
    load_tidy, compute_extra_tidy)

from src.lib.reconstruction.features import count_all_features


def load_data():
    tidy_df, board_set_df = load_tidy('../etc/reconstruction/tidy_data.csv')

    # One position was duplicated between real/fake positions;
    # drop it from data

    valid_ids = tidy_df.pivot_table(index='Position ID', values='Subject ID', aggfunc=len)
    valid_ids = valid_ids.loc[valid_ids['Subject ID'] == 38]
    valid_ids = valid_ids.index.tolist()

    tidy_df = tidy_df.loc[tidy_df['Position ID'].isin(valid_ids)]
    board_set_df = board_set_df.loc[valid_ids]
    board_set_df.sort_index(inplace=True)
    board_set_df.reset_index(inplace=True, drop=True)

    upids = tidy_df.sort_values('Position ID')['Position ID'].unique()
    pid_map = dict(zip(upids, np.arange(0, len(upids), 1, dtype=int)))
    tidy_df['Position ID'] = tidy_df['Position ID'].map(pid_map)

    extra_tidy_df = compute_extra_tidy(tidy_df)

    # Add features
    all_features = tidy_df.apply(count_all_features, axis=1)

    # TODO: hardcode these instead
    feature_names = list(all_features.iloc[0].keys())
    feature_base_names = list(set([fn[:-1] for fn in feature_names]))

    for fn in feature_names:
        extra_tidy_df[f'feature_{fn}'] = np.concatenate(all_features.map(lambda x: x[fn]).values)

    for bn in feature_base_names:
        b = np.concatenate(all_features.map(lambda x: x[f'{bn}b']).values)
        w = np.concatenate(all_features.map(lambda x: x[f'{bn}w']).values)
        extra_tidy_df[f'basef_{bn}'] = b + w

    extra_tidy_df['same'] = extra_tidy_df['same'].astype(float)
    extra_tidy_df['opposite'] = extra_tidy_df['opposite'].astype(float)

    extra_tidy_df['n_pieces'] = extra_tidy_df.position_id.map(board_set_df['Num Pieces'])

    extra_tidy_df['same'] = extra_tidy_df['same'].astype(float)
    extra_tidy_df['opposite'] = extra_tidy_df['opposite'].astype(float)

    return extra_tidy_df, feature_base_names


def get_factors(df, error_type, feature_names):

    # Set up patsy formula

    # Main effects
    formula = f'{error_type} ~ '
    formula += 'C(condition_indicator, Sum) * C(position_type, Sum) + '
    formula += 'n_pieces + same + opposite + C(color, Sum) + '
    formula += ' + '.join(f'basef_{bn}' for bn in feature_names)
    formula += ' + C(usubject, Sum)'

    # Other interactions
    formula += ' + ' + ' + '.join(f'basef_{bn}:C(condition_indicator, Sum)' for bn in feature_names)
    formula += ' + ' + 'C(condition_indicator, Sum):n_pieces'

    # Filter for occupied/unoccupied
    df = df.copy()
    df = df.loc[df['occupied'] == 1]
    df['n_pieces'] = df.n_pieces - df.n_pieces.max() + .5 * (df.n_pieces.max() - df.n_pieces.min())

    # Get indicator dataframes
    exogenous_df, endogenous_df = pt.dmatrices(formula, df, return_type='dataframe', NA_action='raise')

    # Convert some columns to integer datatype
    index_cols = ['Intercept',
                  'C(condition_indicator, Sum)[S.0]',
                  'C(position_type, Sum)[S.0]',
                  'C(condition_indicator, Sum)[S.0]:C(position_type, Sum)[S.0]']

    index_cols += [col for col in endogenous_df.columns if 'usubject' in col]

    for c in index_cols:
        if c in endogenous_df.columns:
            endogenous_df[c] = endogenous_df[c].astype(int)

    return exogenous_df, endogenous_df


def get_coefficients(df):
    features = [c for c in df.columns
                if 'basef' in c and 'color' not in c and 'condition_indicator' not in c]

    feature_x_condition = [c for c in df.columns
                           if 'basef' in c and 'condition_indicator' in c]


    coefficients = {'condition': ['C(condition_indicator, Sum)[S.0]'],
                    'position': ['C(position_type, Sum)[S.0]'],
                    'interaction': ['C(condition_indicator, Sum)[S.0]:C(position_type, Sum)[S.0]'],
                    'n_pieces': ['n_pieces'],
                    'same': ['same'],
                    'opposite': ['opposite'],
                    'color': ['C(color, Sum)[S.-1]'],
                    'features': features,
                    'features_x_condition': feature_x_condition,
                    'condition_x_n_pieces': ['C(condition_indicator, Sum)[S.0]:n_pieces']}

    return coefficients


def create_model(endogenous_df, exogenous_df, error_type, use_beta=False):
    coefficients = get_coefficients(endogenous_df)

    def create_coeff(name, shape, mu=0):
        a = pm.Normal(f'a_{name}', mu=mu, sigma=1, shape=shape)
        return a

    with pm.Model() as logistic_anova:
        # Create intercept coefficient; no prior on variance
        b0 = pm.Normal('intercept', mu=0, sd=1, shape=[1])
        coeffs = {k: create_coeff(k, [len(v)])
                  for k, v in coefficients.items()}

        mu = [coeffs[k] * endogenous_df[v].values
              for k, v in coefficients.items()]

        mu = b0 + T.sum(T.concatenate(mu, axis=1), axis=1)

        mu = pm.invlogit(mu)

        if use_beta:
            kappa = pm.Gamma('beta_variance', .01, .01)
            alpha = mu * (kappa - 2) + 1
            beta = (1 - mu) * (kappa - 2) + 1

            theta = pm.Beta('theta', alpha=alpha, beta=beta, shape=exogenous_df.shape[0])

            mu = theta

        y = pm.Bernoulli('targets', p=mu, observed=exogenous_df[error_type].values)

    return logistic_anova


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('error_type', type=int,
                        help="Target error type for logistic regression model.")
    parser.add_argument('--num_chains', '-c', type=int, default=4,
                        help="Number of MCMC chains to run")
    parser.add_argument('')

    return parser.parse_args()