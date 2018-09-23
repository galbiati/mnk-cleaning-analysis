#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Wei Ji Ma Lab
# By: Gianni Galbiati

# Standard Libraries

# Scientific Libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Custom Libraries
from .config import colors


def bootstrap_mean(x, num_resamples=1000):
    """Compute mean by bootstrap resampling"""
    means = []
    for i in range(1000):
        means.append(np.random.choice(x, size=num_resamples).mean())
    return np.stack(means).mean()


def jackknife_mean(x, num_subsamples=1000):
    """Compute mean by jackknife subsampling"""
    means = []
    for i in range(num_subsamples):
        means.append(np.random.choice(x, size=x.size // 2, replace=False))

    return np.stack(means).mean()


class PermutationTestBetween(object):
    """Constructs function for running between-groups permutation test"""
    def __init__(self, df, num_resamples=1000):
        self.df = df
        self.num_resamples = num_resamples
        self.diff_vals = {}
        self.samples = {}

    def _get_subject_conditions(self, subject_id):
        subject_filter = self.df['Subject ID'] == subject_id
        return self.df.loc[subject_filter, 'Condition'].values[0]

    def _pivot(self, target_var):
        pivot_table = self.df.pivot_table(
            index='Subject ID',
            columns='Is Real',
            values=target_var,
            aggfunc=np.mean
        )

        pivot_table['Condition'] = pivot_table.index.map(self._get_subject_conditions)

        return pivot_table

    def _get_diff(self, pivot_table, position_type):
        """Compute point difference for between groups"""

        x_trained = pivot_table.loc[self.condition_filter, position_type].mean()
        x_untrained = pivot_table.loc[~self.condition_filter, position_type].mean()

        return x_trained - x_untrained

    def _resample(self, pivot_table):
        return pivot_table['Condition'].sample(frac=1, replace=False).values

    def _hist(self, ax, diff_name):
        sns.distplot(
            self.samples[diff_name],
            bins=200, kde=False, norm_hist=False,
            color=colors[diff_name],
            ax=ax
        )

        ax.plot(
            [self.diff_vals[diff_name], ] * 2, [0, 10],
            color=colors[diff_name], label=diff_name.capitalize()
        )

        return None

    def _plot(self, fig, axes):
        self._hist(axes[0], 'real')
        self._hist(axes[1], 'fake')

        plt.figlegend(loc=0)
        plt.setp(
            axes,
            xlabel=r'$\Delta$ Mean',
            ylim=[0, 25]
        )

        fig.suptitle('Difference in mean between conditions')

        sns.despine()
        plt.tight_layout()

        return None

    def _collect_var(self, var_name):
        vals = np.stack(self.samples[var_name])
        index = (vals >= np.abs(self.diff_vals[var_name]))
        index |= (vals <= np.abs(self.diff_vals[var_name]))

        return vals, index

    def _report(self):
        real, real_index = self._collect_var('real')
        fake, fake_index = self._collect_var('fake')

        print(
            'Real positions p:', len(real[real_index]) / len(real),
            'val:', self.diff_vals['real']
        )

        print(
            'Fake positions p:', len(fake[fake_index]) / len(fake),
            'val:', self.diff_vals['fake']
        )

    def __call__(self, target_var):
        pivot_table = self._pivot(target_var)
        self.condition_filter = pivot_table['Condition'] == 'Trained'

        self.samples.update({'real': [], 'fake': []})

        self.diff_vals.update({
            'real': self._get_diff(pivot_table, True),
            'fake': self._get_diff(pivot_table, False)
        })

        for i in range(self.num_resamples):
            pivot_table['Condition'] = self._resample(pivot_table)
            self.condition_filter = pivot_table['Condition'] == 'Trained'

            self.samples['real'].append(self._get_diff(pivot_table, True))
            self.samples['fake'].append(self._get_diff(pivot_table, False))

        fig, axes = plt.subplots(1, 2, figsize=(8, 3), dpi=300)
        self._plot(fig, axes)
        self._report()

        return self.samples, axes


class PermutationTestWithin(PermutationTestBetween):
    """Constructs function for running within-groups permutation test"""

    def _pivot(self, target_var):
        """Pivots table to get per-suject means"""

        if 'Resampled Is Real' not in self.df.columns:
            self.df['Resampled Is Real'] = self.df['Is Real']

        pivot_table = self.df.pivot_table(
            index='Subject ID',
            columns='Resampled Is Real',
            values=target_var,
            aggfunc=np.mean
        )

        pivot_table['Condition'] = pivot_table.index.map(self._get_subject_conditions)

        self.condition_filter = pivot_table['Condition'] == 'Trained'

        return pivot_table

    def _resample_helper(self, df):
        return df[['Is Real', 'Subject ID']].sample(frac=1, replace=False)

    def _resample(self, grouped):
        applied = grouped.apply(self._resample_helper)
        applied = pd.DataFrame(applied)

        melted = pd.melt(
            applied,
            id_vars=['Subject ID'],
            value_vars=['Is Real'], value_name='Is Real'
        )

        return melted['Is Real'].values

    def _get_diff(self, pivot_table, condition):
        """Compute point difference for between groups"""
        f = self.condition_filter if condition == 'Trained' else ~self.condition_filter
        x_real = pivot_table.loc[f, True].mean()
        x_fake = pivot_table.loc[f, False].mean()

        return x_real - x_fake

    def _plot(self, fig, axes):
        self._hist(axes[0], 'trained')
        self._hist(axes[1], 'untrained')

        plt.figlegend(loc=0)

        plt.setp(
            axes,
            xlabel=r'$\Delta$ Mean',
            ylim=[0, 25]
        )

        fig.suptitle('Difference in mean between position types')
        sns.despine()
        plt.tight_layout()

        return None

    def _report(self):
        trained, trained_index = self._collect_var('trained')
        untrained, untrained_index = self._collect_var('untrained')

        print(
            'Trained subjects p:', len(trained[trained_index]) / len(trained),
            'val:', self.diff_vals['trained']
        )

        print(
            'Untrained subjects p:', len(untrained[untrained_index]) / len(untrained),
            'val:', self.diff_vals['untrained']
        )

    def __call__(self, target_var):
        pivot_table = self._pivot(target_var)
        grouped = self.df.groupby('Subject ID', sort=False)

        self.samples.update({'trained': [], 'untrained': []})

        self.diff_vals.update({
            'trained': self._get_diff(pivot_table, 'Trained'),
            'untrained': self._get_diff(pivot_table, 'Untrained')
        })

        for i in range(self.num_resamples):
            self.df['Resampled Is Real'] = self._resample(grouped)
            pivot_table = self._pivot(target_var)

            self.samples['trained'].append(self._get_diff(pivot_table, 'Trained'))
            self.samples['untrained'].append(self._get_diff(pivot_table, 'Untrained'))

        fig, axes = plt.subplots(1, 2, figsize=(8, 3), dpi=300)
        self._plot(fig, axes)
        self._report()

        return self.samples, axes
