#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Wei Ji Ma Lab, New York University Center for Neural Science
# By: Gianni Galbiati

# Standard Python Libraries (alphabetical order)
import os

# Scientific Python Libraries (alphabetical order)
import numpy as np
import pandas as pd
import scipy.stats as sts
import seaborn as sns

# Internal Python Libraries (alphabetical order)

import lib.util.eyelink as el


# Configure plotting style
sns.set(palette='muted')
sns.set_style('white')
B, G, R, P = sns.color_palette("muted", 4)
sns.set_context('poster')

# Set directory references
top_dir = os.path.expanduser('~/Google Drive/Bas Zahy Gianni - Games')

data_dir = os.path.join(top_dir, 'Data/2_eye')
data_source_dir = os.path.join(data_dir, 'Raw')
output_dir = os.path.join(data_dir, 'Clean/temp')
os.makedirs(output_dir, exist_ok=True)

analysis_dir = os.path.join(top_dir, 'Analysis/2_eye')
statistics_dir = os.path.join(analysis_dir, 'Statistics/temp')
os.makedirs(statistics_dir, exist_ok=True)

# Initials for subjects
subject_initials = ["RG", "JL", "JA", "PO", "BA", "AN", "IA", "DL", "JP", "VL"]


def event_data(si):
    s = el.Subject(
        os.path.join(data_source_dir, 'Game', si + '.csv'),
        os.path.join(data_source_dir, 'Eye', si + '.asc'),
        si
    )

    _s = s.events
    _s = _s.fillna(method='backfill')
    s.events = _s

    return s

# def histogram(e, convolve=False):
#     s = e.subject
#     e = e.events
#     e_filter = e['status'].isin(['playing', 'win', 'draw'])
#     e_cols = [
#         'dur', 'etype', 'start', 'end', 'res', 'gi',
#         'mi', 'status', 'exp_time', 'exp_move_start',
#         'bp', 'wp', 'transx', 'transy', 'subject'
#     ]
#
#     e = e.loc[e_filter, e_cols]
#
#     cols = list(range(36)) + ['bp', 'wp', 'res', 'rt', 'gi', 'mi', 'subject']
#     hist = pd.DataFrame(index=[0], columns=cols)
#     trace_list = ['transx', 'transy', 'dur', 'bp', 'wp', 'res', 'subject']
#
#     for g in e.gi.unique():
#         moves = e.loc[e.gi==g, 'mi'].unique()
#         for m in moves:
#             trace = e.loc[(e.gi == g) & (e.mi == m), trace_list]
#             if (len(trace) > 0):
#                 if convolve:
#                     h = np.zeros([4, 9])
#                     xdim = list(range(9))
#                     ydim = list(range(4))
#                     for ti in trace.index.values:
#                         x = trace.loc[ti, 'transx']
#                         y = trace.loc[ti, 'transy']
#                         d = trace.loc[ti, 'dur']
#                         xrv = sts.norm(x, .707)
#                         yrv = sts.norm(y, .707)
#                         for xd in xdim:
#                             for yd in ydim:
#                                 dx = xrv.cdf(xd + 1) - xrv.cdf(xd)
#                                 dy = yrv.cdf(yd + 1) - yrv.cdf(yd)
#                                 h[3 - yd, xd] += dx * dy * d
#                 else:
#                     trace.loc[:, 'x'] = np.floor(trace.transx)
#                     trace.loc[:, 'y'] = np.floor(trace.transy)
#                     h = np.zeros([4, 9])
#                     for i in range(36):
#                         x = i % 9
#                         y = i // 9
#                         h[3-y, x] = trace.loc[(trace.x == x) & (trace.y == y), 'dur'].sum()
#                 hi = hist.index.values[-1] + 1
#                 hist.loc[hi, list(range(36))] = (h / h.sum()).reshape(36)
#                 hist.loc[hi, 'bp'] = trace.bp.values[-1]
#                 hist.loc[hi, 'wp'] = trace.wp.values[-1]
#                 hist.loc[hi, 'rt'] = trace.dur.sum()
#                 hist.loc[hi, 'res'] = trace.res.values[-1]
#                 hist.loc[hi, 'gi'] = g
#                 hist.loc[hi, 'mi'] = m
#                 hist.loc[hi, 'subject'] = s
#
#     return hist


def main():
    for si in subject_initials:
        print('Starting ' + si + '...')
        s = event_data(si) #event_data(si, subjects)

        s.events.to_csv(
            os.path.join(output_dir, 'new_' + si + "_events.csv"),
            index=False
        )

        s.experiment.to_csv(
            os.path.join(output_dir, 'new_' + si + '_exp.csv'),
            index=False
        )

        print(si + ' events done')
        # h = histogram(s)
        #
        # h.to_csv(statistics_dir + 'histograms/' + si + '.csv', index=False)
        #
        # print(si + ' hists done')
        # hc = histogram(s, convolve=True)
        # hc.to_csv(statistics_dir + 'histograms/' + si + '_conv.csv', index=False)
        # print(si + ' convolved hists done')

    return None

if __name__ == '__main__':
    main()