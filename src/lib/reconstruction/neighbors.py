#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Vidrovr Inc.
# By: Gianni Galbiati

# Standard Libraries

import os
import multiprocessing as mp
import threading

# Scientific Libraries
import numpy as np
import pandas as pd
import scipy.signal as signal

# Custom Libraries
from ..util.datatype_utilities import position_string_to_array

count_filter = np.ones((3, 3), dtype=int)
count_filter[1, 1] = 0
dummy = signal.convolve2d(np.ones((4, 9)), count_filter, mode='same')


def count_neighbors(position_array):
    """

    Arguments
    ---------
    position_array : numpy.ndarray
        array representing a board position
        single channel; shape (4, 9)

    Returns
    -------
    numpy.ndarray
        the fraction of occupied neighbors at each board location
    """

    counts = signal.convolve2d(position_array, count_filter, mode='same')

    return counts / dummy


def get_adjacency(row,
                  bp_name='Black Position',
                  wp_name='White Position'):
    bp_string = row[bp_name]
    wp_string = row[wp_name]

    bp = position_string_to_array(bp_string)
    wp = position_string_to_array(wp_string)

    p = bp + wp

    # Average fraction of neighbors at all locations
    t = count_neighbors(p)

    # Average fraction of neighbors at occupied locations
    ct = count_neighbors(p) * p

    # Average fraction of same-color neighbors at occupied locations
    cs = count_neighbors(bp) * bp + count_neighbors(wp) * wp  # is this right?

    # Average fraction of opposite color neighbors at occupied locations
    co = count_neighbors(bp) * wp + count_neighbors(wp) * bp

    results = {
        'Total': t.mean(),
        'Cond. Total': ct[ct > 0].mean() if (ct > 0).any() else 0,
        'Same': cs[cs > 0].mean(),
        'Opposite': co[co > 0].mean()
    }

    return pd.Series(results)


def get_adjacency_per_location(row,
                               bp_name='Black Position',
                               wp_name='White Position'):
    bp_string = row[bp_name]
    wp_string = row[wp_name]

    bp = position_string_to_array(bp_string)
    wp = position_string_to_array(wp_string)

    # p = bp + wp
    black_neighbors = count_neighbors(bp)
    white_neighbors = count_neighbors(wp)
    all_neighbors = black_neighbors + white_neighbors

    same_neighbors = black_neighbors * bp + white_neighbors * wp
    opposite_neighbors = black_neighbors * wp + white_neighbors * bp


    return all_neighbors.reshape(36), same_neighbors.reshape(36), opposite_neighbors.reshape(36)


def main():
    pass


if __name__ == '__main__':
    main()