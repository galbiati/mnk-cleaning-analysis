import pandas as pd
import numpy as np
import scipy.signal as sig

"""
This file contains functions for counting various kinds of errors and other
statistics/properties in the reconstruction experiment where human participants
were asked to recreate a game position from memory.

Most of these are meant to be used with the pandas.DataFrame.apply() method,
where the dataframe contains the columns ['bp true', 'wp true', 'bp', 'wp'] for
the original and recreated positions for each color channel.
"""

def string_to_array(board_string):
    """
    Convert a string representation of a board channel to a numpy
    integer array
    """
    return np.array(list(board_string)).astype(int)

def expand_row(row):
    """
    Utility function for extracting positions in pandas dataframe
    """
    bpt, wpt, bp, wp = row[['bp true', 'wp true', 'bp', 'wp']].map(string_to_array)
    return bpt, wpt, bp, wp

def score(row):
    """
    Returns the score (number INCORRECT) for each item
    """
    bpt, wpt, bp, wp = expand_row(row)
    bperror = (bpt != bp).astype(int).sum()
    wperror = (wpt != wp).astype(int).sum()
    doubleerror = ((bpt != bp) & (wpt != wp)).astype(int).sum()
    return bperror + wperror - doubleerror

def extra_pieces(row):
    """
    Counts the number of additional pieces in the reconstruction
    """
    bpt, wpt, bp, wp = expand_row(row)
    pt = bpt + wpt
    p = bp + wp

    return (pt - p < 0).sum()

def missing_pieces(row):
    """
    Counts the number of missing pieces in the reconstruction
    """
    bpt, wpt, bp, wp = expand_row(row)
    pt = bpt + wpt
    p = bp + wp

    return (pt - p > 0).sum()

def wrong_color(row):
    """
    Counts the number of pieces with the wrong color in the reconstruction
    """
    bpt, wpt, bp, wp = expand_row(row)
    b2w = ((bpt == 1) & (wp == 1)).sum()
    w2b = ((wpt == 1) & (bp == 1)).sum()

    return b2w + w2b

def n_pieces(row):
    """
    Counts the number of pieces in the original position
    """
    bpt, wpt = row[['bp true', 'wp true']]
    n_bpieces = string_to_array(bpt).sum()
    n_wpieces = string_to_array(wpt).sum()
    return n_bpieces + n_wpieces

def n_neighbors(x, f):
    """
    Operates with np.apply_along_axis
    Counts number of neighbors by convolving with appropriate filter
    """
    xin = x.reshape([4, 9])
    c = sig.convolve(xin, f, mode='same')
    return c.reshape(36)

def h_neighbors(x):
    """
    Count horizontal neighbors
    """
    f = np.array([[1, 0, 1]])
    return n_neighbors(x, f)

def v_neighbors(x):
    """
    Count vertical neighbors
    """
    f = np.array([[1, 0, 1]]).T
    return n_neighbors(x, f)

def d_neighbors(x):
    """
    Count diagonal neighbors
    """
    f = np.diag(np.array([1, 0, 1]))
    f = f + f[:, ::-1]
    return n_neighbors(x, f)
