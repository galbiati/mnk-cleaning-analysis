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

# Custom Libraries


def position_string_to_array(position_string):
    """Return the 4x9 single channel position representation from a string"""
    return np.stack(list(position_string)).astype(int).reshape([4, 9])

