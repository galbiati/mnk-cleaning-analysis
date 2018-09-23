#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Wei Ji Ma Lab
# By: Gianni Galbiati

# Standard Libraries

# Scientific Libraries
import numpy as np

# Custom Libraries


def position_string_to_array(position_string):
    """Return the 4x9 single channel position representation from a string"""
    return np.stack(list(position_string)).astype(int).reshape([4, 9])

