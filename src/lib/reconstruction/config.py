#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Wei Ji Ma Lab
# By: Gianni Galbiati

# Standard Libraries

# Scientific Libraries
import numpy as np
import pandas as pd
import seaborn as sns

# Custom Libraries


# Get colors for between/within groups comparisons
colors = sns.cubehelix_palette(
    n_colors=2, start=0.5, hue=1, rot=.1, light=.65
)

colors += sns.cubehelix_palette(
    n_colors=2, start=2.5, hue=1, rot=.1, light=.65
)

colors = dict(zip(['real', 'fake', 'trained', 'untrained'], colors))


def set_library_options():
    # Plotting config
    sns.set_style('white')
    sns.set_context('talk')

    # DataFrame display config
    pd.set_option('display.max_columns', 40)

    return None
