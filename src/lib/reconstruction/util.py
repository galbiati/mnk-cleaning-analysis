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


def clean_condition_name(condition):
    return 'Untrained' if condition == 'Naive' else condition


def load_dataset():
    # Load tidy records from file
    tidy = pd.read_csv('./tidy_data.csv', index_col=0)
    tidy['Condition'] = tidy['Condition'].map(clean_condition_name)
    tidy['Position ID'] = tidy['Position ID'].map(int)

    board_set_vals = [
        'Black Position', 'White Position',
        'Is Real', 'Num Pieces'
    ]

    # Load board set (unique stimuli)
    board_set = tidy.pivot_table(
        index='Position ID', values=board_set_vals,
        aggfunc=lambda x: x.unique()[0]
    )

    return tidy, board_set[board_set_vals]
