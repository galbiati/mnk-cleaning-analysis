import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def bits2boards(num):
    s = '{0:b}'.format(num)
    return '0'*(36-len(s)) + s

# def show_position(ax, df, i):
#     """Reconstruct a game representation from string


#         To use, pass the mpl axis on which you wish to plot, the
#         dataframe containing appropriate data with conventional
#         column names, and the row index of the position you want to plot

        
#     """
#     bp = np.array(list(df.loc[i, 'bp'])).astype(int).reshape([4,9])
#     wp = np.array(list(df.loc[i, 'wp'])).astype(int).reshape([4,9])
#     ax.imshow(wp - bp, interpolation='nearest', cmap = 'gray')
#     ax.get_xaxis().set_ticks(np.arange(9)+.5)
#     ax.get_yaxis().set_ticks(np.arange(4)+.5)
#     ax.grid()
#     return ax
