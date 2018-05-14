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
from scipy.stats import multivariate_normal

# Internal Python Libraries (alphabetical order)


# Set up directory and filepath references
data_dir = os.path.expanduser('~/Google Drive/Bas Zahy Gianni - Games/Data/2_eye/New')
game_dir = os.path.join(data_dir, 'game')
eyet_dir = os.path.join(data_dir, 'eyet')
mous_dir = os.path.join(data_dir, 'mous')
output_dir = os.path.expanduser('~/Google Drive/Bas Zahy Gianni - Games/Analysis/2_eye/histograms/temp')
os.makedirs(output_dir, exist_ok=True)

game_files = [os.path.join(game_dir, g) for g in os.listdir(game_dir) if g[-3:] == 'csv']
eyet_files = [os.path.join(eyet_dir, e) for e in os.listdir(eyet_dir) if e[-3:]=='csv']
mous_files = [os.path.join(mous_dir, m) for m in os.listdir(mous_dir) if m[-3:]=='csv']

# Get subject identifiers
subject_initial_map = [g[-6:-4] for g in game_files]       # get alphabetical list of subject initials from filenames
subject_initial_map = dict(zip(subject_initial_map, np.arange(len(subject_initial_map))))

# Dimensions of board display in pixels
top = 192
bottom = 506
left = 177
right = 889
width = right - left
height = bottom - top


def mouse_x_to_tile(x):
    """Converts mouse x coordinates to board-space"""
    return 9 * (x - left) / width


def mouse_y_to_tile(y):
    """Converts mouse y coordinates to board-space"""
    return 4 * (y - top) / height


def expand_mouse_mt(row):
    """
    Appends start time, end time to mouse timestamp records for a single record

    For use with pd.DataFrame.apply()
    """

    endtime = int(row['ts'])                               # get turn end timestamp
    starttime = endtime - int(row['rt'])                   # get turn start from turn end and turn duration

    # add start, end times to respective ends of record
    if type(row['mt']) == str:                             # check if valid data
        return str(starttime) + ',' + row['mt'] + ',' + str(endtime)


def expand_mouse_mx(row):
    """
    Appends start time location, end time location to mouse spatial coordinates for a single record

    For use with pd.DataFrame.apply()
    """

    endtime = int(row['ts'])                               # get turn end timestamp
    starttime = endtime - int(row['rt'])                   # get turn start from turn end and turn duration

    if type(row['mx']) == str:                             # check if valid data
        locs = row['mx'].split(';')                        # split record into (x, y) pair strings
        endloc = locs[-1]                                  # select first and last coordinate pairs
        startloc = locs[0]
        return startloc + ';' + row['mx'] + ';' + endloc   # add start, end coords to respective ends of record


def fix_game_boards(row):
    """
    Removes move from appropriate color board string representation for a single record

    For use with pd.DataFrame.apply()
    """

    bp, wp = row[['bp', 'wp']]                              # select board string reps

    if row['color']==0:                                     # if Black is player
        p = list(bp)                                        # convert board string to list (mutability)
        p[int(row['zet'])] = '0'                            # set list at zet loc to be '0'
    else:                                                   # if White is player, do the same thing for White's board
        p = list(wp)
        p[int(row['zet'])] = '0'

    return ''.join(p)                                       # rejoin into new string

def load_game_file(gf):
    """Loads and preprocesses data from game observations"""

    # Labels for fields
    gfnames = [
        'idx', 'id', 'color', 'gi', 'mi', 'status',
        'bp', 'wp', 'zet', 'rt',
        'ts', 'mt', 'mx'
    ]

    # Load csv into a dataframe
    D = pd.read_csv(gf, names=gfnames)

    # Remove convenience records by AI
    readyfilter = D['status'] == 'ready'                   # filter on convenience records
    rtfilter = D['rt'] == 0                                # filter on AI moves
    aifilter = ~(readyfilter & rtfilter)                   # filter out convenience AI records
    D = D.loc[aifilter]                                    # apply filter

    # Make necessary data corrections
    D.loc[readyfilter, 'rt'] = 0                           # set human convenience records rt field to 0
    D['subject'] = gf[-6:-4]                               # set subject field to initials in game file name
    D['human'] = D['subject'].map(subject_initial_map)     # set human field to be subject index (alphabetical)
    D['move_start_ts'] = (D['ts'] - D['rt']).shift(-1)     # set move_start_timestamp field to turn beginning
    tsfilter = rtfilter | readyfilter                      # filter on ai OR convenience records
    D.loc[tsfilter, 'ts'] = D.loc[tsfilter, 'move_start_ts']    # replace invalid timestamps with inferred move start
    D['mx'] = D.apply(expand_mouse_mx, axis=1)             # append move start and end mouse spatial coords
    D['mt'] = D.apply(expand_mouse_mt, axis=1)             # append move start and end timestamps to mouse timestamps
    D['is human'] = 1                                      # initialize human player indicator variable
    playfilter = D['status'].isin(['playing', 'win', 'draw'])   # filter on non-convenience records
    D.loc[playfilter & rtfilter, 'is human'] = 0           # set human player indicator to 0 on AI records
    endfilter = D['status'].isin(['win', 'draw'])          # filter on game end records
    idx = D.loc[endfilter].index                           # get indices for game end filter application

    if D.loc[idx[-1], 'rt'] != 0:                          # if human player ended last game
        D.loc[idx[-1], 'gi'] = D.loc[idx[-1], 'gi'] - 1    # subtract 1 from game index (why? probably a data error)

    bpfilter = D['color'] == 0                             # filter on player colors
    wpfilter = D['color'] == 1

    # Apply filters and remove last move from board
    D.loc[bpfilter, 'bp'] = D.loc[bpfilter].apply(fix_game_boards, axis=1)
    D.loc[wpfilter, 'wp'] = D.loc[wpfilter].apply(fix_game_boards, axis=1)

    return D.set_index('ts')                               # set index to timestamps


def load_mouse_file(mf):
    """Loads and preprocesses mouse tracking data"""

    mfnames = [
        'idx', 'id', 'color', 'gi', 'mi',
        'status', 'bp', 'wp', 'zet',
        'rt', 'ts', 'mt', 'mx'
    ]                                                      # names for columns

    D = pd.read_csv(mf, names=mfnames)                     # load csv into pandas dataframe
    D['mx'] = D.apply(expand_mouse_mx, axis=1)             # append start and end mouse spatial coords
    D['mt'] = D.apply(expand_mouse_mt, axis=1)             # append start and end mouse timestamps
    D = D[['mt', 'mx']]                                    # lose the fluff

    valid = pd.notnull(D['mt'])                            # select records with valid mouse time series
    m = (D.loc[valid, 'mt'] + ',').sum().split(',')[:-1]   # combine all mouse timestamp records
    x = [tuple(xy.split(',')) for xy in (D.loc[valid, 'mx'] + ';').sum().split(';')][:-1]
                                                           # combine all mouse coordinate records
    M = pd.DataFrame(index=m, data=x, columns=['x', 'y'])  # new dataframe with timestamp index and coordinates
    M['subject'] = mf[-6:-4]                               # set subject field to initials
    M['human'] = M['subject'].map(subject_initial_map)     # set human field to subject ordinal index
    M.index = M.index.astype(np.int64)                          # cast timestamp index to integers

    return M


def load_eyetracker_file(ef):
    """Loads and preprocesses eyetracker data"""

    D = pd.read_csv(ef)                                    # load EL data into pandas dataframe
    D['subject'] = ef[-6:-4]                               # set subject field to initials
    D['human'] = D['subject'].map(subject_initial_map)     # set human field to subject ordinal index

    # Set start and end fields to ms resolution, integers
    D[['start', 'end']] = (D[['start', 'end']]*1000).astype(np.int64)

    # Lose the fluff; index by start timestamp
    return D[['start', 'end', 'transx', 'transy', 'human']].set_index('start')


def make_tidy(e, m, g):
    """Produces a combined dataframe of mouse and eye coordinates, indexed by timestamp"""

    start_time = int(e.index[0])                           # get the eyetracker start time
    end_time = int(e.loc[e.index.values[-1], 'end'])       # get the eyetracker end time
    mbounds = (m.index >= start_time) & (m.index <= end_time) # filter on mouse records within EL record bounds
    m = m.loc[mbounds]                                     # apply filter
    idx = np.arange(start_time, end_time, 1)               # prepare index for new dataframe
    D = pd.DataFrame(index=idx)                            # new dataframe for tidy timeseries
    D.loc[e.index, 'eyex'] = e['transx'].astype(float)     # get valid eye coordinates in board space
    D.loc[e.index, 'eyey'] = e['transy'].astype(float)
    D.loc[e.index, 'eyeflag'] = 1                          # indicator for eye event
    D.loc[m.index, 'moux'] = m['x'].astype(float).map(mouse_x_to_tile) # get valid mouse coords and map to board space
    D.loc[m.index, 'mouy'] = m['y'].astype(float).map(mouse_y_to_tile)
    D.loc[m.index, 'mouflag'] = 1                          # indicator for mouse event

    _sl = g.loc[g.index > start_time, :]                   # selector for valid game events
    D.loc[_sl.index, 'turn'] = 100*_sl['gi'] + _sl['mi']   # unique id for turns for valid game events
    D.loc[_sl.index, 'task'] = _sl['status']               # task indicator
    D = D.dropna(how='all')                                # shrink dataframe by pruning all event-less records

    # Fill fields forward
    fillcols = ['eyex', 'eyey', 'moux', 'mouy', 'turn', 'task']
    D[fillcols] = D[fillcols].fillna(method='ffill')

    D['ts'] = D.index                                      # convenience field of timestamps

    # Set duration for each event of each type
    D.loc[D['eyeflag'] == 1, 'eyedur'] = D.loc[D['eyeflag'] ==1, 'ts'].diff(periods=1)
    D.loc[D['mouflag'] == 1, 'moudur'] = D.loc[D['mouflag'] == 1, 'ts'].diff(periods=1)

    # Convert board coordinates to tile index
    D['eyetile'] = D['eyex'].astype(np.int64) + 9*D['eyey'].astype(np.int64)
    mouvalid = ~pd.isnull(D['moux'])
    D.loc[mouvalid, 'moutile'] = D.loc[mouvalid, 'moux'].astype(np.int64) + 9*D.loc[mouvalid, 'mouy'].astype(np.int64)

    # Cast valid tile vals to int (np.nan is float)
    D.loc[D['eyeflag']==1, 'eyetile'] =  D.loc[D['eyeflag']==1, 'eyetile'].astype(np.int64)
    D.loc[D['mouflag']==1, 'moutile'] =  D.loc[D['mouflag']==1, 'moutile'].astype(np.int64)

    return D


def mouse_hist(m, g):
    """Modifies mousetracking data to produce histograms over tile indices"""

    g['turn'] = 100*g['gi'] + g['mi']                      # add unique turn ids
    turnfilter = g['status'].isin(['playing', 'draw', 'win'])
                                                           # filter on non-convenience records
    gp = g.loc[turnfilter]                                 # apply filter
    m['turn'] = np.nan                                     # initialize helper fields
    m['turnstart'] = np.nan
    m['turnend'] = np.nan
    m['ts'] = m.index
    m['xtile'] = np.nan
    m['ytile'] = np.nan
    m['tile'] = np.nan
    m['dur'] = np.nan
    m['is human'] = np.nan
    m = m.drop_duplicates(subset='ts')                     # get rid of duplicate timestamps
    m.loc[gp.index, 'turn'] = gp['turn']                   # add helper data to mouse df
    m.loc[gp.index, 'turnstart'] = gp.index - gp['rt']
    m.loc[gp.index, 'turnend'] = gp.index
    m.loc[gp.index, 'is human'] = gp['is human']

    m = m.sort_index()                                     # sort mouse data by timestamp
    fillthese = ['turn', 'turnstart', 'turnend', 'is human']
                                                           # helper columns to fill
    m[fillthese] = m[fillthese].fillna(method='bfill')     # backfill missing data

    m['dur'] = m.index
    m['dur'] = m['dur'].diff(periods=1)                    # compute duration of each event
    eventbounds = (m.index > m['turnstart']) & (m.index <= m['turnend'])
                                                           # filter on mouse data within player turn
    m = m.loc[eventbounds]                                 # apply filter

    m['xtile'] = m['x'].astype(float).map(mouse_x_to_tile) # map mouse coords to board coords
    m['ytile'] = m['y'].astype(float).map(mouse_y_to_tile)
    m['tile'] = (m['xtile'].astype(np.int64) + 9*m['ytile'].astype(np.int64))    # compute mouse tile

    humanfilter = m['is human'] == 1                       # filter on human moves (mouse df)
    mpvt = m.loc[humanfilter].pivot_table(index='turn', columns='tile', values='dur', aggfunc=np.sum)
                                                           # pivot human trials duration per tile idx
    mpvt['rt'] = mpvt.sum(axis=1)                          # recalculate rt for verification

    # Get column names for locations off the board
    offboard = [i for i in mpvt.columns if (i not in list(range(36)) and type(i)==int)]

    mpvt[999] = mpvt[offboard].sum(axis=1)                 # combine all offboard durations
    humanfilter = g['is human'] == 1                       # filter on human moves (game df)
    gt = g.loc[turnfilter & humanfilter].set_index('turn') # get non-convenience human records
    mpvt.loc[gt.index, 'true rt'] = gt['rt']               # set 'true rt' for verification
    mpvt = mpvt.fillna(value=0)                            # nan values mean 0 duration
#     print('Mouse dif from true rt:', np.abs(mpvt['rt'] - mpvt['true rt']).sum())

    for c in ['bp', 'wp', 'zet']:
        mpvt.loc[gt.index, c] = gt[c]                      # set other game info fields on hist records

    for c in range(36):
        if c not in mpvt.columns:                          # set all nonvisited trials to 0 dur
            mpvt[c] = 0

    return m, mpvt


def eye_hist(e, g):
    """
    Modifies eyetracking data to produce histograms per trial

    note: eye hist requires including ready markers to distinguish correctly, due to latency etc
    (mousetracking does not record until after ready stops)
    """

    print('epiv fns')
    # Get identifier for each turn
    g['turn'] = 100*g['gi'] + g['mi']

    # Filter for valid game status records
    turn_filter = g['status'].isin(['ready', 'playing', 'draw', 'win'])
    gp = g.loc[turn_filter]

    # Set turn start and turn end timestamps
    gp['turnstart'] = gp.index - gp['rt']
    gp['turnend'] = gp.index

    # Initialize fields in eyetracking dataframe
    e['turnstart'] = np.nan
    e['turnend'] = np.nan
    e['ts'] = e.index
    e['tile'] = np.nan
    e['dur'] = np.nan

    # Drop duplicate timestamps from e
    e = e.drop_duplicates(subset='ts')

    # Insert rows from game data at timestamps and sort by time
    e = e.append(gp[['turn', 'is human', 'turnstart', 'turnend']])
    e = e.sort_index()

    # Fill appropriate records backwards in time (game records submitted at END of turn)
    fillthese = ['turn', 'turnstart', 'turnend', 'is human']
    e[fillthese] = e[fillthese].fillna(method='bfill')

    # Convert translated coordinates to tile indices
    evalid = ~pd.isnull(e['transx'])
    e.loc[evalid, 'tile'] = e.loc[evalid, 'transx'].astype(np.int64) + 9*e.loc[evalid, 'transy'].astype(np.int64)
    e['tile'] = e['tile'].fillna(method='ffill')
    tilefilter = pd.notnull(e['tile'])
    e.loc[tilefilter, 'tile'] = e.loc[tilefilter, 'tile'].astype(np.int64)

    # Calculate observation durations
    e['dur'] = e.index
    e['dur'] = e['dur'].diff(periods=1)

    # Filter out observations that don't happen during turn
    eyebounds = (e.index >= e['turnstart']) & (e.index <= e['turnend'])
    e = e.loc[eyebounds]

    # Get total duration per turn for human trials
    ehumanfilter = e['is human'] == 1

    eendfilter = pd.notnull(e['end'])
    good_turns = e.loc[eendfilter, 'turn']

    epvt = e.loc[ehumanfilter & eendfilter].pivot_table(
        index='turn', columns='tile', values='dur', aggfunc=np.sum
    )

    epvt.columns = epvt.columns.astype(np.int64)

    # Calculate response time
    epvt['rt'] = epvt.sum(axis=1)

    # Combine off-board locations into single location
    offboard = [
        i for i in epvt.columns
        if (i not in list(range(36)) and type(i) == int)
    ]

    epvt[999] = epvt[offboard].sum(axis=1)

    # Drop convenience records
    turn_filter = g.status.isin(['playing', 'draw', 'win'])
    g_human_filter = g['is human'] == 1
    g_good_turn_filter = g['turn'].isin(good_turns)

    # Get a view of game data indexed by turn (same as epvt)
    gt = g.loc[turn_filter & g_human_filter & g_good_turn_filter].set_index('turn')
    epvt.loc[gt.index, 'true rt'] = gt['rt']

    # Set game data values on epvt
    for c in ['bp', 'wp', 'zet']:
        epvt.loc[gt.index, c] = gt[c]

    # Get rid of entries where gi == mi == 0 and fill in zeros at all missing values
    #   ? - don't remember why I did this, but probably was due to the way timestamps recorded when AI moved first?
    epvt = epvt.loc[(epvt.index % 100) != 0].fillna(value=0)

    # Make sure all columns have values
    for c in range(36):
        if c not in epvt.columns:
            epvt[c] = 0

#     print(np.abs(epvt['rt'] - epvt['true rt']).sum())

    return e, epvt


# Get a grid for norm binning
grid = np.dstack(np.mgrid[0:4, 0:9])


def gausshist(row, cov=1):
    """
    Compute a multivariate normal distribution and filter location

    For use with np.apply_along_axis()
    """

    p = multivariate_normal.pdf(grid, mean=row[:2], cov=cov)
    p *= row[2]
    return p.reshape(36)


def filtermove(df, cov=1):
    """Apply Gaussian filter to all moves"""

    df_ = df.loc[pd.notnull(df['end'])]  # & (df['tile'] >= 0) & (df['tile'] < 36)]
    vals = df_.loc[:, ['transy', 'transx', 'dur']].values

    gh = lambda x: gausshist(x, cov=cov)
    h = np.apply_along_axis(gh, axis=1, arr=vals)
    h = h.sum(axis=0)
    h = h / h.sum()
    return h


def filterhalf(row, which='first'):
    """
    Retrieves only half of an observation

    For use with pd.DataFrame.apply()
    """

    halfway = row['turnstart'] + (row['turnend'] - row['turnstart']) / 2
    if which == 'first':
        return row.name <= halfway
    else:
        return row.name > halfway


def make_filtered_hist(groupeddf, g, filterfunc=filtermove):
    """
    Filter an entire histogram

    For use with pd.DataFrame.groupby()
    """

    filtered = groupeddf.apply(filterfunc)
    filtered = pd.DataFrame(index=filtered.index, data=np.stack(filtered.values))

    for c in ['bp', 'wp', 'zet']:
        filtered[c] = g.loc[filtered.index, c]

    return filtered


def main():

    # Get a list of dataframes for each kind of data
    e_list = [load_eyetracker_file(e) for e in eyet_files]
    m_list = [load_mouse_file(m) for m in mous_files]
    g_list = [load_game_file(g) for g in mous_files]

    # create tidy dfs per subject along timestamp index
    # t = [make_tidy(e_list[i], m_list[i], g_list[i]) for i in range(len(e_list))]

    # Create holding lists for histograms
    mpivs = []
    epivs = []
    fepivs = []
    fepivs_wide = []
    fepivs_narrow = []
    fepivs_half0 = []
    fepivs_half1 = []

    # For each subject, generate histogrmams
    for i in range(len(m_list)):
        g = g_list[i]

        # MOUSE HISTOGRAMS
        m_list[i], mpvt = mouse_hist(m_list[i], g)
        mpivs.append(mpvt)

        # EYE HISTOGRAMS
        e_list[i], epvt = eye_hist(e_list[i], g)
        epivs.append(epvt)
        print("epiv len", len(epvt))

        # FILTERED EYE HISTOGRAMS
        e = e_list[i]
        eendfilter = pd.notnull(e['end'])
        ehumanfilter = e['is human'] == 1
        eclean = e.loc[eendfilter & ehumanfilter]
        half0 = eclean.apply(filterhalf, axis=1)
        eclean_half0 = eclean.loc[half0]
        eclean_half1 = eclean.loc[~half0]

        # good_turns = e.loc[eendfilter, 'turn']
        # turn_filter = g.status.isin(['playing', 'draw', 'win'])
        # g_human_filter = g['is human'] == 1
        # g_good_turn_filter = g['turn'].isin(good_turns)
        # gt = g.loc[turn_filter & g_human_filter & g_good_turn_filter].set_index('turn')

        grouped = eclean.groupby('turn')
        widefunc = lambda x: filtermove(x, cov=[[1.5, 0], [0, 1.5]])
        narrowfunc = lambda x: filtermove(x, cov=[[.66, 0], [0, .66]])

        filtered = make_filtered_hist(grouped, g)
        filtered_wide = make_filtered_hist(grouped, g, filterfunc=widefunc)
        filtered_narrow = make_filtered_hist(grouped, g, filterfunc=narrowfunc)
        filtered_half0 = make_filtered_hist(eclean_half0.groupby('turn'), g)
        filtered_half1 = make_filtered_hist(eclean_half1.groupby('turn'), g)

        print('filtered len', len(filtered))
        fepivs.append(filtered)
        fepivs_wide.append(filtered_wide)
        fepivs_narrow.append(filtered_narrow)
        fepivs_half0.append(filtered_half0)
        fepivs_half1.append(filtered_half1)

    export_cols = list(range(36)) + [999, 'bp', 'wp', 'zet']
    fil_cols = list(range(36)) + ['bp', 'wp', 'zet']

    # EXPORT FILES FOR EACH SUBJECT, HIST TYPE
    for i, mp in enumerate(mpivs):
        ep = epivs[i]
        fep = fepivs[i]
        fepw = fepivs_wide[i]
        fepn = fepivs_narrow[i]
        feph0 = fepivs_half0[i]
        feph1 = fepivs_half1[i]

        mp[export_cols].to_csv(os.path.join(output_dir, 'mouse {}.csv'.format(i)))
        ep[export_cols].to_csv(os.path.join(output_dir, 'eye {}.csv'.format(i)))
        fep[fil_cols].to_csv(os.path.join(output_dir, 'filtered eye {}.csv'.format(i)))
        fepw[fil_cols].to_csv(os.path.join(output_dir, 'filtered eye wide {}.csv'.format(i)))
        fepn[fil_cols].to_csv(os.path.join(output_dir, 'filtered eye narrow {}.csv'.format(i)))
        feph0[fil_cols].to_csv(os.path.join(output_dir, 'filtered eye half0 {}.csv').format(i))
        feph1[fil_cols].to_csv(os.path.join(output_dir, 'filtered eye half1 {}.csv').format(i))

    return None


if __name__ == '__main__':
    main()
