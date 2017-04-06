import os
import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal

data_dir = os.path.expanduser('~/Google Drive/Bas Zahy Gianni - Games/Data/2_eye/New')
game_dir = os.path.join(data_dir, 'game')
eyet_dir = os.path.join(data_dir, 'eyet')
mous_dir = os.path.join(data_dir, 'mous')
output_dir = os.path.expanduser('~/Google Drive/Bas Zahy Gianni - Games/Analysis/2_eye/histograms')
game_files = [os.path.join(game_dir, g) for g in os.listdir(game_dir) if g[-3:]=='csv']
eyet_files = [os.path.join(eyet_dir, e) for e in os.listdir(eyet_dir) if e[-3:]=='csv']
mous_files = [os.path.join(mous_dir, m) for m in os.listdir(mous_dir) if m[-3:]=='csv']

subject_initial_map = [g[-6:-4] for g in game_files]       # get alphabetical list of subject initials from filenames
subject_initial_map = dict(zip(subject_initial_map, np.arange(len(subject_initial_map))))
                                                           # map initials to ordinal indices
top = 192                                                  # board bounds and dimensions in pixels
bottom = 506
left = 177
right = 889
width = right - left
height = bottom - top

def mouse_x_to_tile(x):
    """Converts mouse x coordinates to board-space"""
    return 9*(x - left) / width

def mouse_y_to_tile(y):
    """Converts mouse y coordinates to board-space"""
    return 4*(y - top) / height

def expand_mouse_mt(row):
    """Appends start time, end time to mouse timestamp records for a single record"""

    endtime = int(row['ts'])                               # get turn end timestamp
    starttime = endtime - int(row['rt'])                   # get turn start from turn end and turn duration
    if type(row['mt'])==str:                               # check if valid data
        return str(starttime) + ',' + row['mt'] + ',' + str(endtime)
                                                           # add start, end times to respective ends of record
def expand_mouse_mx(row):
    """Appends start time location, end time location to mouse spatial coordinates for a single record"""

    endtime = int(row['ts'])                               # get turn end timestamp
    starttime = endtime - int(row['rt'])                   # get turn start from turn end and turn duration
    if type(row['mx'])==str:                               # check if valid data
        locs = row['mx'].split(';')                        # split record into (x, y) pair strings
        endloc = locs[-1]                                  # select first and last coordinate pairs
        startloc = locs[0]
        return startloc + ';' + row['mx'] + ';' + endloc    # add start, end coords to respective ends of record

def fix_game_boards(row):
    """Removes move from appropriate color board string representation for a single record"""

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

    gfnames = [
        'idx', 'id', 'color', 'gi', 'mi',
        'status', 'bp', 'wp', 'zet',
        'rt', 'ts', 'mt', 'mx'
    ]                                                      # names for columns

    D = pd.read_csv(gf, names=gfnames)                     # load csv into pandas dataframe
    readyfilter = D['status'] == 'ready'                   # filter on convenience records
    rtfilter = D['rt'] == 0                                # filter on AI moves
    aifilter = ~(readyfilter & rtfilter)                   # filter out convenience AI records
    D = D.loc[aifilter]                                    # apply filter
    D.loc[readyfilter, 'rt'] = 0                           # set human convenience records rt field to 0
    D['subject'] = gf[-6:-4]                               # set subject field to initials in game file name
    D['human'] = D['subject'].map(subject_initial_map)     # set human field to be subject index (alphabetical)
    D['move_start_ts'] = (D['ts'] - D['rt']).shift(-1)     # set move_start_timestamp field to turn beginning
    tsfilter = rtfilter | readyfilter                      # filter on ai OR convenience records
    D.loc[tsfilter, 'ts'] = D.loc[tsfilter, 'move_start_ts']
                                                           # replace invalid timestamps with inferred correct timestamp
    D['mx'] = D.apply(expand_mouse_mx, axis=1)             # append move start and end mouse spatial coords
    D['mt'] = D.apply(expand_mouse_mt, axis=1)             # append move start and end timestamps to mouse timestamps
    D['is human'] = 1                                      # initialize human player indicator variable
    playfilter = D['status'].isin(['playing', 'win', 'draw'])
                                                           # filter on non-convenience records
    D.loc[playfilter & rtfilter, 'is human'] = 0           # set human player indicator to 0 on AI records
    endfilter = D['status'].isin(['win', 'draw'])          # filter on game end records
    idx = D.loc[endfilter].index                                   # get indices for game end filter application
    if D.loc[idx[-1], 'rt'] != 0:                          # if human player ended last game
        D.loc[idx[-1], 'gi'] = D.loc[idx[-1], 'gi'] - 1    # subtract 1 from game index (why? probably a data error)
    bpfilter = D['color']==0                               # filter on player colors
    wpfilter = D['color']==1
    D.loc[bpfilter, 'bp'] = D.loc[bpfilter].apply(fix_game_boards, axis=1)
    D.loc[wpfilter, 'wp'] = D.loc[wpfilter].apply(fix_game_boards, axis=1)
                                                           # apply filters and remove last move from board
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
    M.index = M.index.astype(int)                          # cast timestamp index to integers
    return M

def load_eyetracker_file(ef):
    """Loads and preprocesses eyetracker data"""
    D = pd.read_csv(ef)                                    # load EL data into pandas dataframe
    D['subject'] = ef[-6:-4]                               # set subject field to initials
    D['human'] = D['subject'].map(subject_initial_map)     # set human field to subject ordinal index
    D[['start', 'end']] = (D[['start', 'end']]*1000).astype(int)
                                                           # set start and end fields to ms resolution, integers
    return D[['start', 'end', 'transx', 'transy', 'human']].set_index('start')
                                                           # lose the fluff; index by start timestamp
def make_tidy(e, m, g):
    """
    Produces a combined dataframe of mouse and eye coordinates, indexed by timestamp

    Should consider modifying to not use e_list, m_list, and g_list, but rather to take
    e, m, g as args
    """
                                                           # get appropriate dataframe for each subject
    start_time = int(e.index[0])                           # get the eyetracker start time
    end_time = int(e.loc[e.index.values[-1], 'end'])       # get the eyetracker end time
    mbounds = (m.index >= start_time) & (m.index <= end_time)
                                                           # filter on mouse records within EL record bounds
    m = m.loc[mbounds]                                     # apply filter
    idx = np.arange(start_time, end_time, 1)               # prepare index for new dataframe
    D = pd.DataFrame(index=idx)                            # new dataframe for tidy timeseries
    D.loc[e.index, 'eyex'] = e['transx'].astype(float)     # get valid eye coordinates in board space
    D.loc[e.index, 'eyey'] = e['transy'].astype(float)
    D.loc[e.index, 'eyeflag'] = 1                          # indicator for eye event

    D.loc[m.index, 'moux'] = m['x'].astype(float).map(mouse_x_to_tile)
                                                           # get valid mouse coords and map to board space
    D.loc[m.index, 'mouy'] = m['y'].astype(float).map(mouse_y_to_tile)
    D.loc[m.index, 'mouflag'] = 1                          # indicator for mouse event

    _sl = g.loc[g.index > start_time, :]                   # selector for valid game events

    D.loc[_sl.index, 'turn'] = 100*_sl['gi'] + _sl['mi']   # unique id for turns for valid game events
    D.loc[_sl.index, 'task'] = _sl['status']               # task indicator

    D = D.dropna(how='all')                                # shrink dataframe by pruning all event-less records

    fillcols = [
        'eyex', 'eyey',
        'moux', 'mouy',
        'turn', 'task'
    ]                                                      # fields to fill forward

    D[fillcols] = D[fillcols].fillna(method='ffill')       # fill forward

    D['ts'] = D.index                                      # convenience field of timestamps
    D.loc[D['eyeflag']==1, 'eyedur'] = D.loc[D['eyeflag']==1, 'ts'].diff(periods=1)
                                                           # set duration for each event of each type
    D.loc[D['mouflag']==1, 'moudur'] = D.loc[D['mouflag']==1, 'ts'].diff(periods=1)

    D['eyetile'] = D['eyex'].astype(int) + 9*D['eyey'].astype(int)                 # convert board coordinates to tile index
    mouvalid = ~pd.isnull(D['moux'])
    D.loc[mouvalid, 'moutile'] = D.loc[mouvalid, 'moux'].astype(int) + 9*D.loc[mouvalid, 'mouy'].astype(int)
    D.loc[D['eyeflag']==1, 'eyetile'] =  D.loc[D['eyeflag']==1, 'eyetile'].astype(int)
                                                           # cast valid tile vals to int (np.nan is float)
    D.loc[D['mouflag']==1, 'moutile'] =  D.loc[D['mouflag']==1, 'moutile'].astype(int)

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
    m['tile'] = (m['xtile'].astype(int) + 9*m['ytile'].astype(int))    # compute mouse tile

    humanfilter = m['is human'] == 1                       # filter on human moves (mouse df)
    mpvt = m.loc[humanfilter].pivot_table(index='turn', columns='tile', values='dur', aggfunc=np.sum)
                                                           # pivot human trials duration per tile idx
    mpvt['rt'] = mpvt.sum(axis=1)                          # recalculate rt for verification

    offboard = [
        i for i in mpvt.columns
        if (i not in list(range(36)) and type(i)==int)
    ]                                                      # column names for offboard locs

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


    g['turn'] = 100*g['gi'] + g['mi']
    turnfilter = g['status'].isin(['ready', 'playing', 'draw', 'win'])
    gp = g.loc[turnfilter]
    gp['turnstart'] = gp.index - gp['rt']
    gp['turnend'] = gp.index

    e['turnstart'] = np.nan
    e['turnend'] = np.nan
    e['ts'] = e.index
    e['tile'] = np.nan
    e['dur'] = np.nan
    e = e.drop_duplicates(subset='ts')
    e = e.append(gp[['turn', 'is human', 'turnstart', 'turnend']])
    e = e.sort_index()
    fillthese = ['turn', 'turnstart', 'turnend', 'is human']
    e[fillthese] = e[fillthese].fillna(method='bfill')

    evalid = ~pd.isnull(e['transx'])
    e.loc[evalid, 'tile'] = e.loc[evalid, 'transx'].astype(int) + 9*e.loc[evalid, 'transy'].astype(int)
    e['tile'] = e['tile'].fillna(method='ffill')
    tilefilter = pd.notnull(e['tile'])
    e.loc[tilefilter, 'tile'] = e.loc[tilefilter, 'tile'].astype(int)
    e['dur'] = e.index
    e['dur'] = e['dur'].diff(periods=1)

    eyebounds = (e.index >= e['turnstart']) & (e.index <= e['turnend'])
    e = e.loc[eyebounds]

    ehumanfilter = e['is human'] == 1
    epvt = e.loc[ehumanfilter].pivot_table(index='turn', columns='tile', values='dur', aggfunc=np.sum)
    epvt.columns = epvt.columns.astype(int)

    epvt['rt'] = epvt.sum(axis=1)
    offboard = [
        i for i in epvt.columns
        if (i not in list(range(36)) and type(i)==int)
    ]

    epvt[999] = epvt[offboard].sum(axis=1)
    turnfilter = g.status.isin(['playing', 'draw', 'win'])
    ghumanfilter = g['is human'] == 1
    gt = g.loc[turnfilter & ghumanfilter].set_index('turn')
    epvt.loc[gt.index, 'true rt'] = gt['rt']

    for c in ['bp', 'wp', 'zet']:
        epvt.loc[gt.index, c] = gt[c]

    epvt = epvt.loc[(epvt.index%100)!=0].fillna(value=0)

    for c in range(36):
        if c not in epvt.columns:
            epvt[c] = 0
#     print(np.abs(epvt['rt'] - epvt['true rt']).sum())

    return e, epvt

grid = np.dstack(np.mgrid[0:4, 0:9])                # grid for norm binning

def gausshist(row):
    p = multivariate_normal.pdf(grid, mean=row[:2])
    p *= row[2]
    return p.reshape(36)

def filtermove(df):
    df_ = df.loc[pd.notnull(df['end'])] # & (df['tile'] >= 0) & (df['tile'] < 36)]
    vals = df_.loc[:, ['transy', 'transx', 'dur']].values
    h = np.apply_along_axis(gausshist, axis=1, arr=vals)
    h = h.sum(axis=0)
    h = h / h.sum()
    return h

def main():
    e_list = [load_eyetracker_file(e) for e in eyet_files]     # lists of dataframes containing respective data
    m_list = [load_mouse_file(m) for m in mous_files]
    g_list = [load_game_file(g) for g in mous_files]

    t = [
        make_tidy(e_list[i], m_list[i], g_list[i])
        for i in range(len(e_list))
    ]                                                          # create tidy dfs per subject along timestamp index

    mpivs = []                                                 # holding lists for pivoted histograms
    epivs = []
    fepivs = []
    fmpivs = []

    for i in range(len(m_list)):                               # for each subject
        g = g_list[i]

        # MOUSE HISTOGRAMS
        m_list[i], mpvt = mouse_hist(m_list[i], g)
        mpivs.append(mpvt)

        # EYE HISTOGRAMS
        e_list[i], epvt = eye_hist(e_list[i], g)
        epivs.append(epvt)

        # FILTERED EYE HISTOGRAMS
        e = e_list[i]
        eclean = e.loc[pd.notnull(e['end'])]
        g = g.set_index('turn')
        filtered = eclean.groupby('turn').apply(filtermove)
        filtered = pd.DataFrame(index=filtered.index, data=np.stack(filtered.values))
        for c in ['bp', 'wp', 'zet']:
            filtered[c] = g.loc[filtered.index, c]
        fepivs.append(filtered)

    export_cols = list(range(36)) + [999, 'bp', 'wp', 'zet']
    filtered_cols = list(range(36)) + ['bp', 'wp', 'zet']

    # EXPORT FILES FOR EACH SUBJECT, HIST TYPE
    for i, mp in enumerate(mpivs):
        ep = epivs[i]
        fep = fepivs[i]

        mp[export_cols].to_csv(os.path.join(output_dir, 'mouse {}.csv'.format(i)))
        ep[export_cols].to_csv(os.path.join(output_dir, 'eye {}.csv'.format(i)))
        fep[filtered_cols].to_csv(os.path.join(output_dir, 'filtered eye {}.csv'.format(i)))

    return None

if __name__ == '__main__':
    main()
