import os
import pandas as pd

# Load data
data_dir = os.path.expanduser('~/Google Drive/Bas Zahy Gianni - Games/Data/1_gen/Clean/_summaries/')
data = pd.read_csv(os.path.join(data_dir, 'all_games_all_fields.csv'))

# Select game end records
endlocs = data['status'].isin(['win', 'draw'])
ends = data.loc[endlocs, :]

# New DF with wins
wins = pd.DataFrame(index=ends.index, columns=['human', 'computer', 'black_player', 'outcome'])
wins['human'] = ends['human']
wins['computer'] = ends['computer']

def who_is_black(row):
    # 1 if human is black, 0 if comp
    if row['subject'] == row['human']:
        if row['color'] == 0:
            return 1
        else:
            return 0
    else:
        if row['color']== 0:
            return 0
        else:
            return 1

def who_won(row):
    # 1 if black wins, 0 if draw, -1 if white wins
    if row['status'] == 'draw':
        return 0
    elif row['color'] == 0:
        return 1
    else:
        return -1

def main():
    wins['black player'] = wins.apply(who_is_black, axis=1)
    wins['outcome'] = wins.apply(who_won, axis=1)
    wins.to_csv(os.path.join(data_dir, 'outcomes.csv'), index=False, header=False)

if __name__ == '__main__':
    main()
