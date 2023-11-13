from datetime import datetime

import numpy as np
import pandas as pd


def parse_game_date(t):
    if not pd.isnull(t):
        return datetime.strptime(t[:-4], '%Y-%m-%dT%M:%S')
    else:
        return np.nan


def add_features2(df):
    df = df.copy()

    df['gameDateTime'] = df['gameDateTime'].map(parse_game_date)
    df['gameEndDateTime'] = df['gameEndDateTime'].map(parse_game_date)

    df['game_seconds'] = df.apply(
        lambda t:
            (t['gameEndDateTime'] - t['gameDateTime']).total_seconds()
            if not pd.isnull(t['gameEndDateTime'])
            else np.nan,
        axis=1)
    # BAD DATA: DROP GAMES WITH MORE THAN 200 SECONDS

    df = df.sort_values(['gamePk', 'periodTime'], ascending=[True, True])
    df['last_event_type_id'] = df.groupby('gamePk')['eventTypeId'].shift(1)
    df['last_event_x'] = df.groupby('gamePk')['x'].shift(1)
    df['last_event_y'] = df.groupby('gamePk')['y'].shift(1)
    df['last_event_time'] = df.groupby('gamePk')['periodTime'].shift(1)
    df['last_event_angle'] = df.groupby('gamePk')['angle_from_net'].shift(1)

    def period_time_distance(pt1, pt2):
        if (not isinstance(pt2, str)) or (not isinstance(pt1, str)):
            return np.nan

        dt1 = datetime(2000, 1, 1, 0, int(pt1.split(':')[0]), int(pt1.split(':')[1]))
        dt2 = datetime(2000, 1, 1, 0, int(pt2.split(':')[0]), int(pt2.split(':')[1]))

        return (dt2 - dt1).total_seconds()

    df['time_from_last_event'] = df.apply(
        lambda t: period_time_distance(t['last_event_time'], t['periodTime']),
        axis=1
    )
    df['dist_from_last_event'] = df.apply(
        lambda t: np.linalg.norm(np.array([t['x'], t['y']]) - np.array([t['last_event_x'], t['last_event_y']])),
        axis=1
    )

    df['rebound'] = df.apply(lambda t: t['eventTypeId'] == t['last_event_type_id'], axis=1)
    df['change_in_angle'] = df.apply(lambda t: abs(t['angle_from_net'] - t['last_event_angle']), axis=1)
    df['speed'] = df.apply(lambda t: t['dist_from_last_event'] / (t['time_from_last_event'] + 1), axis=1)

    df[['gamePk', 'eventTypeId', 'periodTime', 'x', 'y', 'last_event_type_id', 'last_event_x', 'last_event_y',
        'last_event_time', 'time_from_last_event', 'dist_from_last_event', 'rebound', 'angle_from_net', 'last_event_angle', 'change_in_angle', 'speed']].head(60)

    return df


def infer_side(tidied_training_set, tidied_test_set):

    a = pd.concat((tidied_training_set, tidied_test_set))
    sides = a.groupby(['gamePk', 'period', 'team_name']).agg(
        {'x': 'median'})  # , 'rink_side': lambda t: [k for k in t if not pd.isnull(k)][0]})
    sides = sides.reset_index()

    sides['norm_x'] = sides.apply(lambda t: t['x'] * ((t['period'] % 2) * 2 - 1), axis=1)

    norm_sides = sides.groupby(['gamePk', 'team_name'])['norm_x'].median().reset_index()

    # norm_sides_max = norm_sides.groupby(['gamePk']).norm_x.max().reset_index().rename(columns={'norm_x': 'max_x'})
    norm_sides_min = norm_sides.groupby(['gamePk']).norm_x.min().reset_index().rename(columns={'norm_x': 'min_x'})

    norm_sides = norm_sides.merge(norm_sides_min)

    norm_sides['period_1_side'] = norm_sides.apply(lambda t: ('right' if t['norm_x'] == t['min_x'] else 'left'), axis=1)

    #norm_sides[['gamePk', 'team_name', 'norm_x', 'period_1_side']].to_csv('resources/period_1_sides.csv', index=False)

















