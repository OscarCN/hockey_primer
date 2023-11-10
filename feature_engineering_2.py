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

    df['game_seconds'] = df.apply(lambda t: (t['gameEndDateTime'] - t['gameDateTime']).total_seconds() if not pd.isnull(t['gameEndDateTime']) else np.nan, axis=1)
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
    # TODO: CHECK ANGLE DIFFERENCE: MODULE 360, DEGREES, ETC
    df['change_in_angle'] = df.apply(lambda t: abs(t['angle_from_net'] - t['last_event_angle']), axis=1)
    df['speed'] = df.apply(lambda t: t['dist_from_last_event'] / (t['time_from_last_event'] + 1), axis=1)

    df[['gamePk', 'eventTypeId', 'periodTime', 'x', 'y', 'last_event_type_id', 'last_event_x', 'last_event_y',
        'last_event_time', 'time_from_last_event', 'dist_from_last_event', 'rebound']].head(60)



















