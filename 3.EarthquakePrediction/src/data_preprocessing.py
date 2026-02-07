import pandas as pd
import datetime
import time

def load_and_clean_data(path):
    data = pd.read_csv(path)

    data = data[['Date', 'Time', 'Latitude', 'Longitude', 'Depth', 'Magnitude']]

    timestamps = []
    for d, t in zip(data['Date'], data['Time']):
        try:
            ts = datetime.datetime.strptime(
                d + ' ' + t, '%m/%d/%Y %H:%M:%S'
            )
            timestamps.append(time.mktime(ts.timetuple()))
        except:
            timestamps.append(None)

    data['Timestamp'] = timestamps
    data.dropna(inplace=True)

    return data
