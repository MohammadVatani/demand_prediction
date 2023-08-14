import pandas as pd


df = pd.read_csv('api/data/location_ids/segmentation.csv')

res = df.to_dict('records')

high = [i['location_id'] for i in res if i['segment'] == 'HIGH']
mid = [i['location_id'] for i in res if i['segment'] == 'MID']
low = [i['location_id'] for i in res if i['segment'] == 'LOW']

result = {}

result['HIGH'] = high
result['MID'] = mid
result['LOW'] = low

import json

with open('api/data/location_ids/segmentation.json', 'w', encoding='utf-8') as f:
    json.dump(result, f)