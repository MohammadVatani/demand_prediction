import json


with open('data/location_ids/segmentation.json') as file:
    data = json.load(file)

high_demands = data['HIGH']
low_demands = data['LOW']
mid_demands = data['MID']

START_TRAINING_DATE = '2023-01-01'
END_TRAINING_DATE = '2023-04-31'

NUMBER_INTERVAL_PER_DAY = 8
