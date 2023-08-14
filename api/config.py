import json


with open('data/location_ids/segmentation.json') as file:
    data = json.load(file)

high_demands = data['HIGH']
low_demands = data['LOW']
mid_demands = data['MID']