import sys
import json

with open(sys.argv[1], 'r') as f:
    json_data = json.load(f)

# print(json_data.keys())
print(json_data['info']['categories'])
print()
print(json_data['images'][0])
print()
print(json_data['annotations'][0])