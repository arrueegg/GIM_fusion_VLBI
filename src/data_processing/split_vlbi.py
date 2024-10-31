
# Online Python - IDE, Editor, Compiler, Interpreter
import json

# Load the JSON file
with open('/scratch2/arrueegg/WP1/PyIono_Output/Results/station_coords.json', 'r') as f:
    data = json.load(f)

# Get all keys from the dictionary
keys_list = list(data.keys())

print("Number of Stations: ", len(keys_list))

# Save the list to a text file
with open('./src/data_processing/sit_all_vlbi.list', 'w') as f:
    for item in keys_list:
        f.write(f"{item}\n")


validation_stations = []
with open('./src/data_processing/sit_val_vlbi.list', 'w') as f:
    for item in validation_stations:
        f.write(f"{item}\n")

test_stations = ['HOBART12']    # or GGAO12M or WESTFORD or WETTZ13S or HOBART12 or WETTZELL
with open('./src/data_processing/sit_test_vlbi.list', 'w') as f:
    for item in test_stations:
        f.write(f"{item}\n")

train_list = [item for item in keys_list if item not in (validation_stations + test_stations)]
with open('./src/data_processing/sit_train_vlbi.list', 'w') as f:
    for item in train_list:
        f.write(f"{item}\n")