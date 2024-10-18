import json

# load json data
with open('../prosody/data/multi_label_features.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

ids_to_extract = ['003', '014', '017', '013', '032']

# dict to hold training data
training_data = {}

# dict to hold test data
test_data = {}

# go through data
for file in data:
    #check if file string contains any of the ids to extract
    if any(id in file for id in ids_to_extract):
        test_data[file] = data[file]
    else:
        training_data[file] = data[file]

# save training data
with open('../prosody/data/prosody_multi_label_features_train.json', 'w', encoding='utf-8') as f:
    json.dump(training_data, f, ensure_ascii=False, indent=2)

# save test data
with open('../prosody/data/prosody_multi_label_features_eval.json', 'w', encoding='utf-8') as f:
    json.dump(test_data, f, ensure_ascii=False, indent=2)
