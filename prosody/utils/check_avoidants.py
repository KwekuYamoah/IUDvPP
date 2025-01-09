import json

def load_json(file_path='../prosody/data/ambiguous_prosody_multi_label_features_train.json'):
    with open(file_path) as f:
        data = json.load(f)
    return data

def check(data):

    for key, value in data.items():
        for label in value['labels']:
            if label == 3:
                print(f"Found avoidant label in {key}")
            
if __name__ == "__main__":
    data = load_json()
    check(data)