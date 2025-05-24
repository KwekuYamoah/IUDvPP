"""
    average_json_features.py
    This script processes a JSON file containing multi-label prosody features for words in audio files.
    For each word, it computes the average feature value across all feature vectors and saves the result
    in a new JSON file. The output JSON preserves the original structure but replaces the "features"
    field with a list of [word, average_feature_value] pairs.
    
    Author: Kweku Andoh Yamoah

    Date: 2024-10-18
"""
import json
import numpy as np

# Load existing JSON
with open('../prosody/data/ambiguous_prosody_multi_label_features_eval.json', 'r') as f:
    data = json.load(f)

new_data = {}

for key, value in data.items():
    new_entry = {
        "filepath": value["filepath"],
        "words": value["words"],
        "positions": value["positions"],
        "labels": value["labels"],
    }
    
    words = value["words"]
    features = value["features"]
    
    # Compute average feature vector for each word
    avg_features = []
    for i in range(len(words)):
        word_features = [features[j][i] for j in range(len(features))]
        avg_feature_value = np.mean(word_features)
        avg_features.append([words[i], avg_feature_value])
    
    new_entry["features"] = avg_features
    new_data[key] = new_entry

# Save the new JSON
with open('../prosody/data/ambiguous_prosody_multi_label_features_eval_averaged.json', 'w') as f:
    json.dump(new_data, f, indent=2)

print("New JSON file saved as output.json")