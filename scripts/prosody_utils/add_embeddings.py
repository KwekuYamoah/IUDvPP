'''
    A utility function to add embeddings for raw audio set and the raw+prosody set.
'''

import json
import os

def load_json(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def add_embedding(source_json, target_json, new_json):
    """
    Add embeddings from source JSON to target JSON and save the result to a new JSON file.

    Args:
        source_json (str): Path to the source JSON file containing embeddings.
        target_json (str): Path to the target JSON file to which embeddings will be added.
        new_json (str): Path to the new JSON file where the updated target data will be saved.

    Returns:
        None

    The function loads the source and target JSON data, iterates over the source data, and adds the 
    'word_embeddings' from the source to the corresponding entries in the target data. The updated 
    target data is then saved to a new JSON file specified by new_json.
    """
    # Load the source and target JSON data
    source_data = load_json(source_json)
    target_data = load_json(target_json)

    # Iterate over the source_json data
    for key, value in source_data.items():
        # Now find the key in the target_json
        if key in target_data:
            # Add the embeddings to the target_json which is the word_embeddings field in the source_json
            target_data[key]['word_embeddings'] = value.get('word_embeddings', [])

    # Save the target_json
    with open(new_json, 'w') as f:
        json.dump(target_data, f, indent=2)

    # communicate the end of the process
    print("All embeddings added.")

if __name__ == "__main__":
    source_json = '../prosody/data/ambiguous_prosody_multi_label_features_train_embeddings.json'
    target_json = '../prosody/data/ambiguous_prosodic_raw_acoustic_ml_features_train.json'
    new_json = '../prosody/data/ambiguous_prosodic_raw_acoustic_ml_features_train_embeddings.json'

    add_embedding(source_json, target_json, new_json)

    print(f"New JSON file saved as {new_json}")