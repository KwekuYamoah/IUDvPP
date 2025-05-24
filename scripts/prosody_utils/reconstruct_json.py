""" 
    reconstruct_json.py
    Overview:
    ----------
    This module provides utility functions for reconstructing and merging JSON-like data structures, 
    primarily for processing and combining features extracted from text and audio files. The main 
    functionalities include:
        - Reconstructing a grouped JSON structure from a list of data entries, with special handling 
        for word cleaning and grouping by filename.
        - Loading and saving JSON files.
        - Merging two JSON datasets by aligning entries on matching keys and combining their features 
        under distinct fields.
    Functions:
    ----------
    - reconstruct_json(data): Groups and cleans word-level data entries by filename, preparing them 
    for JSON serialization.
    - load_json(file_path): Loads JSON data from a specified file.
    - merge_json_entries(json1, json2): Merges two JSON-like dictionaries, combining their features 
    under separate keys for each entry.
    - save_json(data, output_file): Saves a dictionary as a JSON file with pretty formatting.
    Author: Kweku Andoh Yamoah
    Date: 2024-09-30
"""

import json
import string
import re
from collections import defaultdict


def reconstruct_json(data):
    """
    Reconstructs a JSON-like dictionary structure from a list of data entries, grouping information by filename.
    Each entry in the input data should be a dictionary containing at least the following keys:
    - "filename": The name of the file to group by.
    - "filepath": The path to the file.
    - "word": The word to process and include.
    - "label": The label associated with the word.
    - "position": The position of the word.
    - "features": Additional features associated with the word.
    Words are cleaned by removing punctuation except for apostrophes, preserving contractions (e.g., "don't").
    Returns a dictionary where each key is a filename and each value is a dictionary containing:
        - "filepath": The file path.
        - "words": List of cleaned words.
        - "labels": List of labels.
        - "positions": List of positions.
        - "features": List of features.
    Args:
        data (list of dict): List of data entries to reconstruct.
    Returns:
        dict: A dictionary grouped by filename with reconstructed data suitable for JSON serialization.
    """
    reconstructed_data = defaultdict(lambda: {
        "filepath": "",
        "words": [],
        "labels": [],
        "positions": [],
        "features": []
    })
    
    # Define punctuation to remove, excluding the apostrophe
    punctuation_to_remove = string.punctuation.replace("'", "")

    # Regular expression pattern to match words with apostrophes
    contraction_pattern = re.compile(r"\b\w+'\w+\b")

    for entry in data:
        filename = entry["filename"]
        reconstructed_data[filename]["filepath"] = entry["filepath"]

        word = entry["word"]
        # Preserve words with apostrophes (contractions)
        if contraction_pattern.match(word):
            cleaned_word = word
        else:
            # Remove punctuation except for apostrophes
            cleaned_word = word.translate(str.maketrans('', '', punctuation_to_remove))

        reconstructed_data[filename]["words"].append(cleaned_word)
        reconstructed_data[filename]["labels"].append(entry["label"])
        reconstructed_data[filename]["positions"].append(entry["position"])
        reconstructed_data[filename]["features"].append(entry["features"])

    # Convert defaultdict to regular dict for JSON serialization
    reconstructed_data = {k: v for k, v in reconstructed_data.items()}

    return reconstructed_data


def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


def merge_json_entries(json1, json2):
    """
    Merges two JSON-like dictionaries by combining entries with matching keys.
    For each key present in both `json1` and `json2`, this function:
    - Copies all fields from the entry in `json1`, excluding the 'filepath' field.
    - Renames the 'features' field from `json1` to 'prosodic_features'.
    - Adds the 'features' field from `json2` as 'raw_acoustic_features'.
    Args:
        json1 (dict): The first JSON-like dictionary, containing entries with a 'features' field.
        json2 (dict): The second JSON-like dictionary, containing entries with a 'features' field.
    Returns:
        dict: A dictionary containing merged entries for keys present in both input dictionaries.
    """
    merged_data = {}

    for key in json1.keys():
        if key in json2:
            # Step 3: Get data from both JSONs for the same key
            entry1 = json1[key]
            entry2 = json2[key]
            
            # Step 4: Rename 'features' from JSON 1 to 'prosodic_features' 
            # and from JSON 2 to 'raw_acoustic_features'
            merged_entry = {k: v for k, v in entry1.items() if k != 'filepath'}  # Exclude 'filepath'
            merged_entry['prosodic_features'] = merged_entry.pop('features')  # Rename features to prosodic_features
            merged_entry['raw_acoustic_features'] = entry2['features']  # Add the raw acoustic features from json2
            
            # Add the merged entry to the merged_data
            merged_data[key] = merged_entry

    return merged_data


def save_json(data, output_file):
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)



if __name__ == "__main__":
    # # Read the input JSON data from a file
    # input_filename = "combined_extracted_features.json"
    # with open(input_filename, "r") as file:
    #     json_data = json.load(file)

    # # Reconstruct JSON data
    # reconstructed_data = reconstruct_json(json_data)

    # # Write the reconstructed JSON data to an output file
    # output_filename = "reconstructed_extracted_features.json"
    # with open(output_filename, "w") as file:
    #     json.dump(reconstructed_data, file, indent=4)

    # print(f"Reconstructed JSON data has been written to {output_filename}")

    # Paths to the two JSON files
    json1_path = '../prosody/data/ambiguous_prosody_multi_label_features_train.json'
    json2_path = '../prosody/data/ambiguous_raw_extracted_audio_ml_features_train.json'
    
    # Load the JSON data
    json1 = load_json(json1_path)
    json2 = load_json(json2_path)
    
    # Merge the entries
    merged_dataset = merge_json_entries(json1, json2)
    
    # Save the new dataset to a JSON file
    output_file = '../prosody/data/ambiguous_prosodic_raw_acoustic_ml_features_train.json'
    save_json(merged_dataset, output_file)
    
    print(f"New dataset saved to {output_file}")
