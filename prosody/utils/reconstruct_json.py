import json
import string
import re
from collections import defaultdict

# Function to reconstruct the JSON
def reconstruct_json(data):
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


# Step 1: Load the two JSON files
def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# Step 2: Check if top-level keys are the same and merge entries
def merge_json_entries(json1, json2):
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

# Step 5: Save the merged dataset to a new JSON file
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
    json1_path = '../prosody/data/multi_label_features.json'
    json2_path = '../prosody/data/multi_label_extracted_raw_audio_features.json'
    
    # Load the JSON data
    json1 = load_json(json1_path)
    json2 = load_json(json2_path)
    
    # Merge the entries
    merged_dataset = merge_json_entries(json1, json2)
    
    # Save the new dataset to a JSON file
    output_file = '../prosody/data/multi_label_prosodic_raw_acoustic_features.json'
    save_json(merged_dataset, output_file)
    
    print(f"New dataset saved to {output_file}")
