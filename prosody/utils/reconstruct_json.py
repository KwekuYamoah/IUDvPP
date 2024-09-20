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

# Read the input JSON data from a file
input_filename = "combined_extracted_features.json"
with open(input_filename, "r") as file:
    json_data = json.load(file)

# Reconstruct JSON data
reconstructed_data = reconstruct_json(json_data)

# Write the reconstructed JSON data to an output file
output_filename = "reconstructed_extracted_features.json"
with open(output_filename, "w") as file:
    json.dump(reconstructed_data, file, indent=4)

print(f"Reconstructed JSON data has been written to {output_filename}")
