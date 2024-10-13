import json
import re
from collections import defaultdict

def preprocess(text):
    # Remove punctuation, convert to lowercase, and split by spaces
    tokens = re.sub(r'[^\w\s]', '', text).lower().split()
    return tokens

def find_matching_keys(file_path):
    # Load data from the JSON file
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Preprocess values by tokenizing and normalizing text
    preprocessed_data = {k: preprocess(v) for k, v in data.items()}

    # Dictionary to store counts of matching keys
    matching_counts = defaultdict(int)

    # Find matches
    for key1, tokens1 in preprocessed_data.items():
        for key2, tokens2 in preprocessed_data.items():
            if key1 != key2:
                # Calculate match percentage
                match_count = sum(1 for token in tokens1 if token in tokens2)
                match_percentage = match_count / len(tokens1)
                
                # Check if match percentage is >= 80%
                if match_percentage >= 0.80:
                    matching_counts[key1] += 1

    # Filter for keys that have more than 2 matches
    result_keys = [key for key, count in matching_counts.items() if count > 2]

    # Print the result keys and their corresponding values
    for key in result_keys:
        print(f"{key}: {data[key]}")


def reorder_json(file_path):
    # Load data from the JSON file
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Sort the data by the numeric part of the keys
    sorted_data = dict(sorted(data.items(), key=lambda item: int(re.search(r'\d+', item[0]).group())))

    # Save the sorted data back to the file (optional)
    with open(file_path, 'w') as f:
        json.dump(sorted_data, f, indent=4)

    # Print the sorted data
    for key, value in sorted_data.items():
        print(f"{key}: {value}")
        

if __name__ == "__main__":
    # Example usage:
    file_path = 'transcriptions_find_error.json'
    reorder_json(file_path)
