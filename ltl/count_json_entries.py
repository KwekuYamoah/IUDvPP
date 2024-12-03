import json

def count_json_entries(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        return len(data)

# Example usage
file_path = 'filtered_prosody_bilstm_multiclass_results.json'
entry_count = count_json_entries(file_path)
print(f'The JSON file contains {entry_count} entries.')
