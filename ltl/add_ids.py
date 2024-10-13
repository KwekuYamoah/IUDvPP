import json

# Load the JSON file
json_path = "instructions_ambiguous.json"
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
    print(len(data))

# Add unique IDs to each entry
for idx, entry in enumerate(data, start=1):
    entry['id'] = idx

# Save the updated JSON file
updated_json_path = "instructions_ambiguous_with_ids.json"
with open(updated_json_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=4)

print(f"Updated JSON file saved to {updated_json_path}")