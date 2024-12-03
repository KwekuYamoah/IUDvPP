import json

# def count_json_entries(file_path):
#     with open(file_path, 'r') as file:
#         data = json.load(file)
#         return len(data)

# # Example usage
# file_path = 'filtered_prosody_bilstm_multiclass_results.json'
# entry_count = count_json_entries(file_path)
# print(f'The JSON file contains {entry_count} entries.')


def count_matches(file_path, instructions):
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    match_dict = {instruction: 0 for instruction in instructions}
    non_match_dict = {}

    for entry in data:
        # Normalize the sentence from the JSON entry
        sentence = entry['sentence'].strip().lower().rstrip('.')
        matched = False
        
        for instruction in instructions:
            # Normalize the instruction
            normalized_instruction = instruction.strip().lower().rstrip('.')
            if sentence == normalized_instruction:
                match_dict[instruction] += 1
                matched = True
                break
        
        if not matched:
            non_match_dict[entry['audio_file']] = entry['sentence']
    
    return match_dict, non_match_dict

# Example usage
file_path = 'filtered_prosody_bilstm_multiclass_results.json'
instructions = [
    "Pick up the book on the table with the red cover.",
    "Place the vase near the flowers on the table.",
    "Place the coke can beside the pringles on the counter.",
    "Bring the book and the magazine on the nightstand.",
    "Bring the mug from the table near the sink."
]

match_dict, non_match_dict = count_matches(file_path, instructions)

print("Match Counts:")
for instruction, count in match_dict.items():
    print(f"Instruction: {instruction}, Matches: {count}")

print("\nNon-Matching Entries:")
for audio_file, sentence in non_match_dict.items():
    print(f"Audio File: {audio_file}, Sentence: {sentence}")

    # write the non-matching entries to a new JSON file
    with open('non_matching_entries.json', 'w') as f:
        json.dump(non_match_dict, f)