import json
from difflib import SequenceMatcher

# Load the JSON data
with open('../prosody/reconstructed_extracted_features.json', encoding="utf-8") as f:
    features_data = json.load(f)

with open('../ltl/instructions_data.json', encoding="utf-8") as f:
    instructions_data = json.load(f)

# Function to normalize and split words
def normalize_and_split(word):
    return word.lower().split('_')

# Function to check if a word belongs to a referent list
def check_word_in_referents(word, referents):
    return any(word in normalize_and_split(referent) for referent in referents)

# Function to calculate word match percentage
def match_percentage(reconstructed_sentence, instruction):
    words_reconstructed = set(reconstructed_sentence.split())
    words_instruction = set(instruction.lower().split())
    intersection = words_reconstructed.intersection(words_instruction)

    
    return len(intersection) / len(words_instruction)

# Iterate over features.json and process each entry
for feature_id, feature in features_data.items():
    reconstructed_sentence = ' '.join(feature['words']).lower()

    for instruction_entry in instructions_data:
        instruction = instruction_entry['instruction'].lower()

        if match_percentage(reconstructed_sentence, instruction) >= 0.7:
            
            new_labels = []
            for word in feature['words']:
                word = word.lower()
                
                if check_word_in_referents(word, instruction_entry['goal_intent_referents']):
                    new_labels.append(1)
                elif check_word_in_referents(word, instruction_entry['avoidance_intent_referents']):
                    new_labels.append(3)
                elif check_word_in_referents(word, instruction_entry['detail_intent_referents']):
                    new_labels.append(2)
                else:
                    new_labels.append(0)
            feature['labels'] = new_labels
            break

# Save the modified features.json
with open('multi_label_features.json', 'w') as f:
    json.dump(features_data, f, indent=4)

print("Labels updated and saved to modified_features.json")
