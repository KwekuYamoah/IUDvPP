import os
import json
import shutil


import torch
from torch import Tensor as tensor

from disvoice.prosody import Prosody
from pydub import AudioSegment


#import disvoice

def extract_prosody_features(audio_path):
    '''
    extract_prosody_features takes in the path to an audio file and returns
    a pytorch tensor containing the extracted prosodic features.

    Params:
        audio_path (string): This represents the path to the audio file
    
    Returns:
        prosody_features (tensor): This is a pytorch tensor that contains the extracted
                                prosodic features.
    '''
    #create the prosody feature extractor object
    prosody_extractor = Prosody()
    
    #extract the prosody features
    prosody_features = prosody_extractor.extract_features_file(audio_path, static=True, plots=False, fmt='torch')

  

    return prosody_features


def construct_feature_json(previous_json_path, new_json_storage_path):
    '''
    construct_feature_json takes in the previous json path containing the previously extracted features and then
    calls the extract prosody features function to extract new input features. After that it writes these new
    features and their corresponding information to the new json storage path.

    Params:
        previous_json_path (string): This is the old json file that contains the current extracted prosody features and audio information.
        new_json-storage_path (string): This is the new storage json file to house the new extracted prosody features and audio information.

    Returns:
        None
    '''
    new_json_file_contents = []

    #read the contents of the old json file
    with open(previous_json_path, 'r') as input_file:
        input_data = json.load(input_file)
    
    #iterate through the input data and for each entry extract new prosodic features
    for item in input_data:
        file_path = list(item.keys())[0]
        new_features = extract_prosody_features(file_path)

        new_json_file_contents.append({file_path: {'input_features': new_features.tolist(), 'labels': item[file_path]['labels'], 'words': item[file_path]['words']}})
    

    with open(new_json_storage_path, 'w') as output_file:
        json.dump(new_json_file_contents, output_file)


    return



def restructure_json_objects(json_object_path):
    with open(json_object_path, 'r') as inputfile:
        input_data = json.load(inputfile)


    #this is the list to hold the extracted prosody features of each of the words
    # in the audio files.
    general_feature_matrix = []

    words_matrix = []

    word_position_matrix = []

    current_file_feature_matrix = []

    current_file_word_matrix = []

    current_file_word_position_matrix = []

    previous_filename = input_data[0]['filename']

    #iterate through the items in json_objects
    for json_item_index in range(len(input_data)):
        #obtain the filename
        current_filename = input_data[json_item_index]['filename'] 

        #obtain the extracted prosody features
        current_prosody_features = input_data[json_item_index]['features']

        #obtain the current word
        current_word = input_data[json_item_index]['word']

        #obtain the current position
        current_position = input_data[json_item_index]['position']

        #check if the current filename is the same as the previous filename
        if current_filename == previous_filename:
            current_file_word_matrix.append(current_word)
            current_file_feature_matrix.append(current_prosody_features)
            current_file_word_position_matrix.append(current_position)

            #set the previous filename to be equal to the current filename
            previous_filename = current_filename
        

        #get the next filename
        if json_item_index + 1 < len(input_data):
            next_filename = input_data[json_item_index+1]['filename']

            if next_filename != current_filename:
                #push all of the accumulated information obtained so far
                general_feature_matrix.append(current_prosody_features)
                words_matrix.append(current_file_word_matrix)
                word_position_matrix.append(current_file_word_position_matrix)

                #clear all of the current file information lists
                current_file_word_matrix = []
                current_file_feature_matrix = []
                current_file_word_position_matrix = []

                #set the previous filename to be equal to the next filename
                previous_filename = next_filename
        
        
    
    #print('words matrix: ', words_matrix)
    print(general_feature_matrix)
    return




# Utility functions
def load_json(json_path):
    """Load the JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def empty_temp_folder(temp_folder):
    """Empty the temporary folder."""
    if os.path.exists(temp_folder):
        shutil.rmtree(temp_folder)
    os.makedirs(temp_folder)

# Function to normalize and split words
def normalize_and_split(word):
    return word.lower().split(' ')

# Function to check if a word belongs to a referent list
def check_word_in_referents(word, referents):
    return any(word in normalize_and_split(referent) for referent in referents)

def create_audio_slices(audio_path, textgrid_path, output_folder):
    """Create audio slices from the TextGrid file."""
    # Load the audio file
    audio = AudioSegment.from_file(audio_path)
    print(audio)

    # Load the TextGrid file and read its contents
    with open(textgrid_path, 'r') as f:
        file_contents = f.readlines()

    # Iterate through each line in the TextGrid file
    for index, line in enumerate(file_contents):
        # Split the line into start time, end time, and word
        start_time_str, end_time_str, word = line.strip().split('\t')
        start_time = float(start_time_str)
        end_time = float(end_time_str)

        # Skip empty words
        if not word:
            continue

        # Create a buffer window around the word, except for the first word
        if index != 0:
            start_time = max(0, start_time - 0.010)  # Subtract 5 milliseconds
        end_time = end_time + 0.035  # Add 35 milliseconds

        # Create a slice of the original audio
        start_ms = int(start_time * 1000)  # Convert to milliseconds
        end_ms = int(end_time * 1000)  # Convert to milliseconds
        audio_slice = audio[start_ms:end_ms]

        # Construct the filename
        filename = f"{os.path.basename(audio_path).split('.')[0]}_{word}_{index}_{start_time:.3f}_{end_time:.3f}.wav"
        output_path = os.path.join(output_folder, filename)

        # Export the slice
        audio_slice.export(output_path, format="wav")
        print(f"Saved: {output_path}")

def prepare_features_json(input_folder, output_json_path, data):
    """
    Processes audio slices to extract prosodic features and save them in a JSON file.

    Args:
        input_folder (str): Folder containing sliced audio files.
        output_json_path (str): Path to save the JSON file.
        data (list): JSON data containing intent groundtruth information.
    """
    # Create a dictionary to store the results
    results = {}

    # Iterate through each file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".wav"):
            # Construct the full file path
            audio_path = os.path.join(input_folder, filename)

            # Extract the root name, word, and position from the filename
            parts = filename.split('_')
            

            root_name = '_'.join(parts[0:4]) # get root name from index 0-3
            word = parts[4].rstrip('.')  # Remove full stop if it exists
            position = int(parts[5])
            start_time = float(parts[6])
            end_time = float(parts[7].replace('.wav', ''))

            # Extract prosodic features
            features = extract_prosody_features(audio_path)
            features[torch.isnan(features)] = 0

            # Convert the tensor to a list
            features_list = features.tolist()

            # Get current directory
            current_dir = os.getcwd()

            # Prepare data structure for the file
            if root_name not in results:
                results[root_name] = {
                    "filepath": os.path.join(current_dir, root_name + ".wav"),
                    "words": [],
                    "positions": [],
                    "features": [],
                    "labels": []
                }

            # Append the word, position, and features to the respective lists
            results[root_name]["words"].append((position, word))
            results[root_name]["positions"].append((position, position))
            results[root_name]["features"].append((position, features_list))

            # Determine label based on JSON data
            id = int(root_name.split('_')[2])  # Extract ID from root name
            interpretation_index = 0 if 'i1' in root_name else 1
            json_entry = next((entry for entry in data if entry['id'] == id), None)
            if json_entry:
                intent_groundtruth = json_entry['intent_groundtruth'][interpretation_index]
                if check_word_in_referents(word, intent_groundtruth['goal_intent_referents']):
                    label = 1
                elif check_word_in_referents(word, intent_groundtruth['detail_intent_referents']):
                    label = 2
                elif check_word_in_referents(word, intent_groundtruth['avoidance_intent_referents']):
                    label = 3
                else:
                    label = 0
            else:
                label = 0

            results[root_name]["labels"].append((position, label))

    # Sort the words, positions, features, and labels by their positions
    for root_name, data in results.items():
        data["words"] = [word for position, word in sorted(data["words"])]
        data["positions"] = [position for position, _ in sorted(data["positions"])]
        data["features"] = [features for position, features in sorted(data["features"])]
        data["labels"] = [label for position, label in sorted(data["labels"])]

    # Write the results to a JSON file
    with open(output_json_path, 'w') as json_file:
        json.dump(results, json_file, indent=4)
    print(f"Saved results to {output_json_path}")

# Main function
def main():
    # Define paths
    json_path = "../ltl/instructions_ambiguous_with_ids.json"
    voice_samples_folder = "../voice_sample_two_ways"
    text_grid_folder = "./text_grid_files_set2"
    temp_folder = "tmp/"
    output_json_path = "./data/ambiguous_prosody_multi_label_features.json"

    # Load JSON data
    data = load_json(json_path)

    # Empty the temp folder
    empty_temp_folder(temp_folder)

    # Recursively process all .wav files
    for root, _, files in os.walk(voice_samples_folder):
        for filename in sorted(files):
            if filename.endswith('.wav'):
                audio_path = os.path.join(root, filename)
                root_name = os.path.splitext(filename)[0]
                textgrid_path = os.path.join(text_grid_folder, f"{root_name}.TextGrid")

                if os.path.exists(textgrid_path):
                    create_audio_slices(audio_path, textgrid_path, temp_folder)
                else:
                    print(f"TextGrid file not found for {audio_path}")

    # Prepare features JSON from the sliced audio files
    prepare_features_json(temp_folder, output_json_path, data)

    # Empty the temp folder
    empty_temp_folder(temp_folder)

if __name__ == "__main__":
    main()
