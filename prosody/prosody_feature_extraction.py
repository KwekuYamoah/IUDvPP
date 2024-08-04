import torch
from torch import Tensor as tensor
from disvoice.prosody import Prosody
import json

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


#extract_prosody_features('/Users/dasa/Desktop/prosody_ltl/IUDvPP-main/language_files/en/pm04_in_003.wav')
#construct_feature_json('./en_extracted_features.json', './new_en_extracted_features.json')
# restructure_json_objects('./combined_extracted_features.json')
