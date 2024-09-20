import os
import json
import string
import numpy as np
from pyAudioAnalysis import audioBasicIO, ShortTermFeatures


def extract_audio_features_for_word(audio_file, start_time, end_time):
    # Load the audio file
    [Fs, x] = audioBasicIO.read_audio_file(audio_file)
    
    # Adjust start and end times to expand the context window
    start_time = max(0, float(start_time) - 0.005)  # Subtract 5ms, but not below 0
    end_time = float(end_time) + 0.010  # Add 5ms
    
    # Convert start_time and end_time to samples
    start_sample = int(start_time * Fs)
    end_sample = int(end_time * Fs)
    
    # Ensure the indices are valid and within the bounds of the audio signal
    if start_sample >= len(x) or end_sample > len(x) or start_sample >= end_sample:
        print(f"Invalid segment: {start_time} - {end_time} for {audio_file}")
        return np.zeros((34,))  # Return zeros as default feature vector
    
    # Extract the segment
    word_segment = x[start_sample:end_sample]
    
    # Check if the segment is non-empty and long enough for analysis
    if len(word_segment) == 0:
        print(f"Empty audio segment between {start_time} and {end_time} in {audio_file}")
        return np.zeros((34,))
    elif len(word_segment) < int(0.015 * Fs):  # If the segment is too short for 5ms window
        print(f"Segment too short for feature extraction: {start_time} - {end_time}")
        return np.zeros((34,))
    
    try:
        # Extract features with a window size of 25ms and a step size of 5ms
        features, feature_names = ShortTermFeatures.feature_extraction(word_segment, Fs, 0.015*Fs, 0.005*Fs, deltas=False)

        # Check if features are empty or if feature extraction failed
        if features.size == 0 or len(features) == 0:
            print(f"No features extracted for the segment {start_time}-{end_time} in {audio_file}")
            return np.zeros((34,))  # Default zero vector for empty feature extraction
        
        return features.mean(axis=1)  # Return the mean of the features across the segment
    
    except Exception as e:
        print(f"Error during feature extraction for segment {start_time}-{end_time}: {e}")
        return np.zeros((34,))  # Return zeros in case of any error


def process_files(textgrid_dir, audio_dir, gold_labels_dir, output_file):
    results = {}

    # Loop through the gold label text files
    for label_file in os.listdir(gold_labels_dir):
        if label_file.endswith('.txt'):
            # Extract filename (without extension) and gold labels
            file_id = label_file.replace('.txt', '')  # The id without file extension
            audio_filename = f"{file_id}.wav"
            gold_labels_file = os.path.join(gold_labels_dir, label_file)
            
            # Load the gold labels (word and corresponding label)
            gold_labels = {}
            with open(gold_labels_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        word, label = parts
                        gold_labels[word.lower().strip(string.punctuation)] = label  # Sanitize word
            
            # Find corresponding TextGrid and audio file
            textgrid_file = os.path.join(textgrid_dir, f"{file_id}.TextGrid")
            audio_file = os.path.join(audio_dir, audio_filename)
            
            if os.path.exists(textgrid_file) and os.path.exists(audio_file):
                # Initialize the data for this file
                file_data = {
                    "words": [],
                    "labels": [],
                    "features": []
                }
                
                with open(textgrid_file, 'r') as tg:
                    for line in tg:
                        parts = line.strip().split()
                        if len(parts) >= 3:
                            start_time, end_time, word = parts[0], parts[1], parts[2]
                            sanitized_word = word.lower().strip(string.punctuation)  # Remove punctuation, lowercase
                            
                            # Perform sanity check with gold labels
                            if sanitized_word in gold_labels:
                                label = gold_labels[sanitized_word]
                                
                                # Extract audio features for the word
                                features = extract_audio_features_for_word(audio_file, start_time, end_time)
                                
                                # Append data for this word
                                file_data["words"].append(sanitized_word)
                                file_data["labels"].append(int(label))  # Assuming labels are integers
                                file_data["features"].append(features.tolist())  # Convert numpy array to list for JSON serialization
                            else:
                                print(f"Warning: Word '{sanitized_word}' not found in gold labels.")
                
                results[file_id] = file_data  # Use file ID as the key
    
    # Write the final JSON output
    with open(output_file, 'w') as json_file:
        json.dump(results, json_file, indent=4)


if __name__ == "__main__":
    text_grid_dir = "../prosody/text_grid_files_v2"
    audio_dir = "../voice_samples"
    gold_labels_dir = "../prosody/gold_label_txt"
    output_file = "../prosody/data/extracted_audio_features.json"

    process_files(textgrid_dir=text_grid_dir,
                  audio_dir=audio_dir,
                  gold_labels_dir=gold_labels_dir,
                  output_file=output_file)
