import json
import os
import string
from pyAudioAnalysis import audioBasicIO, ShortTermFeatures

def extract_audio_features_for_word(audio_file, start_time, end_time):
    # Load the audio file
    [Fs, x] = audioBasicIO.read_audio_file(audio_file)
    
    # Convert start_time and end_time to samples
    start_sample = int(float(start_time) * Fs)
    end_sample = int(float(end_time) * Fs)
    
    # Extract the segment
    word_segment = x[start_sample:end_sample]
    
    # Extract features
    features, feature_names = ShortTermFeatures.feature_extraction(word_segment, Fs, 0.050*Fs, 0.025*Fs)
    
    return features.mean(axis=1)  # Returning the mean of the features across the segment

def process_files(textgrid_dir, audio_dir, gold_labels_dir, output_file):
    results = []
    
    # Loop through the gold label text files
    for label_file in os.listdir(gold_labels_dir):
        if label_file.endswith('.txt'):
            # Extract filename (without extension) and gold labels
            audio_filename = label_file.replace('.txt', '.wav')
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
            textgrid_file = os.path.join(textgrid_dir, label_file.replace('.txt', '.TextGrid'))
            audio_file = os.path.join(audio_dir, audio_filename)
            
            if os.path.exists(textgrid_file) and os.path.exists(audio_file):
                # Process the TextGrid file as a normal text file
                file_data = {
                    "filename": audio_filename,
                    "words": []
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
                                
                                # Append word data
                                file_data["words"].append({
                                    "word": word,
                                    "gold_label": label,
                                    "features": features.tolist()  # Convert numpy array to list for JSON serialization
                                })
                            else:
                                print(f"Warning: Word '{sanitized_word}' not found in gold labels.")
                
                results.append(file_data)
    
    # Write the final JSON output
    with open(output_file, 'w') as json_file:
        json.dump(results, json_file, indent=4)



if __name__ == "__main__":
    process_files(textgrid_dir="path_to_textgrid_dir",
                audio_dir="path_to_audio_dir",
                gold_labels_dir="path_to_gold_labels_dir",
                output_file="output.json")
