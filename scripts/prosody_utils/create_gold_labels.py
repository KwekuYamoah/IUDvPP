""" 
    create_gold_labels.py
    Overview:
        This script processes a JSON file containing audio file metadata, specifically word and label annotations, 
        and writes the word-label pairs to individual text files for each audio file. The script supports both 
        list and dictionary JSON structures and ensures that each output file is named after the corresponding 
        audio file, with the extension changed from .wav to .txt. Each output text file contains one word-label 
        pair per line, facilitating downstream processing or evaluation tasks.
    
    Author: Kweku Andoh Yamoah
    Date: 2024-09-30

"""
import json
import os

def write_word_label_pairs(json_file, output_dir):
    """Writes word-label pairs from a JSON file to individual text files.
    This function processes a JSON file containing audio file information with associated
    words and labels, and writes them to separate text files in the specified output directory.
    The function can handle both list and dictionary JSON structures.
    Args:
        json_file (str): Path to the input JSON file containing the word-label data.
        output_dir (str): Directory path where the output text files will be saved.
    The JSON file should have either of these structures:
        - List of dictionaries: [{audio_file: {words: [...], labels: [...]}, ...}]
        - Dictionary: {audio_file: {words: [...], labels: [...]}, ...}
    Each output file will be named after the corresponding audio file (with .wav replaced by .txt)
    and will contain word-label pairs, one per line.
    Example output file format:
        word1 label1
        word2 label2
        ...
    Returns:
        None
    Raises:
        JSONDecodeError: If the JSON file is invalid.
        IOError: If there are issues reading the input file or writing output files.
    """
    # Load the JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Check if data is a list
    if isinstance(data, list):
        for entry in data:
            for audio_file, content in entry.items():
                # Get the filename without the directory and extension
                filename = os.path.basename(audio_file).replace('.wav', '.txt')
                
                # Prepare the path for the output text file
                output_file = os.path.join(output_dir, filename+'.txt')
                
                # Extract words and labels
                words = content.get("words", [])
                labels = content.get("labels", [])
                
                # Write the words and labels to the text file
                with open(output_file, 'w') as f_out:
                    for word, label in zip(words, labels):
                        f_out.write(f"{word} {label}\n")
    else:
        # Handle the case where data is not a list but a dictionary
        for audio_file, content in data.items():
            # Get the filename without the directory and extension
            filename = os.path.basename(audio_file).replace('.wav', '.txt')
            
            # Prepare the path for the output text file
            output_file = os.path.join(output_dir, filename+'.txt')
            
            # Extract words and labels
            words = content.get("words", [])
            labels = content.get("labels", [])
            
            # Write the words and labels to the text file
            with open(output_file, 'w') as f_out:
                for word, label in zip(words, labels):
                    f_out.write(f"{word} {label}\n")

    print(f"Word-label pairs successfully written to {output_dir}")


if __name__ == "__main__":

    json_file = '../prosody/data/prosody_multi_label_features_train.json'
    output_dir = 'gold_label_txt/text_grid_files_set1/train'
    write_word_label_pairs(json_file, output_dir)
