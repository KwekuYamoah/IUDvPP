import json
import os

def write_word_label_pairs(json_file, output_dir):
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

# Example usage
json_file = '../prosody/data/reconstructed_extracted_features.json'
output_dir = 'gold_label_txt'
write_word_label_pairs(json_file, output_dir)
