import json

def reconstruct_json(original_data):
    reconstructed = []
    
    for entry in original_data:
        audio_file = entry.get("audio_file", "")
        words = entry.get("words", [])
        gold_labels = entry.get("gold_labels", [])
        predicted_labels = entry.get("predicted_labels", [])
        
        # Reconstruct the full sentence
        sentence = ' '.join(words)
        
        # Filter words where predicted_label is 1 or 2
        filtered_words_info = []
        for word, gold, pred in zip(words, gold_labels, predicted_labels):
            if pred in [1, 2]:
                filtered_words_info.append({
                    "word": word,
                    "gold_label": gold,
                    "predicted_label": pred
                })
        
        # Create the new entry
        new_entry = {
            "audio_file": audio_file,
            "sentence": sentence,
            "words_info": filtered_words_info
        }
        
        reconstructed.append(new_entry)
    
    return reconstructed

def main():
    # Load the original JSON data
    with open('prosody_bilstm_multiclass_results.json', 'r') as f:
        original_data = json.load(f)
    
    # Reconstruct the JSON data
    reconstructed_data = reconstruct_json(original_data)
    
    # Save the reconstructed data to a new JSON file
    with open('filtered_prosody_bilstm_multiclass_results.json', 'w') as f:
        json.dump(reconstructed_data, f, indent=4)
    
    print("Reconstruction complete. Check 'filtered_prosody_bilstm_multiclass_results.json' for the output.")

if __name__ == "__main__":
    main()
