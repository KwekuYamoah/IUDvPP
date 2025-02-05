import json


def reconstruct_json(original_data):
    
    
    """
    Reconstructs a JSON structure from the original data by filtering and organizing words based on their predicted labels.
    Args:
        original_data (list): A list of dictionaries, where each dictionary contains the following keys:
            - "audio_file" (str): The path to the audio file.
            - "words" (list): A list of words in the sentence.
            - "gold_labels" (list): A list of gold standard labels corresponding to the words.
            - "predicted_labels" (list): A list of predicted labels corresponding to the words.
    Returns:
        list: A list of dictionaries, where each dictionary contains:
            - "audio_file" (str): The path to the audio file.
            - "sentence" (str): The reconstructed full sentence.
            - "words_info" (list): A list of dictionaries for words with predicted labels 1 or 2, each containing:
                - "word" (str): The word.
                - "gold_label" (int): The gold standard label for the word.
                - "predicted_label" (int): The predicted label for the word.
            - "intent_dict" (dict): A dictionary with keys "Goal_intent", "Avoidance_intent", and "Detail_intent", each containing a list of words corresponding to the predicted labels.
    """
    reconstructed = []

    for entry in original_data:
        audio_file = entry.get("audio_file", "")
        words = entry.get("words", [])
        gold_labels = entry.get("gold_labels", [])
        predicted_labels = entry.get("predicted_labels", [])

        # Reconstruct the full sentence
        sentence = ' '.join(words)

        # Initialize intent dictionary
        intent_dict = {
            'Goal_intent': [],
            'Avoidance_intent': [],
            'Detail_intent': []
        }

        # Filter words where predicted_label is 1 or 2
        filtered_words_info = []
        for word, gold, pred in zip(words, gold_labels, predicted_labels):
            if pred in [1, 2]:
                filtered_words_info.append({
                    "word": word,
                    "gold_label": gold,
                    "predicted_label": pred
                })

            # Add words to intent_dict based on predicted_label
            if pred == 1:
                intent_dict['Goal_intent'].append(word)
            elif pred == 2:
                intent_dict['Detail_intent'].append(word)
            elif pred == 3:
                intent_dict['Avoidance_intent'].append(word)

        # Create the new entry
        new_entry = {
            "audio_file": audio_file,
            "sentence": sentence,
            "words_info": filtered_words_info,
            "intent_dict": intent_dict
        }

        reconstructed.append(new_entry)

    return reconstructed


def main():
    """
    Main function to load, reconstruct, and save JSON data.
    This function performs the following steps:
    1. Loads the original JSON data from a specified file.
    2. Reconstructs the JSON data using the `reconstruct_json` function.
    3. Saves the reconstructed data to a new JSON file.
    4. Prints a message indicating the completion of the reconstruction process.
    Note:
        The paths to the input and output JSON files are hardcoded within the function.
    """
    # Load the original JSON data
    with open('../ltl/data/prosody_bilstm_multiclass_results.json', 'r') as f:
        original_data = json.load(f)

    # Reconstruct the JSON data
    reconstructed_data = reconstruct_json(original_data)

    # Save the reconstructed data to a new JSON file
    with open('../ltl/data/filtered_prosody_bilstm_multiclass_results.json', 'w') as f:
        json.dump(reconstructed_data, f, indent=4)

    print("Reconstruction complete. Check 'filtered_prosody_bilstm_multiclass_results.json' for the output.")


if __name__ == "__main__":
    main()
