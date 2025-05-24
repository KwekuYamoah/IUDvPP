"""
This script updates the transcriptions in a JSON file with corresponding instructions from another JSON file.
The script performs the following steps:
1. Loads the transcriptions from "transcriptions_ambiguous_instructions_prosody.json".
2. Loads the instructions from "../ltl/data/instructions_ambiguous_with_ids.json".
3. Iterates over each instruction and updates the corresponding transcription in the data based on matching IDs.
4. Saves the updated transcriptions back to "transcriptions_ambiguous_instructions_prosody.json".
Functions:
- None
Dependencies:
- json
- os
Filepaths:
- Input: "transcriptions_ambiguous_instructions_prosody.json"
- Input: "../ltl/data/instructions_ambiguous_with_ids.json"
- Output: "transcriptions_ambiguous_instructions_prosody.json"
"""

import json
import os

with open("transcriptions_ambiguous_instructions_prosody.json", "r") as file:
    data = json.load(file)

with open("../ltl/data/instructions_ambiguous_with_ids.json", "r") as file:
    instructions = json.load(file)

for instruction in instructions:
    for key, value in data.items():
        # get filename without extension
        filename = os.path.basename(key).replace(".wav", "")
        # split file at _ and get index 2
        file_index = filename.split("_")[2]
        #if id in instructions matches the file index
        if instruction["id"] == int(file_index):
            # update the value for this key with instruction["instruction"].lower()
            data[key] = instruction["instruction"].lower()

with open("transcriptions_ambiguous_instructions_prosody.json", "w") as file:
    json.dump(data, file, indent=2)
           