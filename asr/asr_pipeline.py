"""
Author: Kweku Andoh Yamoah
Date: 2021-09-30
Last Modified: 2021-09-30
Version: 1.0

This file contains the ASR pipeline for the project, IUDvPP.

Approach:
First load the whisper Model.

Define a function that takes a path to a folder with audio files:
    - Load the audio files.
    - Transcribe the audio files.
    - Store the audio file name and the transcription in a dictionary.
    - Convert the dictionary to a json file.
    - Return the json file.
"""



# importing the necessary libraries
import os
import json
import whisper



def asr_pipeline(audio_folder: str) -> str:
    """
    This function takes in a folder containing
    audio files and transcribes them using the
    whisper model. The transcriptions are then
    stored in a json file and returned.

    Parameters:
    - audio_folder (str): The path to the folder containing audio files.

    Returns:
    - json_file (str): The path to the generated json file containing the transcriptions.
    """
    model = whisper.load_model("medium.en")

    # Get the list of audio files in the folder
    audio_files = os.listdir(audio_folder)

    # Create a dictionary to store the transcriptions
    transcriptions = {}

    # Loop through the audio files and transcribe them
    for audio_file in audio_files:
        if audio_file.endswith(".mp3") or audio_file.endswith(".wav"):
            # Get the audio file path
            audio_file_path = os.path.join(audio_folder, audio_file)

            # Transcribe the audio file
            transcription = model.transcribe(audio_file_path)

            # Store the transcription in the dictionary
            transcriptions[audio_file] = transcription["text"]

    # Convert the dictionary to a json file
    json_file = "transcriptions_k_sample.json"
    with open(json_file, "w") as json_file:


        json.dump(transcriptions, json_file, indent=4)
    
    return json_file


if __name__ == "__main__":
    # Test the asr_pipeline function
    audio_folder = "p001/"
    json_file = asr_pipeline(audio_folder)
    print(f"Transcriptions saved to {json_file}")