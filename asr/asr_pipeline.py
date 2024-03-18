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
from openai import OpenAI
from dotenv import load_dotenv
import argparse

# Load the environment variables
load_dotenv()

# Load the OpenAI API key
API_KEY = os.getenv("OPENAI_API_KEY")


def asr_pipeline_openai(audio_folder: str) -> str:
    """
    Transcribes audio files in a given folder using OpenAI's Whisper ASR model.

    Args:
        audio_folder (str): The path to the folder containing the audio files.

    Returns:
        str: The name of the JSON file containing the transcriptions.

    """

    #create client object
    client = OpenAI(api_key=API_KEY)

    #test for audio file example

    #Adapt for multiple files in a folder
    audio_files = os.listdir(audio_folder)
    transcriptions = {}

    for audio_file in audio_files:
        if audio_file.endswith(".m4a") or audio_file.endswith(".wav"):
            # Get the audio file path
            audio_file_path = os.path.join(audio_folder, audio_file)
            # Transcribe the audio file
            audio = open(audio_file_path, "rb")
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio,
                language= "en",
                response_format= "text",
            )

            # Store the transcription in the dictionary
            transcriptions[audio_file] = transcription
            
    # Convert the dictionary to a json file
    json_file = "transcriptions_openai_sample.json"
    with open(json_file, "w") as json_file:
        json.dump(transcriptions, json_file, indent=4)
    
    return json_file

def asr_pipeline_local(audio_folder: str) -> str:
    """
    This function takes in a folder containing
    audio files locally and transcribes them using the
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


def main():
    """
    Main function for the ASR Pipeline.

    This function parses command line arguments, determines the ASR mode, and calls the appropriate ASR pipeline function.

    Args:
        --audio_folder (str): Path to the folder containing audio files.
        --mode (str): ASR mode: 'openai' or 'local'.

    Returns:
        None
    """

    parser = argparse.ArgumentParser(description="ASR Pipeline")
    parser.add_argument("--audio_folder", type=str, help="Path to the folder containing audio files")
    parser.add_argument("--mode", type=str, help="ASR mode: 'openai' or 'local'")
    args = parser.parse_args()

    if args.mode == "openai":
        json_file = asr_pipeline_openai(args.audio_folder)
        print(f"Transcriptions from API endpoint: {json_file}")
    elif args.mode == "local":
        json_file = asr_pipeline_local(args.audio_folder)
        print(f"Transcriptions from local model: {json_file}")
    else:
        print("Invalid mode. Please choose 'openai' or 'local'.")

if __name__ == "__main__":
    main()
