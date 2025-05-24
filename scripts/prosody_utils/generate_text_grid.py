""" 
    generate_text_grid.py

    Overview:
        This script automates the generation of Praat TextGrid files for a set of audio samples using the aeneas forced alignment tool. 
        It reads a JSON file containing transcriptions for each audio file, iterates through a directory of .wav files, and for each file 
        with a corresponding transcription, invokes aeneas to align the audio with the text and outputs a TextGrid file. The script 
        ensures necessary directories exist, handles file paths robustly, and provides informative logging throughout the process.

    Author: Kweku Andoh Yamoah
    Date: 2024-09-30
"""
import json
import os
import subprocess

# Define directories with absolute paths
PROJECT_DIR = '/Users/kayems/Documents/GitHub/IUDvPP/'
AENEAS_DIR = os.path.join(PROJECT_DIR, 'aeneas')  # Assuming 'aeneas' is directly under project_dir
VOICE_SAMPLES_DIR = os.path.join(PROJECT_DIR, 'voice_sample_two_ways')
TEXT_GRID_DIR = os.path.join(PROJECT_DIR, 'prosody', 'text_grid_files_set2')
TRANSCRIPTIONS_JSON_PATH = os.path.join(PROJECT_DIR, 'asr', 'transcriptions_ambiguous_instructions.json')
TEMP_TEXT_FILE_PATH = os.path.join(PROJECT_DIR, 'temp_transcription.txt')

# Ensure the output directory exists
if not os.path.exists(TEXT_GRID_DIR):
    os.makedirs(TEXT_GRID_DIR)
    print(f"Created directory {TEXT_GRID_DIR}")

# Load transcriptions
if not os.path.exists(TRANSCRIPTIONS_JSON_PATH):
    raise FileNotFoundError(f"Transcriptions JSON file not found at: {TRANSCRIPTIONS_JSON_PATH}")
with open(TRANSCRIPTIONS_JSON_PATH, 'r', encoding='utf-8') as f:
    transcriptions = json.load(f)

# Function to call aeneas using subprocess
def call_aeneas(audio_file, transcription_text, output_file):
    """
    Call aeneas to generate TextGrid file.

    Args:
        audio_file (str): The path to the audio file.
        transcription_text (str): The transcription text.
        output_file (str): The path to save the generated TextGrid file.

    Raises:
        subprocess.CalledProcessError: If the command to run aeneas fails.

    Returns:
        None
    """
    # Write the transcription text to the temp file
    with open(TEMP_TEXT_FILE_PATH, 'w', encoding='utf-8') as temp_text_file:
        for word in transcription_text.split():
            temp_text_file.write(word + '\n')

    # Save the current working directory
    current_dir = os.getcwd()
    try:
        # Change to the aeneas directory
        os.chdir(AENEAS_DIR)

        # Define the command to run aeneas
        command = [
            'python3', '-m', 'aeneas.tools.execute_task',
            audio_file,
            TEMP_TEXT_FILE_PATH,
            'task_language=eng|is_text_type=plain|os_task_file_format=aud',
            output_file
        ]

        # Run the command
        subprocess.run(command, check=True)
        print(f"Generated TextGrid for {audio_file}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to generate TextGrid for {audio_file}: {e}")
    finally:
        # Return to the original directory
        os.chdir(current_dir)

# Iterate over voice samples recursively
for root, _, files in os.walk(VOICE_SAMPLES_DIR):
    for filename in sorted(files):
        if filename.endswith('.wav'):
            file_path = os.path.join(root, filename)
            # Generate relative path from VOICE_SAMPLES_DIR to match JSON keys
            relative_path = os.path.relpath(file_path, PROJECT_DIR)

            if relative_path in transcriptions:
                transcription = transcriptions[relative_path]
                output_file_path = os.path.join(TEXT_GRID_DIR, f"{os.path.splitext(filename)[0]}.TextGrid")
                print(f"Generating TextGrid for {file_path} to {output_file_path}")
                call_aeneas(file_path, transcription, output_file_path)
            else:
                print(f"No transcription found for {relative_path}")
        else:
            print(f"Ignoring non-wav file: {filename}")