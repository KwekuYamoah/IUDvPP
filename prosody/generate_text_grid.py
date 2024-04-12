import json
import os
import subprocess

# Define directories with absolute paths
PROJECT_DIR = '/Users/kayems/Library/CloudStorage/OneDrive-AshesiUniversity/IUDvPP/'
AENEAS_DIR = os.path.join(PROJECT_DIR, 'aeneas')  # Assuming 'aeneas' is directly under project_dir
VOICE_SAMPLES_DIR = os.path.join(PROJECT_DIR, 'voice_samples')
TEXT_GRID_DIR = os.path.join(PROJECT_DIR, 'prosody', 'text_grid_files')
TRANSCRIPTIONS_JSON_PATH = os.path.join(PROJECT_DIR, 'asr', 'transcriptions.json')
TEMP_TEXT_FILE_PATH = os.path.join(PROJECT_DIR, 'temp_transcription.txt')

# Ensure the output directory exists
if not os.path.exists(TEXT_GRID_DIR):
    os.makedirs(TEXT_GRID_DIR)
    print(f"Created directory {TEXT_GRID_DIR}")

# Load transcriptions
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
            'task_language=eng|is_text_type=plain|os_task_file_format=textgrid',
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

# Iterate over voice samples
for filename in sorted(os.listdir(VOICE_SAMPLES_DIR)):
    if filename.endswith('.wav'):
        file_path = os.path.join(VOICE_SAMPLES_DIR, filename)
        if filename in transcriptions:
            transcription = transcriptions[filename]
            output_file_path = os.path.join(TEXT_GRID_DIR, f"{os.path.splitext(filename)[0]}.TextGrid")
            print(f"Generating TextGrid for {file_path} to {output_file_path}")
            call_aeneas(file_path, transcription, output_file_path)
        else:
            print(f"No transcription found for {filename}")
    else:
        print(f"Ignoring non-wav file: {filename}")
