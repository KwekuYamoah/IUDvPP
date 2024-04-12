import json
import os
import subprocess

# Define directories with absolute paths
project_dir = '/Users/kayems/Library/CloudStorage/OneDrive-AshesiUniversity/IUDvPP/'
aeneas_dir = os.path.join(project_dir, 'aeneas')  # Assuming 'aeneas' is directly under project_dir
voice_samples_dir = os.path.join(project_dir, 'voice_samples')
text_grid_dir = os.path.join(project_dir, 'prosody', 'text_grid_files')
transcriptions_json_path = os.path.join(project_dir, 'asr', 'transcriptions.json')
temp_text_file_path = os.path.join(project_dir, 'temp_transcription.txt')

# Ensure the output directory exists
if not os.path.exists(text_grid_dir):
    os.makedirs(text_grid_dir)
    print(f"Created directory {text_grid_dir}")

# Load transcriptions
with open(transcriptions_json_path, 'r') as f:
    transcriptions = json.load(f)

# Function to call aeneas using subprocess
def call_aeneas(audio_file, transcription_text, output_file):
    # Write the transcription text to the temp file
    with open(temp_text_file_path, 'w') as temp_text_file:
        for word in transcription_text.split():
            temp_text_file.write(word + '\n')

    # Save the current working directory
    current_dir = os.getcwd()
    try:
        # Change to the aeneas directory
        os.chdir(aeneas_dir)

        # Define the command to run aeneas
        command = [
            'python3', '-m', 'aeneas.tools.execute_task',
            audio_file,
            temp_text_file_path,
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
for filename in sorted(os.listdir(voice_samples_dir)):
    if filename.endswith('.wav'):
        file_path = os.path.join(voice_samples_dir, filename)
        if filename in transcriptions:
            transcription = transcriptions[filename]
            output_file_path = os.path.join(text_grid_dir, f"{os.path.splitext(filename)[0]}.TextGrid")
            print(f"Generating TextGrid for {file_path} to {output_file_path}")
            call_aeneas(file_path, transcription, output_file_path)
        else:
            print(f"No transcription found for {filename}")
    else:
        print(f"Ignoring non-wav file: {filename}")
