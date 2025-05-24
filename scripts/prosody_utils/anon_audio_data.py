"""Renames and copies audio files from a participant's folder to a new output folder with 
    a standardized naming convention.
    Author: Kweku Andoh Yamoah
    Date: 2024-10-13
"""

import os
import shutil

def rename_files(participant_folder, project_name, participant_id, gender, output_folder=None):
    """
    Renames and copies audio files from a participant's folder to a new output folder with a standardized naming convention.
    Args:
        participant_folder (str): Path to the folder containing the participant's original audio files.
        project_name (str): Name of the project to include in the new filenames.
        participant_id (int): Numeric identifier for the participant.
        gender (str): Gender code or label to include in the new filenames.
        output_folder (str, optional): Path to the output folder where renamed files will be saved. 
            If not specified, a new folder with '_renamed' appended to the participant_folder name will be created.
    Notes:
        - Expects exactly 70 files in the participant_folder, named in a way that the second underscore-separated part is a number (e.g., 'audio_1.m4a').
        - Files are renamed using the format: '{project_name}_{gender}{participant_id:02d}_{instruction_num:03d}_i{interpretation_num}.m4a'.
        - instruction_num ranges from 1 to 35, and interpretation_num is either 1 or 2 for each instruction.
        - Files are copied (not moved) to the output folder.
        - Prints a warning if the number of files is not 70.
        - Prints progress for each file processed.
    """
    # Create output folder if not specified
    if output_folder is None:
        output_folder = f"{participant_folder}_renamed"
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over each file in the participant's folder
    files = sorted(os.listdir(participant_folder), key=lambda x: int(x.split('_')[1].split('.')[0]))
    if len(files) != 70:
        print(f"Warning: Expected 70 files, but found {len(files)} in {participant_folder}")
    
    for idx, filename in enumerate(files):
        print(f"Processing file, {idx}...{filename}...")
        # Determine the instruction number and interpretation number based on file index
        instruction_num = ((idx) // 2) + 1  # Instruction number (1 to 35)
        interpretation_num = ((idx) % 2) + 1  # Interpretation number (1 or 2)

        # Create the new filename
        new_filename = f"{project_name}_{gender}{participant_id:02d}_{instruction_num:03d}_i{interpretation_num}.m4a"

        # Build source and destination paths
        # Build source and destination paths
        src = os.path.join(participant_folder, filename)
        dst = os.path.join(output_folder, new_filename)
        shutil.copy2(src, dst)

        
    print(f"Renaming completed for participant folder '{participant_folder}'. Files saved in '{output_folder}'.")



if __name__ == "__main__":
    # Example usage
    participant_folder = "pre_process_unamed/r_samples"  # Folder containing original audio files for the participant
    project_name = "pid"  # Project name
    participant_id = 12  # participant number
    gender = "f"  # Female participant

    rename_files(participant_folder, project_name, participant_id, gender, output_folder=f'{project_name}_{gender}0{participant_id}')