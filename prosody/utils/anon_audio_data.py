import os
import shutil

def rename_files(participant_folder, project_name, participant_id, gender, output_folder=None):
    # Create output folder if not specified
    if output_folder is None:
        output_folder = f"{participant_folder}_renamed"
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over each file in the participant's folder
    files = sorted(os.listdir(participant_folder))
    if len(files) != 70:
        print(f"Warning: Expected 70 files, but found {len(files)} in {participant_folder}")
    
    for idx, filename in enumerate(files):
        # Determine the instruction number and interpretation number
        instruction_num = (idx // 2) + 1  # Instruction number (1 to 35)
        interpretation_num = (idx % 2) + 1  # Interpretation number (1 or 2)

        # Create the new filename
        new_filename = f"{project_name}_{gender}{participant_id:02d}_{instruction_num:03d}_i{interpretation_num}.wav"

        # Build source and destination paths
        src = os.path.join(participant_folder, filename)
        dst = os.path.join(output_folder, new_filename)

        # Copy the file to the new location with the new name
        shutil.copy2(src, dst)

    print(f"Renaming completed for participant folder '{participant_folder}'. Files saved in '{output_folder}'.")

if __name__ == "__main__":
    # Example usage
    participant_folder = "/Users/kayems/Documents/GitHub/IUDvPP/prosody/voice_sample_two_ways/pid_f05"  # Folder containing original audio files for the participant
    project_name = "pid"  # Project name
    participant_id = 5  # Fourth participant
    gender = "f"  # Female participant

    rename_files(participant_folder, project_name, participant_id, gender)