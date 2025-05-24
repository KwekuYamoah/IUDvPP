"""
    audio_features_extractor.py
    Summary:
        This module provides utilities for extracting word-level audio features from speech recordings,
        using corresponding TextGrid and gold label files. It includes functions for slicing audio segments
        with context windows, extracting fixed-length raw audio features, calculating segment lengths,
        and processing batches of files to generate datasets suitable for machine learning tasks.
    Functions:
        - slice_audio: Slices a WAV audio file based on start and end times, adding context windows.
        - extract_raw_audio_features: Extracts fixed-length raw audio features from a specified audio segment.
        - get_audio_segment_length: Calculates the number of samples in a context-expanded audio segment.
        - process_files: Processes directories of TextGrid, audio, and gold label files to extract features and labels,
            saving results in a JSON file.
    Author: Kweku Andoh Yamoah
    Date Written: 2024-10-18

"""

import os
import json
import string
import numpy as np
import matplotlib.pyplot as plt
import librosa
from pydub import AudioSegment

def slice_audio(audio_file, start_time, end_time, tmp_folder):
    """
    Slices an audio file based on start and end times with additional context window.
    This function takes a WAV audio file and extracts a segment based on specified time boundaries.
    It adds a 10ms context window before the start time and 25ms after the end time to ensure
    complete phoneme capture.
    Args:
        audio_file (str): Path to the input WAV audio file.
        start_time (float): Starting time in seconds for the slice.
        end_time (float): Ending time in seconds for the slice.
        tmp_folder (str): Directory path where the temporary sliced audio file will be saved.
    Returns:
        str: Path to the temporary file containing the sliced audio segment.
    Note:
        - The function automatically adjusts start time to prevent negative values
        - Times are converted from seconds to milliseconds for pydub processing
        - Output is saved as WAV format
    """
    
    audio = AudioSegment.from_wav(audio_file)

    # adjusting start and end times to expand context window
    start_time = max(0, start_time - 0.010) # 10ms before the start time
    end_time = float(end_time) + 0.025 # 25ms after the end time 
    
    # Convert time from seconds to milliseconds for pydub
    start_ms = max(0, start_time * 1000)
    end_ms = end_time * 1000
    
    # Extract the slice of the audio corresponding to the word
    word_audio = audio[start_ms:end_ms]
    
    # Save to a temporary file
    temp_filename = os.path.join(tmp_folder, "temp_word.wav")
    word_audio.export(temp_filename, format="wav")
    
    return temp_filename


def extract_raw_audio_features(audio_file, start_time, end_time, tmp_folder, fixed_length=28114, target_sr=22050):
    """
    Extract raw audio features from an audio file for a specific time segment.
    This function slices an audio file based on start and end times, then processes
    the audio to ensure a fixed length output by either truncating or padding the signal.
    Parameters:
    ----------
    audio_file : str
        Path to the input audio file
    start_time : float
        Start time of the segment in milliseconds
    end_time : float
        End time of the segment in milliseconds
    tmp_folder : str
        Path to temporary folder for storing intermediate files
    fixed_length : int, optional
        Target length of the output signal in samples (default: 28114)
    target_sr : int, optional
        Target sampling rate in Hz (default: 22050)
    Returns:
    -------
    numpy.ndarray
        Raw audio waveform of fixed length (time-domain signal)
    Notes:
    -----
    The function uses pydub for initial audio slicing and librosa for subsequent processing.
    The output is always of length fixed_length, achieved through truncation or zero-padding.
    """
    
    # Slice the word's segment using pydub
    temp_word_file = slice_audio(audio_file, start_time, end_time, tmp_folder)
    
    # Load the sliced audio using librosa
    y, sr = librosa.load(temp_word_file, sr=target_sr)

    # Truncate or pad the audio signal to fixed_length
    if len(y) > fixed_length:
        y = y[:fixed_length]
    elif len(y) < fixed_length:
        y = np.pad(y, (0, fixed_length - len(y)), 'constant')
    
    # Return the raw waveform as the features
    return y  # The raw audio data (time-domain waveform)

def get_audio_segment_length(audio_file, start_time, end_time):
    """
    Calculates the number of audio samples in a specified segment of an audio file, 
    optionally expanding the segment by 10ms before the start time and 35ms after the end time 
    to provide additional context.
    Args:
        audio_file (str): Path to the audio file.
        start_time (float): Start time of the segment in seconds.
        end_time (float): End time of the segment in seconds.
    Returns:
        int: Number of audio samples in the specified (and context-expanded) segment.
    """
    
    # adjusting start and end times to expand context window
    start_time = max(0, start_time - 0.010) # 10ms before the start time
    end_time = float(end_time) + 0.035 # 35ms after the end time 

    duration = float(end_time) - float(start_time)
    y, sr = librosa.load(audio_file, sr=22050, offset=float(start_time), duration=duration)
    return len(y)


def process_files(textgrid_dir, audio_dir, gold_labels_dir, output_file, tmp_folder):
    """
    Processes audio, TextGrid, and gold label files to extract word-level audio features and labels, 
    and saves the results in a JSON file.
    This function recursively searches for gold label text files, corresponding TextGrid files, 
    and audio files based on file IDs. For each word in the TextGrid that matches a gold label, 
    it extracts raw audio features from the corresponding audio segment and associates it with the label.
    The results are saved in a JSON file, with each entry containing the words, labels, and extracted features.
    Args:
        textgrid_dir (str): Path to the directory containing TextGrid files (can be nested).
        audio_dir (str): Path to the directory containing audio (.wav) files (can be nested).
        gold_labels_dir (str): Path to the directory containing gold label text files (can be nested).
        output_file (str): Path to the output JSON file where results will be saved.
        tmp_folder (str): Path to a temporary folder used during audio feature extraction.
    Returns:
        None
    Notes:
        - Assumes gold label files are .txt files with lines formatted as "<word> <label>".
        - Assumes TextGrid files are plain text with lines containing "<start_time> <end_time> <word>".
        - Assumes audio files are in .wav format.
        - Prints warnings if corresponding files or labels are missing.
        - Requires the function `extract_raw_audio_features` to be defined elsewhere.
    """
    results = {}
    all_lengths = []

    # Collect all gold label text files recursively
    gold_label_files = []
    for root, _, files in os.walk(gold_labels_dir):
        for file in files:
            if file.endswith('.txt'):
                gold_label_files.append(os.path.join(root, file))

    # Loop through the collected gold label text files
    for gold_labels_file in gold_label_files:
        # Extract filename (without extension) and gold labels
        label_file = os.path.basename(gold_labels_file)
        file_id = label_file.replace('.txt', '')  # The ID without file extension
        
        # Load the gold labels (word and corresponding label)
        gold_labels = {}
        with open(gold_labels_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    word, label = parts
                    gold_labels[word.lower().strip(string.punctuation)] = label  # Sanitize word
        
        # Recursively find the corresponding TextGrid file
        textgrid_file = None
        for root, _, files in os.walk(textgrid_dir):
            if f"{file_id}.TextGrid" in files:
                textgrid_file = os.path.join(root, f"{file_id}.TextGrid")
                break
        
        # Recursively find the corresponding audio file
        audio_file = None
        for root, _, files in os.walk(audio_dir):
            if f"{file_id}.wav" in files:
                audio_file = os.path.join(root, f"{file_id}.wav")
                break
        
        # Check if both the TextGrid and audio files were found
        if textgrid_file and os.path.exists(textgrid_file) and audio_file and os.path.exists(audio_file):
            # Initialize the data for this file
            file_data = {
                "words": [],
                "labels": [],
                "features": []
            }
            
            with open(textgrid_file, 'r') as tg:
                for line in tg:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        start_time, end_time, word = float(parts[0]), float(parts[1]), parts[2]
                        sanitized_word = word.lower().strip(string.punctuation)  # Remove punctuation, lowercase
                        
                        # Perform sanity check with gold labels
                        if sanitized_word in gold_labels:
                            label = gold_labels[sanitized_word]
                            
                            # Extract raw audio features for the word using librosa
                            raw_features = extract_raw_audio_features(audio_file, start_time, end_time, tmp_folder)
                            
                            # Append data for this word
                            file_data["words"].append(sanitized_word)
                            file_data["labels"].append(int(label))  # Assuming labels are integers
                            file_data["features"].append(raw_features.tolist())  # Convert numpy array to list for JSON serialization
                        else:
                            print(f"Warning: Word '{sanitized_word}' not found in gold labels. With file ID: {file_id}")
            if file_data["words"]:
                results[file_id] = file_data  # Use file ID as the key
            else:
                print(f"Warning: No valid words found in TextGrid for file ID: {file_id}")
        else:
            print(f"Warning: Corresponding TextGrid or audio file not found for file ID: {file_id}")

    # Write the final JSON output
    with open(output_file, 'w') as json_file:
        json.dump(results, json_file, indent=4)



if __name__ == "__main__":
    text_grid_dir = "../prosody/text_grid_files_set2"
    audio_dir = "../voice_sample_two_ways"
    gold_labels_dir = "../prosody/gold_label_txt/text_grid_files_set2/eval"
    output_file = "../prosody/data/ambiguous_raw_extracted_audio_ml_features_eval.json"
    tmp_folder = "./tmp_audio_slices"

    # Ensure the temporary folder exists
    if not os.path.exists(tmp_folder):
        os.makedirs(tmp_folder)

    process_files(textgrid_dir=text_grid_dir,
                  audio_dir=audio_dir,
                  gold_labels_dir=gold_labels_dir,
                  output_file=output_file,
                  tmp_folder=tmp_folder)