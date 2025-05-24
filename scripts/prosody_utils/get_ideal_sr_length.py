
""" 
    get_ideal_sr_length.py

    Overview:
        This script provides utilities for analyzing audio datasets to determine the ideal fixed length (in samples)
        for raw audio features and the most common sampling rate across a collection of audio files. It processes
        audio segments corresponding to words, as defined by TextGrid files and gold label files, and computes
        the 90th, 95th, and 99th percentiles of segment lengths. The script also identifies the most frequent
        sampling rate used in the dataset. Audio slicing is performed using pydub, and feature extraction is
        handled by librosa.

    Author: Kweku Andoh Yamoah
    Date: 2024-10-14
"""
import os
import librosa
import numpy as np
import json
import string


def extract_raw_audio_features(audio_file, start_time, end_time, tmp_folder, target_sr=None):
    """
    Slice the audio file based on the word's start and end times using pydub, and load with librosa.
    """
    temp_word_file = slice_audio(audio_file, start_time, end_time, tmp_folder)
    y, sr = librosa.load(temp_word_file, sr=target_sr)
    return y, sr


def slice_audio(audio_file, start_time, end_time, tmp_folder):
    """
    Slice the audio file based on the word's start and end times using pydub.
    """
    from pydub import AudioSegment
    audio = AudioSegment.from_wav(audio_file)

    # adjusting start and end times to expand context window
    start_time = max(0, start_time - 0.010)  # 10ms before the start time
    end_time = float(end_time) + 0.025  # 25ms after the end time

    # Convert time from seconds to milliseconds for pydub
    start_ms = max(0, start_time * 1000)
    end_ms = end_time * 1000

    # Extract the slice of the audio corresponding to the word
    word_audio = audio[start_ms:end_ms]

    # Save to a temporary file
    temp_filename = os.path.join(tmp_folder, "temp_word.wav")
    word_audio.export(temp_filename, format="wav")

    return temp_filename


def compute_ideal_fixed_length(textgrid_dir, audio_dir, gold_labels_dir, tmp_folder):
    """
    Compute the ideal fixed length for raw audio features based on a given percentile (90th, 95th, 99th).
    Also compute the ideal sampling rate for all audio sets.
    """
    all_lengths = []
    all_sample_rates = []

    # Recursively walk through the audio_dir to find all .wav files
    for root, _, files in os.walk(audio_dir):
        for file in files:
            if file.endswith('.wav'):
                # Extract the file id and corresponding label and TextGrid
                file_id = file.replace('.wav', '')
                textgrid_file = os.path.join(textgrid_dir, f"{file_id}.TextGrid")
                audio_file = os.path.join(root, file)
                gold_labels_file = os.path.join(gold_labels_dir, f"{file_id}.txt")

                # Ensure we have matching TextGrid and gold labels file
                if os.path.exists(textgrid_file) and os.path.exists(gold_labels_file):
                    # Load the gold labels (word and corresponding label)
                    gold_labels = {}
                    with open(gold_labels_file, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) == 2:
                                word, label = parts
                                gold_labels[word.lower().strip(string.punctuation)] = label

                    # Extract word segments from TextGrid file
                    with open(textgrid_file, 'r') as tg:
                        for line in tg:
                            parts = line.strip().split()
                            if len(parts) >= 3:
                                start_time, end_time, word = float(parts[0]), float(parts[1]), parts[2]
                                sanitized_word = word.lower().strip(string.punctuation)

                                # Perform sanity check with gold labels
                                if sanitized_word in gold_labels:
                                    # Extract raw audio features for the word using librosa
                                    y, sr = extract_raw_audio_features(audio_file, start_time, end_time, tmp_folder, target_sr=22050)

                                    # Append length of the waveform
                                    all_lengths.append(len(y))
                                    all_sample_rates.append(sr)

    # Calculate percentiles for lengths
    lengths_np = np.array(all_lengths)
    percentile_90 = np.percentile(lengths_np, 90)
    percentile_95 = np.percentile(lengths_np, 95)
    percentile_99 = np.percentile(lengths_np, 99)

    # Determine the most common sampling rate
    unique_sample_rates, counts = np.unique(all_sample_rates, return_counts=True)
    ideal_sr = unique_sample_rates[np.argmax(counts)]

    print(f"90th percentile length: {percentile_90}")
    print(f"95th percentile length: {percentile_95}")
    print(f"99th percentile length: {percentile_99}")
    print(f"Ideal sampling rate: {ideal_sr}")

    return percentile_90, percentile_95, percentile_99, ideal_sr





if __name__ == "__main__":
    text_grid_dir = "../prosody/text_grid_files_set2"
    audio_dir = "../voice_sample_two_ways"
    gold_labels_dir = "../prosody/gold_label_txt/text_grid_files_set2"
    tmp_folder = "./tmp_audio_slices"

   # Ensure the temporary folder exists
    if not os.path.exists(tmp_folder):
        os.makedirs(tmp_folder)

    # Compute ideal fixed length and sampling rate
    percentile_90, percentile_95, percentile_99, ideal_sr = compute_ideal_fixed_length(
        textgrid_dir=text_grid_dir,
        audio_dir=audio_dir,
        gold_labels_dir=gold_labels_dir,
        tmp_folder=tmp_folder
    )