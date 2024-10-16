""" 
@author: Kweku Yamoah
@date: 07-06-2024
@description: This file contains the function to compute the total time of all the audio files in the dataset
"""

import os
import wave
import contextlib

def compute_time(data_path):
    """
    This function computes the total time of all the audio files in the dataset
    :param data_path: The path to the dataset
    :return: The total time of all the audio files in the dataset
    """
    total_time = 0
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith('.wav'):
                with contextlib.closing(wave.open(os.path.join(root, file), 'r')) as f:
                    frames = f.getnframes()
                    rate = f.getframerate()
                    duration = frames / float(rate)
                    total_time += duration
    return total_time

if __name__ == '__main__':
    data_path = '../voice_sample_two_ways/'

    print("Total time for audio files is",compute_time(data_path), "seconds")
    print("Total time for audio files is",compute_time(data_path)/60, "minutes")
    print("Total time for audio files is",compute_time(data_path)/3600, "hours")