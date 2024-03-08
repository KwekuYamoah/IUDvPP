#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AUTHOR

    Sébastien Le Maguer <lemagues@tcd.ie>

DESCRIPTION

LICENSE
    This script is in the public domain, free from copyrights or restrictions.
    Created: 27 January 2020
"""

# System/default

import sys
import os
import glob
import math

# Arguments
import argparse

# Messaging/logging
import traceback
import time
import logging
import copy
from torch.nn.functional import threshold

# Configuration
import yaml
from collections import defaultdict

# Math and plotting
import numpy as np
import scipy.ndimage
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import seaborn as sns

# Parallel job managment
from joblib import Parallel, delayed

# Dataframes for data storage
import pandas as pd


# acoustic features
from Prosody_tools import energy_processing
from Prosody_tools import f0_processing
from Prosody_tools import duration_processing
from Prosody_tools import misc
from Prosody_tools import smooth_and_interp
from Prosody_tools import cwt_utils, loma, lab


LEVEL = [logging.WARNING, logging.INFO, logging.DEBUG]

#dictionary to keep track of the prominence labels of the words in a given dataset corpus
PROMINENCE_GOLD_LABEL_DICTIONARY = {}





def get_logger(verbosity, log_file):

    # create logger and formatter
    logger = logging.getLogger("prosody labeller")
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Verbose level => logging level
    log_level = verbosity
    if (log_level >= len(LEVEL)):
        log_level = len(LEVEL) - 1
        logger.setLevel(log_level)
        logging.warning("verbosity level is too high, I'm gonna assume you're taking the highest (%d)" % log_level)
    else:
        logger.setLevel(LEVEL[log_level])

    # create console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # create file handler
    if log_file is not None:
        fh = logging.FileHandler(log_file)
        logger.addHandler(fh)

    return logger


def apply_configuration(current_configuration, updating_part):
    """Utils to update the current configuration using the updating part

    Parameters
    ----------
    current_configuration: dict
        The current state of the configuration

    updating_part: dict
        The information to add to the current configuration

    Returns
    -------
    dict
       the updated configuration
    """
    if not isinstance(current_configuration, dict):
        return updating_part

    if current_configuration is None:
        return updating_part

    if updating_part is None:
        return current_configuration

    for k in updating_part:
        if k not in current_configuration:
            current_configuration[k] = updating_part[k]
        else:
            current_configuration[k] = apply_configuration(current_configuration[k], updating_part[k])



    return current_configuration


'''
The extract_word_information function splits a given array of signal values extracted from an audio file into sub-arrays where each sub-array represents the array of values associated with a particular 
word in that audio file.

Params:
    signal_info(list[float]): This is a list where each value within that list represents an amplitude value within an audio signal at a particular time. 
    filename (string): This is a string that represents the name of the input file.
    language (string): This is a string that represents the language in which the audio files are spoken in.
    word_info(list[list]): This is a list containing sub-lists where each sublist contains information about a word within an audio signal and the time values within which the word is spoken in the audio.

Returns:
    word_signal_info (list[list]): This is a list containing sub-lists where each sublist contains information about a word, its corresponding time values as well as its corresponding signal values.
'''
def extract_word_information(signal_info, filename, language, word_info):
    #extract the filename
    filename = filename.split('/')[2].split('.')[0]

    #initialize word_signal_info
    word_signal_info = []

    #extract the dictionary containing gold label information on the words in word_info
    gold_labels = PROMINENCE_GOLD_LABEL_DICTIONARY[language][filename]


    #obtain the keys of the gold label dictionary for a specified file to ensure that we only include words for which we have their prosodic annotations
    gold_label_words = list(gold_labels.keys())
   


    #iterate through the words and slice the audio signal to capture the signal values corresponding to the each word
    for word in word_info:

        #check to see if we have the gold label for that specific word
        if word[2] in gold_label_words:
            #slice the signal
            word_signal = signal_info[int(word[0]):int(word[1])]

            #extract the prosodic gold label of the word
            prosodic_gold_label = gold_labels[word[2]]
            
            #updated word list
            word_list = [word[0], word[1], word[2], word_signal, prosodic_gold_label]

            #place the word signal information into word_signal_info
            word_signal_info.append(word_list)


    return word_signal_info

'''
The generate_gold_labels_dictionary function takes as parameters a language tag and the path to the gold labels for the words in the audio files and 
populates a dictionary with that information.

Params:
    language_tag (string): This is a string that represents a particular language.
    gold_label_path (string): This is a path to a textfile containing the gold labels for the words in the audio files.

Returns:
    None
'''
def generate_gold_labels_dictionary(language_tag, gold_label_path):
    #open the file represented by the gold_label_path
    gold_label_file = open(gold_label_path, 'r').readlines()
    gold_label_file_words = [x.strip('\n') for x in gold_label_file]

   

    #initialize a variable to keep track of the current file during iteration through the gold label textfile
    current_filename = ''

    #initialize a variable to keep represent a dictionary that holds the words of the gold label textfile and their corresponding labels
    gold_label_dictionary = {}

    #iterate through each line in the file
    for word in gold_label_file_words:
        #check to see if the word represents a filename or it is a normal word
        if '.txt' in word:
            current_filename = word.split('.')[0]

            #initialize the value of the key with the name representing the file
            gold_label_dictionary[current_filename] = {}
        
        #if the word does not represent a filename, place the word in the gold_label_dictionary as a key and place its gold label as a value associated with that key
        else:
            word_key = word.split(',')[0].strip(' ').lower()
            gold_label = int(word.split(',')[1])
            gold_label_dictionary[current_filename][word_key] = gold_label
    
    #place the populated dictionary into the global dictionary for keeping track of the words in the input audio files and their corresponding gold labels
    PROMINENCE_GOLD_LABEL_DICTIONARY[language_tag] = gold_label_dictionary



    return



def analysis(input_file, language, cfg, logger, annotation_dir=None, output_dir=None, plot=False):


    # Load the wave file
    #sig is refering to the extracted original signal and orig_sr is refering to the original sample rate.
    orig_sr, sig = misc.read_wav(input_file)

    np.set_printoptions(threshold=np.inf)

    sig = sig.tolist()
    

    # extract energy
    energy = energy_processing.extract_energy(sig, orig_sr,
                                              cfg["energy"]["band_min"],
                                              cfg["energy"]["band_max"],
                                              cfg["energy"]["calculation_method"])
    energy = np.cbrt(energy+1)
    if cfg["energy"]["smooth_energy"]:
        energy = smooth_and_interp.peak_smooth(energy, 30, 3)  # FIXME: 30? 3?
        energy = smooth_and_interp.smooth(energy, 10)

  

    # extract f0
    raw_pitch = f0_processing.extract_f0(sig, orig_sr,
                                         f0_min=cfg["f0"]["min_f0"],
                                         f0_max=cfg["f0"]["max_f0"],
                                         voicing=cfg["f0"]["voicing_threshold"],
                                         configuration=cfg["f0"]["pitch_tracker"])

   
    # interpolate, stylize
    pitch = f0_processing.process(raw_pitch)

    # extract speech rate
    rate = np.zeros(len(pitch))


    # Get annotations (if available)
    tiers = []
    if annotation_dir is None:
        annotation_dir = os.path.dirname(input_file)
    basename = os.path.splitext(os.path.basename(input_file))[0]
    grid =  os.path.join(annotation_dir, "%s.TextGrid" % basename)
    
    
    if os.path.exists(grid):
        tiers = lab.read_textgrid(grid)
    else:
        grid =  os.path.join(annotation_dir, "%s.lab" % basename)
        if not os.path.exists(grid):
            raise Exception("There is no annotations associated with %s" % input_file)
        tiers = lab.read_htk_label(grid)

   

    #experiment to obtain prosody information
    words_info = tiers['words']
    experiment_tiers_information = words_info
    

    # Extract duration
    if len(tiers) > 0:
        dur_tiers = []
        for level in cfg["duration"]["duration_tiers"]:
            assert(level.lower() in tiers), level+" not defined in tiers: check that duration_tiers in config match the actual textgrid tiers"
            try:
                dur_tiers.append(tiers[level.lower()])
                #experiment_tiers_information.append(tiers[level.lower()])
            except:
                print("\nerror: "+"\""+level+"\"" +" not in labels, modify duration_tiers in config\n\n")
                raise
    

    if not cfg["duration"]["acoustic_estimation"]:
        rate = duration_processing.get_duration_signal(dur_tiers,
                                                       weights=cfg["duration"]["weights"],
                                                       linear=cfg["duration"]["linear"],
                                                       sil_symbols=cfg["duration"]["silence_symbols"],
                                                       bump = cfg["duration"]["bump"])

    else:
        rate = duration_processing.get_rate(energy)
        rate = smooth_and_interp.smooth(rate, 30)

    if cfg["duration"]["delta_duration"]:
            rate = np.diff(rate)

    # Combine signals
    min_length = np.min([len(pitch), len(energy), len(rate)])
 
  
    pitch = pitch[:min_length]
 
    energy = energy[:min_length]
 
    rate = rate[:min_length]


    '''
    Figuring out if the combination method is a summation or a product of the signals.
    '''


    if cfg["feature_combination"]["type"] == "product":
        pitch = misc.normalize_minmax(pitch) ** cfg["feature_combination"]["weights"]["f0"]
        energy = misc.normalize_minmax(energy) ** cfg["feature_combination"]["weights"]["energy"]
        rate =  misc.normalize_minmax(rate) ** cfg["feature_combination"]["weights"]["duration"]
        params = pitch * energy * rate
        pitch_energy = pitch * energy
        pitch_rate = pitch * rate 
        energy_rate = energy * rate

    else:
        params = misc.normalize_std(pitch) * cfg["feature_combination"]["weights"]["f0"] + \
                 misc.normalize_std(energy) * cfg["feature_combination"]["weights"]["energy"] + \
                 misc.normalize_std(rate) * cfg["feature_combination"]["weights"]["duration"]

        pitch_energy = misc.normalize_std(pitch) * cfg["feature_combination"]["weights"]["f0"] + \
                       misc.normalize_std(energy) * cfg["feature_combination"]["weights"]["energy"]

        pitch_rate = misc.normalize_std(pitch) * cfg["feature_combination"]["weights"]["f0"] + \
                     misc.normalize_std(rate) * cfg["feature_combination"]["weights"]["energy"]

        energy_rate = misc.normalize_std(energy) * cfg["feature_combination"]["weights"]["f0"] + \
                     misc.normalize_std(rate) * cfg["feature_combination"]["weights"]["energy"]

    if cfg["feature_combination"]["detrend"]:
         params = smooth_and_interp.remove_bias(params, 800)

         pitch_energy = smooth_and_interp.remove_bias(pitch_energy, 800)

         pitch_rate = smooth_and_interp.remove_bias(pitch_rate, 800)

         energy_rate = smooth_and_interp.remove_bias(energy_rate, 800)

    
    params = [round(x,2) for x in misc.normalize_std(params).tolist()]
    pitch_energy = [round(x,2) for x in misc.normalize_std(pitch_energy).tolist()]
    pitch_rate = [round(x,2) for x in misc.normalize_std(pitch_rate).tolist()]
    energy_rate = [round(x,2) for x in misc.normalize_std(energy_rate).tolist()]



    #split the information of various signals into the different words within the audio signal
    word_raw_data = extract_word_information(sig,input_file, language, experiment_tiers_information)
    word_f0 = extract_word_information(pitch.tolist(), input_file, language, experiment_tiers_information)
    word_f0 = extract_word_information(pitch.tolist(), input_file, language, experiment_tiers_information)
    word_en = extract_word_information(energy.tolist(), input_file, language, experiment_tiers_information)
    word_dur = extract_word_information(rate.tolist(), input_file, language, experiment_tiers_information)
    word_f0_energy = extract_word_information(pitch_energy, input_file, language, experiment_tiers_information)
    word_f0_rate = extract_word_information(pitch_rate, input_file, language, experiment_tiers_information)
    word_energy_rate = extract_word_information(energy_rate, input_file, language, experiment_tiers_information)
    word_f0_energy_rate = extract_word_information(params, input_file, language, experiment_tiers_information)

    pitch = [round(x,2) for x in pitch.tolist()]
    energy = [round(x,2) for x in energy.tolist()]
    rate = [round(x,2) for x in rate.tolist()]


    #return all of the extracted features as a dictionary represented as a string
    extracted_features =  input_file + '=' + '{"raw_audio_data":' + str(sig)+',"f0":'+str(pitch)+',"en":'+ str(energy)+',"dur":'+str(rate)+',"f0_energy":'+str(pitch_energy)+',"f0_rate":'+str(pitch_rate)+',"en_rate":'+str(energy_rate)+',"f0_en_rate":'+str(params)+',"word_raw_data":'+str(word_raw_data)+',"word_f0":'+str(word_f0)+',"word_en":'+str(word_en)+',"word_dur":'+str(word_dur)+',"word_f0_energy":'+str(word_f0_energy)+',"word_f0_rate":'+str(word_f0_rate)+',"word_energy_rate":'+str(word_energy_rate)+',"word_f0_energy_rate":'+str(word_f0_energy_rate)+'}'

    return extracted_features


    
   



def analysis_batch_wrap(input_file, cfg, annotation_dir=None, output_dir=None, plot=0, logger=None):
    # Encapsulate running
    try:
        print(input_file)
        analysis(input_file, cfg, logger, annotation_dir, output_dir, plot)
    except Exception as ex:
        logging.error(str(ex))
        traceback.print_exc(file=sys.stderr)




#to run the main function in the terminal, be sure to provide the folder containing the .wav files,
# the .TextGrid files and the .prom files as a parameter in the command line terminal.
def main():
    """Main entry function
    """
    global args, logger

    # Load configuration
    #defaultdict() creates a dictionary-like object to provide support for elements that might
    #not even yet exist in the dictionary
    configuration = defaultdict()
    with open(os.path.dirname(os.path.realpath(__file__)) + "/configs/default.yaml", 'r') as f:
        configuration = apply_configuration(configuration, defaultdict(lambda: False, yaml.load(f, Loader=yaml.FullLoader)))

    if args.config:
        try:
            with open(args.config, 'r') as f:
                configuration = apply_configuration(configuration, defaultdict(lambda: False, yaml.load(f, Loader=yaml.FullLoader)))
        except IOError as ex:
            print("configuration file " + args.config + " could not be loaded:")

            sys.exit(1)
    logger.debug("Current configuration:")
    logger.debug(configuration)

    # Get the number of jobs
    nb_jobs = args.nb_jobs

    # Loading files
    if os.path.isfile(args.input):
        input_files = [args.input]
    else:
        #list all of the directories in args.input
        language_files = os.listdir(args.input)
        #for every directory listed, take the .wav files within that directory and place it in a list
        input_files = []
        for language_file in language_files:
            language_file_path = args.input + language_file
            wav_files = glob.glob(language_file_path + "/*.wav")
            wav_files.append(language_file)
            input_files.append(wav_files)

    
    if len(input_files) == 1:
        nb_jobs = 1

    plot_flag = 0

  
   
    #obtain the audio data features for each audio file and write it into the provided output file
    for language in input_files:
        if '.DS_Store' not in language:
            #obtain the name of the language being dealt with
            focus_language = language[len(language)-1]

            #construct a path to the prominence gold labels for the words in the audio files for the focus language
            prominence_gold_label_path = args.prominence_gold_labels + focus_language + '/gold_labels.txt'

           

            #call the generate_gold_labels_dictionary function to generate a dictionary to store the gold labels of the words in the audio files for the specified language
            generate_gold_labels_dictionary(focus_language, prominence_gold_label_path)

            

            #construct the path to which the extracted features for a particular language should be written to
            output_file_path = args.extracted_features + focus_language + '_extracted_features.txt'

            #open the path to which the extracted features should be written to
            write_output_file = open(output_file_path, 'a')

            for f in language[:len(language)-1]:
                #first, obtain the filename from the filepath to ensure that we have gold labels for that file
                filename = f.split('/')[2].split('.')[0]

                keys_gold_label_dict = list(PROMINENCE_GOLD_LABEL_DICTIONARY[focus_language].keys())
    

                if filename in keys_gold_label_dict:
                    print(f)
                    extracted_features = analysis(f, focus_language, configuration, logger, args.annotation_directory, args.output_directory, plot_flag)

                    #write the extracted features into a text file
                    write_output_file.write(extracted_features+'\n')
            
            #close the file
            write_output_file.close()


    




if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(description="Command line application to analyze prosody using wavelets.")

        # Add options
        parser.add_argument("-a", "--annotation_directory", default=None, type=str,
                            help="Annotation directory. If not specified, the tool will by default try to load annotations from the directory containing the wav files")
        parser.add_argument("-j", "--nb_jobs", default=4, type=int,
                            help="Define the number of jobs to run in parallel")
        parser.add_argument("-c", "--config", default=None, type=str,
                            help="configuration file")
        parser.add_argument("-l", "--log_file", default=None, type=str,
                            help="Logger file")
        parser.add_argument("-o", "--output_directory", default=None, type=str,
                            help="The output directory. If not specified, the tool will output the result in a .prom file in the same directory than the wave files")
        parser.add_argument("-p", "--plot", default=False, action="store_true",
                            help="Plot the result (the number of jobs is de facto set to 1 if activated)")
        parser.add_argument("-v", "--verbosity", action="count", default=1,
                            help="increase output verbosity")
        #adding can argument to represent the directory where the audio language data is stored
        parser.add_argument("-e", "--extracted_features", default='/Users/dr_david_sasu/Desktop/Research Code/extracted_audio_features/', type=str,
                            help="Provide the directory where the extracted features of the audio files would be stored.")

        # Add argument for the directory containing the .wav files
        parser.add_argument("input", help="directory with wave files or wave file to analyze (a label file with the same basename should be available)")

        # Add argument for the directory containg the prominence gold labels for the words in each input audio file
        parser.add_argument("-g", "--prominence_gold_labels", default='/Users/dr_david_sasu/Desktop/Research Code/sample prominence gold labels/', type=str, help="directory where the prominence gold labels for the words in each input audio file.")




        # Parsing arguments
        args = parser.parse_args()
        if args.plot:
            args.nb_jobs = 1
        # Get the logger
        logger = get_logger(args.verbosity, args.log_file)

        # Debug time
        start_time = time.time()
        logger.info("start time = " + time.asctime())

        # Running main function <=> run application
        main()

        
        # Debug time
        logger.info("end time = " + time.asctime())
        logger.info('TOTAL TIME IN MINUTES: %02.2f' %
                     ((time.time() - start_time) / 60.0))

        # Exit program
        sys.exit(0)
    except KeyboardInterrupt as e:  # Ctrl-C
        raise e
    except SystemExit:  # sys.exit()
        pass
    except Exception as e:
        logging.error('ERROR, UNEXPECTED EXCEPTION')
        logging.error(str(e))
        traceback.print_exc(file=sys.stderr)
        sys.exit(-1)
