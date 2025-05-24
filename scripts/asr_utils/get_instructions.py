""" 
 Author: Kweku Andoh Yamoah
 Date: 2024-09-30
 Last Modified: 2024-09-30
 Version: 1.0

 This file creates a text file containing the instructions for the project, IUDvPP.
"""

import json

def get_instructions(filename):
    """
    Reads a JSON file containing a set of instructions and writes each instruction to a separate text file.

    Args:
        filename (str): The path to the JSON file containing the instruction set.

    Returns:
        None
    """

    with open(filename, "r", encoding="utf-8") as file:
        instruction_set = json.load(file)
    
    with open("instructions_alone.txt", "w", encoding="utf-8") as file:
        for index, instruction in enumerate(instruction_set):
            file.write(f"Instruction {index + 1}: {instruction['instruction']}\n\n")



if __name__ == "__main__":
    get_instructions("instructions_data.json")