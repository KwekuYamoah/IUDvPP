import os
import json
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# set openai client with api key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# function to read json file


def load_json(file_path='../prosody/data/ambiguous_prosody_multi_label_features_eval.json'):
    with open(file_path) as f:
        data = json.load(f)
    return data


def get_embedding(text, model="text-embedding-3-large"):
    """
    Generates an embedding for the given text using the specified model.

    Args:
        text (str): The input text to generate the embedding for.
        model (str, optional): The model to use for generating the embedding. Defaults to "text-embedding-3-small".

    Returns:
        list: A list representing the embedding of the input text.
    """
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding[:300]


def normalize_l2(x):
    """
    Normalize the input array using L2 norm.

    Parameters:
    x (array-like): Input array to be normalized. Can be a 1D or 2D array.

    Returns:
    array-like: L2 normalized array. If the input is a 1D array, returns a 1D array.
                If the input is a 2D array, returns a 2D array with each row normalized.
                If the norm of a vector is 0, the original vector is returned.
    """
    x = np.array(x)
    if x.ndim == 1:
        norm = np.linalg.norm(x)
        if norm == 0:
            return x
        return x / norm
    else:
        norm = np.linalg.norm(x, 2, axis=1, keepdims=True)
        return np.where(norm == 0, x, x / norm)

# define a function to generate embeddings taking a loaded json file as input


def generate_embeddings(data):
    """
    Generate embeddings for each word in the provided data.

    Args:
        data (dict): A dictionary where each key is an identifier and each value is another dictionary
                     containing the following keys:
                     - "filepath" (str): The file path associated with the entry.
                     - "words" (list): A list of words for which embeddings need to be generated.
                     - "positions" (list): A list of positions corresponding to the words.
                     - "labels" (list): A list of labels corresponding to the words.
                     - "features" (list): A list of features corresponding to the words.

    Returns:
        dict: A new dictionary with the same structure as the input data, but with an additional key
              "word_embeddings" for each entry. The "word_embeddings" key contains a list of normalized embeddings
              corresponding to each word in the "words" list.
    """
    # create new entry dict
    new_data = {}
    # iterate over the data
    for key, value in data.items():
        new_entry = {
            "filepath": value["filepath"],
            "words": value["words"],
            "positions": value["positions"],
            "labels": value["labels"],
            "features": value["features"],
        }
        # now we create a new field called "embeddings"
        embeddings = []
        # for each word in the value["words"] list
        for word in value["words"]:
            # get the embedding of the word
            embedding = get_embedding(word)
            # normalize the embedding
            normalized_embedding = normalize_l2(embedding)
            # convert the normalized embedding to a list before appending
            embeddings.append(normalized_embedding.tolist())  # Convert ndarray to list

        # add the embeddings list to the new entry
        new_entry["word_embeddings"] = embeddings
        # add the new entry to the new data
        new_data[key] = new_entry
        # comunicate the progress
        print(f"Embeddings generated for {key}")
    # communicate the end of the process
    print("All embeddings generated.")
    return new_data


def write_json(data, file_path='../prosody/data/ambiguous_prosody_multi_label_features_eval_embeddings.json'):
    """
    Write the given data to a JSON file.

    Args:
        data (dict): The data to be written to the JSON file.
        file_path (str, optional): The path to the JSON file where the data will be written. 
                                   Defaults to '../prosody/data/ambiguous_prosody_multi_label_features_train_embeddings.json'.

    Returns:
        None
    """
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"New JSON file saved as {file_path}")


if __name__ == "__main__":

    # load the json file
    data = load_json()

    # generate embeddings
    new_data = generate_embeddings(data)

    # write the new data to a file
    write_json(new_data)

    print("New JSON file saved as ambiguous_prosody_multi_label_features_eval_embeddings.json")
