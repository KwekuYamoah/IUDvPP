# Script for validating a saved model on a new dataset for inference

import torch
import json
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from prosody_bilstm_features_only_multiclass import ProsodyDataset, collate_fn, Seq2Seq, Encoder, Decoder, clean_up_sentence
from prosody_raw_features_only import FeatureProjection
from prosody_raw_features_only import Seq2Seq as ProRawSeq2Seq, Encoder as ProRawEncoder, Decoder as ProRawDecoder, ProsodyDataset as ProRawDataset, collate_fn as pro_raw_collate_fn

# Load data function
def load_data(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
    return data

# Load model and perform inference on new dataset
def validate_prosody_saved_model(model_path, new_dataset_path, batch_size):
    # Load new dataset
    new_data = load_data(new_dataset_path)
    new_dataset = ProsodyDataset(new_data)
    new_loader = DataLoader(new_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Determine feature dimension dynamically from the dataset
    sample_words, sample_features, sample_labels, sample_lengths = next(iter(new_loader))
    feature_dim = sample_features.shape[2]

    # Define encoder and decoder to initialize Seq2Seq
    hidden_dim = 128   # Replace with actual hidden dimension
    num_layers = 2     # Replace with actual number of layers
    dropout = 0.46413941258903124      # Replace with actual dropout value
    num_attention_layers = 4  # Replace with actual number of attention layers
    num_classes = 4  # Determine number of classes from the labels

    # Load the saved model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    encoder = Encoder(feature_dim, hidden_dim, num_layers, dropout).to(device)
    decoder = Decoder(hidden_dim, num_layers=num_layers, output_dim=num_classes, dropout=dropout, num_attention_layers=num_attention_layers).to(device)
    model = Seq2Seq(encoder, decoder).to(device)

    # Load the model state dictionary to CPU
    state_dict = torch.load(model_path, weights_only=True, map_location=torch.device('cpu'))

    # Load the state dictionary into the model
    model.load_state_dict(state_dict)

    # Move the model to the appropriate device
    model = model.to(device)

   
    model.eval()

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for words, features, labels, lengths in new_loader:
            features = features.to(device)
            lengths = lengths.to(device)

            # Perform inference
            output = model(features, lengths)
            preds = torch.argmax(output, dim=2)

            for i in range(features.shape[0]):
                word_sentence = words[i]
                gold_labels = labels[i].cpu().numpy().flatten()
                pred_labels = preds[i].cpu().numpy().flatten()

                # Clean up the sentence by excluding padding positions
                cleaned_words, cleaned_gold_labels, cleaned_pred_labels = clean_up_sentence(
                    word_sentence, gold_labels, pred_labels
                )

                # Collect valid labels and predictions for metrics
                all_labels.extend(cleaned_gold_labels)
                all_preds.extend(cleaned_pred_labels)

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    # Print metrics
    print('Validation on Unambiguous Set Prosodic Features:')
    print(f'  Accuracy: {accuracy * 100:.2f}%')
    print(f'  Precision: {precision * 100:.2f}%')
    print(f'  Recall: {recall * 100:.2f}%')
    print(f'  F1 Score: {f1 * 100:.2f}%')

# Load model and perform inference for another trained model on new dataset
def validate_raw_saved_model(model_path, new_dataset_path, batch_size):

    # Load new dataset
    new_data = load_data(new_dataset_path)
    new_dataset = ProsodyDataset(new_data)
    new_loader = DataLoader(new_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Determine feature dimension dynamically from the dataset
    sample_words, sample_features, sample_labels, sample_lengths = next(iter(new_loader))
    feature_dim = sample_features.shape[2]
    target_feature_dim = 28114  # Set to the required dimension by the model

    # Pad features with zeros if the current feature dimension is smaller than the target
    if feature_dim < target_feature_dim:
        pad_size = target_feature_dim - feature_dim
        sample_features = torch.nn.functional.pad(sample_features, (0, pad_size), "constant", 0)

    # Define encoder and decoder to initialize Seq2Seq
    hidden_dim = 256   # Replace with actual hidden dimension
    num_layers = 8     # Replace with actual number of layers
    dropout = 0.13234009854266668  # Replace with actual dropout value
    num_attention_layers = 8  # Replace with actual number of attention layers
    num_classes = 4  # Determine number of classes from the labels

    # Load the saved model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = Encoder(target_feature_dim, hidden_dim, num_layers, dropout).to(device)
    decoder = Decoder(hidden_dim, output_dim=num_classes, num_layers=num_layers, dropout=dropout, num_attention_layers=num_attention_layers).to(device)
    model = Seq2Seq(encoder, decoder).to(device)

    # Load the model state dictionary to CPU
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))

    # Load the state dictionary into the model
    model.load_state_dict(state_dict)

    # Move the model to the appropriate device
    model = model.to(device)

    model.eval()

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for words, features, labels, lengths in new_loader:
            # Pad features if necessary
            if features.shape[2] < target_feature_dim:
                pad_size = target_feature_dim - features.shape[2]
                features = torch.nn.functional.pad(features, (0, pad_size), "constant", 0)

            features = features.to(device)
            lengths = lengths.to(device)

            # Perform inference
            output = model(features, lengths)
            preds = torch.argmax(output, dim=2)

            for i in range(features.shape[0]):
                word_sentence = words[i]
                gold_labels = labels[i].cpu().numpy().flatten()
                pred_labels = preds[i].cpu().numpy().flatten()

                # Clean up the sentence by excluding padding positions
                cleaned_words, cleaned_gold_labels, cleaned_pred_labels = clean_up_sentence(
                    word_sentence, gold_labels, pred_labels
                )

                # Collect valid labels and predictions for metrics
                all_labels.extend(cleaned_gold_labels)
                all_preds.extend(cleaned_pred_labels)

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    # Print metrics
    print('Validation on Unambiguous Set Raw Audio Features:')
    print(f'  Accuracy: {accuracy * 100:.2f}%')
    print(f'  Precision: {precision * 100:.2f}%')
    print(f'  Recall: {recall * 100:.2f}%')
    print(f'  F1 Score: {f1 * 100:.2f}%')

def validate_prosody_raw_model(model_path, new_dataset_path, batch_size):
    # Load new dataset
    new_data = load_data(new_dataset_path)
    new_dataset = ProRawDataset(new_data)
    new_loader = DataLoader(new_dataset, batch_size=batch_size, shuffle=False, collate_fn=pro_raw_collate_fn)

    # Determine feature dimensions dynamically from the dataset
    sample_words, sample_prosodic_features, sample_raw_acoustic_features, sample_labels, sample_lengths = next(iter(new_loader))
    prosodic_features_dim = sample_prosodic_features.shape[2]
    raw_acoustic_features_dim = sample_raw_acoustic_features.shape[2]

    # Define model parameters (same as training)
    PROJECTED_DIM = 128
    HIDDEN_DIM = 256
    NUM_LAYERS = 8
    DROPOUT = 0.13234009854266668
    NUM_ATTENTION_LAYERS = 8
    num_classes = 4  # Determine number of classes from the labels

    # Load the saved model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Instantiate the projection layers
    prosodic_projection = FeatureProjection(prosodic_features_dim, PROJECTED_DIM).to(device)
    acoustic_projection = FeatureProjection(raw_acoustic_features_dim, PROJECTED_DIM).to(device)

    # Instantiate the encoder and decoder
    encoder = ProRawEncoder(PROJECTED_DIM * 2, HIDDEN_DIM, NUM_LAYERS, DROPOUT).to(device)
    decoder = ProRawDecoder(HIDDEN_DIM, num_classes, NUM_LAYERS, DROPOUT, NUM_ATTENTION_LAYERS).to(device)
    model = ProRawSeq2Seq(encoder, decoder, prosodic_projection, acoustic_projection).to(device)

    # Load the model state dictionary to CPU
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))

     # Adjust the shape of the projection layer weights if there is a size mismatch
    if 'acoustic_projection.linear.weight' in state_dict and state_dict['acoustic_projection.linear.weight'].shape[1] != raw_acoustic_features_dim:
        print("Adjusting acoustic projection layer weight dimensions due to size mismatch.")
        state_dict['acoustic_projection.linear.weight'] = torch.nn.functional.pad(
            state_dict['acoustic_projection.linear.weight'],
            (0, raw_acoustic_features_dim - state_dict['acoustic_projection.linear.weight'].shape[1]),
            mode='constant',
            value=0
        )

    # Load the state dictionary into the model
    model.load_state_dict(state_dict)

    # Move the model to the appropriate device
    model = model.to(device)

    model.eval()

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for words, prosodic_features, raw_acoustic_features, labels, lengths in new_loader:
            prosodic_features = prosodic_features.to(device)
            raw_acoustic_features = raw_acoustic_features.to(device)
            lengths = lengths.to(device)

            # Perform inference
            output = model(prosodic_features, raw_acoustic_features, lengths)
            preds = torch.argmax(output, dim=2)

            for i in range(prosodic_features.shape[0]):
                word_sentence = words[i]
                gold_labels = labels[i].cpu().numpy().flatten()
                pred_labels = preds[i].cpu().numpy().flatten()

                # Clean up the sentence by excluding padding positions
                cleaned_words, cleaned_gold_labels, cleaned_pred_labels = clean_up_sentence(
                    word_sentence, gold_labels, pred_labels
                )

                # Collect valid labels and predictions for metrics
                all_labels.extend(cleaned_gold_labels)
                all_preds.extend(cleaned_pred_labels)

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    # Print metrics
    print('Validation on New Dataset:')
    print(f'  Accuracy: {accuracy * 100:.2f}%')
    print(f'  Precision: {precision * 100:.2f}%')
    print(f'  Recall: {recall * 100:.2f}%')
    print(f'  F1 Score: {f1 * 100:.2f}%')



if __name__ == "__main__":
    # Specify paths and parameters
    prosody_model_path = '../prosody/models/best-model-ambiguous_instructions-prosody_multiclass.pt'
    prosody_dataset_path = '../prosody/data/multi_label_features.json'
    batch_size = 32

    # Validate the saved model on the new dataset
    validate_prosody_saved_model(prosody_model_path, prosody_dataset_path, batch_size)

    raw_model_path = '../prosody/models/best-model-ambiguous_instructions-raw_audio_multiclass.pt'
    raw_dataset_path = '../prosody/data/multi_label_extracted_raw_audio_features.json'
    # Validate another trained model on the new dataset
    validate_raw_saved_model(raw_model_path, prosody_dataset_path, batch_size)

    # Validate the prosody-raw model on the new dataset
    pro_raw_model_path = '../prosody/models/best-model-ambiguous_instructions-prosody_raw_multiclass.pt'
    pro_raw_dataset_path = '../prosody/data/multi_label_prosodic_raw_acoustic_features.json'
    validate_prosody_raw_model(pro_raw_model_path, pro_raw_dataset_path, batch_size)