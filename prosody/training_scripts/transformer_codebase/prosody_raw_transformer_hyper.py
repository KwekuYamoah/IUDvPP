import os
import json
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    multilabel_confusion_matrix, precision_recall_fscore_support
)
import re
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# Import Optuna
import optuna

# Set the random seeds for reproducibility
def set_seed(seed):
    """
    Set the random seed for all random number generators to ensure reproducibility.

    Args:
        seed (int): The seed value to use for random number generation.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using CUDA
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# Custom padding value for labels
PADDING_VALUE = -100  # Use -100 for ignore_index in CrossEntropyLoss

class EarlyStopping:
    """
    Implements early stopping to stop training when the validation loss stops improving.
    
    Attributes:
        patience (int): Number of epochs to wait before stopping after no improvement.
        min_delta (float): Minimum change to consider as an improvement.
        best_loss (float): Best observed validation loss.
        counter (int): Counter for the number of epochs without improvement.
        early_stop (bool): Flag indicating whether training should stop.
    """
    def __init__(self, patience=5, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_metric = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, metric):
        """
        Call method to check if training should stop based on validation metric.

        Args:
            metric (float): Current validation metric (e.g., F1 score).
        """
        if self.best_metric is None:
            self.best_metric = metric
        elif metric > self.best_metric + self.min_delta:
            self.best_metric = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# Load data from a JSON file
def load_data(json_path):
    """
    Load data from a JSON file.

    Args:
        json_path (str): The path to the JSON file containing data.

    Returns:
        dict: The data loaded from the JSON file.
    """
    with open(json_path, 'r') as file:
        data = json.load(file)
    return data

# Split the data into train, validation, and test sets
def split_data(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Split the data into train, validation, and test sets.

    Args:
        data (dict): The input data as a dictionary.
        train_ratio (float): The proportion of data to use for training.
        val_ratio (float): The proportion of data to use for validation.
        test_ratio (float): The proportion of data to use for testing.

    Returns:
        tuple: A tuple containing the train, validation, and test datasets as dictionaries.
    """
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1"

    data_items = list(data.items())
    train_val_items, test_items = train_test_split(
        data_items, test_size=test_ratio, random_state=42
    )
    train_items, val_items = train_test_split(
        train_val_items, test_size=val_ratio / (train_ratio + val_ratio), random_state=42
    )

    return dict(train_items), dict(val_items), dict(test_items)

# Check for overlap between datasets
def check_overlap(train_data, val_data, test_data):
    """
    Check for any overlap between the train, validation, and test sets.

    Args:
        train_data (dict): The training dataset.
        val_data (dict): The validation dataset.
        test_data (dict): The test dataset.

    Raises:
        AssertionError: If there is overlap between the datasets.

    Returns:
        None
    """
    train_keys = set(train_data.keys())
    val_keys = set(val_data.keys())
    test_keys = set(test_data.keys())

    assert train_keys.isdisjoint(val_keys), "Train and Validation sets overlap!"
    assert train_keys.isdisjoint(test_keys), "Train and Test sets overlap!"
    assert val_keys.isdisjoint(test_keys), "Validation and Test sets overlap!"

    print("No overlap between datasets.")

# Preprocess words (remove punctuation and tokenize)
def preprocess_text(words):
    """
    Preprocess a list of words by removing punctuation and tokenizing.

    Args:
        words (list): A list of words or sentences to preprocess.

    Returns:
        list: A list of processed words.
    """
    processed_words = []
    for word in words:
        processed_words.extend(re.findall(r"[\w']+|[.,!?;]", word))
    return processed_words

# Clean up sentence by removing padding
def clean_up_sentence(words, gold_labels, pred_labels, padding_value):
    """
    Remove padding from words, gold labels, and predicted labels.

    Args:
        words (list): A list of words.
        gold_labels (list): A list of gold labels.
        pred_labels (list): A list of predicted labels.
        padding_value (int): The padding value to ignore.

    Returns:
        tuple: A tuple containing filtered words, gold labels, and predicted labels.
    """
    filtered_words = []
    filtered_gold_labels = []
    filtered_pred_labels = []

    for i in range(len(words)):
        if gold_labels[i] != padding_value:
            filtered_words.append(words[i])
            filtered_gold_labels.append(int(gold_labels[i]))
            filtered_pred_labels.append(int(pred_labels[i]))
    return filtered_words, filtered_gold_labels, filtered_pred_labels

# Custom dataset class for prosody and raw acoustic features
class ProsodyDataset(Dataset):
    """
    Custom Dataset for handling prosody and raw acoustic features for sequence labeling.

    Args:
        data (dict): The dataset containing words, features, and labels.
        scaler_prosody (StandardScaler, optional): A scaler for normalizing the prosodic features.
        scaler_acoustic (StandardScaler, optional): A scaler for normalizing the acoustic features.
    """
    def __init__(self, data, scaler_prosody=None, scaler_acoustic=None):
        self.entries = list(data.items())
        self.scaler_prosody = scaler_prosody
        self.scaler_acoustic = scaler_acoustic

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        key, item = self.entries[idx]
        words = preprocess_text(item['words'])
        prosodic_features = torch.tensor(item['prosodic_features'], dtype=torch.float32)
        raw_acoustic_features = torch.tensor(item['raw_acoustic_features'], dtype=torch.float32)
        if self.scaler_prosody is not None:
            prosodic_features_np = prosodic_features.numpy()
            prosodic_features_np = self.scaler_prosody.transform(prosodic_features_np)
            prosodic_features = torch.tensor(prosodic_features_np, dtype=torch.float32)
        if self.scaler_acoustic is not None:
            acoustic_features_np = raw_acoustic_features.numpy()
            acoustic_features_np = self.scaler_acoustic.transform(acoustic_features_np)
            raw_acoustic_features = torch.tensor(acoustic_features_np, dtype=torch.float32)
        labels = torch.tensor(item['labels'], dtype=torch.long)
        return words, prosodic_features, raw_acoustic_features, labels

# Custom collate function to handle padding
def collate_fn(batch):
    """
    Custom collate function to handle padding for batching.

    Args:
        batch (list): A list of tuples containing words, features, and labels.

    Returns:
        tuple: A tuple containing padded features, padded labels, and feature lengths.
    """
    words = [item[0] for item in batch]  # List of lists of words
    prosodic_features = [item[1] for item in batch]
    raw_acoustic_features = [item[2] for item in batch]
    labels = [item[3] for item in batch]

    prosodic_features_padded = torch.nn.utils.rnn.pad_sequence(prosodic_features, batch_first=True, padding_value=0.0)
    acoustic_features_padded = torch.nn.utils.rnn.pad_sequence(raw_acoustic_features, batch_first=True, padding_value=0.0)
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=PADDING_VALUE)
    feature_lengths = torch.tensor([len(f) for f in prosodic_features])  # Assuming lengths are the same

    return words, prosodic_features_padded, acoustic_features_padded, labels_padded, feature_lengths

# Positional Encoding for Transformer
class PositionalEncoding(nn.Module):
    """
    Implements positional encoding for the Transformer model.

    Args:
        d_model (int): The dimension of the model.
        max_len (int): The maximum length of the input sequence.
    """
    def __init__(self, d_model, max_len=10000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))  # [d_model/2]
        pe[:, 0::2] = torch.sin(position * div_term)  # [max_len, d_model/2]
        pe[:, 1::2] = torch.cos(position * div_term)  # [max_len, d_model/2]
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Add positional encoding to the input tensor.

        Args:
            x (Tensor): Input tensor of shape [batch_size, seq_len, d_model].

        Returns:
            Tensor: The input tensor with positional encoding added.
        """
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return x

# Transformer Encoder with Padding Masking
class TransformerEncoder(nn.Module):
    """
    Implements a Transformer encoder for sequence modeling.

    Args:
        prosodic_dim (int): Dimension of the prosodic features.
        acoustic_dim (int): Dimension of the raw acoustic features.
        hidden_dim (int): Dimension of the hidden state.
        num_layers (int): Number of encoder layers.
        dropout (float): Dropout rate.
        num_heads (int): Number of attention heads.
    """
    def __init__(self, prosodic_dim, acoustic_dim, hidden_dim, num_layers, dropout, num_heads=8):
        super(TransformerEncoder, self).__init__()
        # Project each feature type to hidden_dim
        self.prosodic_fc = nn.Linear(prosodic_dim, hidden_dim)
        self.acoustic_fc = nn.Linear(acoustic_dim, hidden_dim)
        # Combine projected features
        self.combine_fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.positional_encoding = PositionalEncoding(d_model=hidden_dim, max_len=5000)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True  # Align with data format
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, prosodic_features, acoustic_features, src_key_padding_mask=None):
        """
        Forward pass for the Transformer encoder.

        Args:
            prosodic_features (Tensor): Prosodic features [batch_size, seq_len, prosodic_dim]
            acoustic_features (Tensor): Raw acoustic features [batch_size, seq_len, acoustic_dim]
            src_key_padding_mask (Tensor, optional): Padding mask.

        Returns:
            Tensor: Output memory of the encoder.
        """
        # Project features separately
        prosodic_proj = self.prosodic_fc(prosodic_features)  # [batch_size, seq_len, hidden_dim]
        acoustic_proj = self.acoustic_fc(acoustic_features)  # [batch_size, seq_len, hidden_dim]
        # Combine features by concatenation
        combined_features = torch.cat((prosodic_proj, acoustic_proj), dim=2)  # [batch_size, seq_len, hidden_dim * 2]
        # Reduce back to hidden_dim
        src = self.combine_fc(combined_features)  # [batch_size, seq_len, hidden_dim]
        src = self.positional_encoding(src)  # [batch_size, seq_len, hidden_dim]
        src = self.dropout(src)
        memory = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)  # [batch_size, seq_len, hidden_dim]
        return memory

# Function to create padding masks based on lengths
def create_src_padding_mask(lengths, max_len):
    """
    Creates a padding mask for the encoder based on the lengths of the input sequences.

    Args:
        lengths (Tensor): Tensor of shape [batch_size] containing the lengths of each sequence.
        max_len (int): Maximum sequence length in the batch.

    Returns:
        Tensor: Padding mask of shape [batch_size, max_len], where True indicates padding positions.
    """
    batch_size = lengths.size(0)
    device = lengths.device  # Get the device of lengths tensor
    mask = torch.arange(max_len, device=device).expand(batch_size, max_len) >= lengths.unsqueeze(1)
    return mask  # [batch_size, max_len]

# Transformer-based Sequence Labeling Model
class TransformerClassifier(nn.Module):
    """
    Implements a Transformer-based sequence labeling model.

    Args:
        prosodic_dim (int): Dimension of the prosodic features.
        acoustic_dim (int): Dimension of the raw acoustic features.
        hidden_dim (int): Dimension of the hidden state.
        num_layers (int): Number of encoder layers.
        num_classes (int): Number of output classes.
        dropout (float): Dropout rate.
        num_heads (int): Number of attention heads.
    """
    def __init__(self, prosodic_dim, acoustic_dim, hidden_dim, num_layers, num_classes, dropout, num_heads=8):
        super(TransformerClassifier, self).__init__()
        self.encoder = TransformerEncoder(prosodic_dim, acoustic_dim, hidden_dim, num_layers, dropout, num_heads)
        self.fc_out = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, prosodic_features, acoustic_features, lengths):
        """
        Forward pass for the Transformer-based classifier.

        Args:
            prosodic_features (Tensor): Prosodic features [batch_size, seq_len, prosodic_dim]
            acoustic_features (Tensor): Raw acoustic features [batch_size, seq_len, acoustic_dim]
            lengths (Tensor): Lengths of each sequence in the batch.

        Returns:
            Tensor: Output logits of shape [batch_size, seq_len, num_classes].
        """
        max_len = prosodic_features.size(1)
        src_key_padding_mask = create_src_padding_mask(lengths, max_len)
        memory = self.encoder(prosodic_features, acoustic_features, src_key_padding_mask=src_key_padding_mask)  # [batch_size, seq_len, hidden_dim]
        outputs = self.fc_out(memory)  # [batch_size, seq_len, num_classes]
        return outputs

# Training function
def train_model(model, iterator, optimizer, criterion, num_classes):
    """
    Train the Transformer model for one epoch.

    Args:
        model (nn.Module): The Transformer model to train.
        iterator (DataLoader): DataLoader for the training data.
        optimizer (Optimizer): Optimizer for model parameters.
        criterion (Loss): Loss function.
        num_classes (int): Number of output classes.

    Returns:
        float: The average loss for the epoch.
    """
    model.train()
    epoch_loss = 0
    for batch_idx, (words, prosodic_features, acoustic_features, labels, lengths) in enumerate(iterator):
        prosodic_features = prosodic_features.to(device)
        acoustic_features = acoustic_features.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)

        optimizer.zero_grad()
        outputs = model(prosodic_features, acoustic_features, lengths)  # [batch_size, seq_len, num_classes]
        outputs = outputs.view(-1, num_classes)  # [batch_size * seq_len, num_classes]
        labels_flat = labels.view(-1)  # [batch_size * seq_len]

        # Compute loss
        loss = criterion(outputs, labels_flat)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

# Evaluation function
def evaluate_model(model, iterator, criterion, num_classes):
    """
    Evaluate the Transformer model on validation data.

    Args:
        model (nn.Module): The Transformer model to evaluate.
        iterator (DataLoader): DataLoader for the validation data.
        criterion (Loss): Loss function.
        num_classes (int): Number of output classes.

    Returns:
        tuple: A tuple containing the average loss, accuracy, precision, recall, and F1 score.
    """
    model.eval()
    epoch_loss = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for batch_idx, (words, prosodic_features, acoustic_features, labels, lengths) in enumerate(iterator):
            prosodic_features = prosodic_features.to(device)
            acoustic_features = acoustic_features.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)

            outputs = model(prosodic_features, acoustic_features, lengths)  # [batch_size, seq_len, num_classes]
            outputs = outputs.view(-1, num_classes)  # [batch_size * seq_len, num_classes]
            labels_flat = labels.view(-1)  # [batch_size * seq_len]

            preds = torch.argmax(outputs, dim=1)  # [batch_size * seq_len]

            loss = criterion(outputs, labels_flat)
            epoch_loss += loss.item()

            # Exclude padding indices from metrics
            non_pad_indices = labels_flat != PADDING_VALUE
            labels_np = labels_flat[non_pad_indices].cpu().numpy()
            preds_np = preds[non_pad_indices].cpu().numpy()
            all_labels.extend(labels_np)
            all_preds.extend(preds_np)

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    return epoch_loss / len(iterator), accuracy, precision, recall, f1

# Test model function
def test_model(model, iterator):
    """
    Test the Transformer model on test data and output predictions and metrics.

    Args:
        model (nn.Module): The Transformer model to test.
        iterator (DataLoader): DataLoader for the test data.

    Returns:
        tuple: A tuple containing all labels and predictions for the test data.
    """
    model.eval()
    all_labels = []
    all_preds = []

    os.makedirs('./outputs', exist_ok=True)
    with open('./outputs/prosody_raw_transformer_multiclass_results.txt', 'w') as file:
        file.write("")

    with torch.no_grad():
        for batch_idx, (words, prosodic_features, acoustic_features, labels, lengths) in enumerate(iterator):
            prosodic_features = prosodic_features.to(device)
            acoustic_features = acoustic_features.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)

            outputs = model(prosodic_features, acoustic_features, lengths)  # [batch_size, seq_len, num_classes]
            preds = torch.argmax(outputs, dim=2)  # [batch_size, seq_len]

            for i in range(prosodic_features.size(0)):
                word_sentence = words[i]  # List of words
                gold_labels = labels[i].cpu().numpy().flatten()
                pred_labels = preds[i].cpu().numpy().flatten()

                # Clean up the sentence by excluding padding positions
                cleaned_words, cleaned_gold_labels, cleaned_pred_labels = clean_up_sentence(
                    word_sentence, gold_labels, pred_labels, padding_value=PADDING_VALUE
                )

                # Create DataFrame
                data = {
                    'Word': cleaned_words,
                    'Gold Label': cleaned_gold_labels,
                    'Predicted Label': cleaned_pred_labels
                }

                df = pd.DataFrame(data)
                with open('./outputs/prosody_raw_transformer_multiclass_results.txt', 'a') as file:
                    file.write(df.to_string(index=False))
                    file.write("\n" + "-" * 50 + "\n")

                # Collect valid labels and predictions for metrics
                all_labels.extend(cleaned_gold_labels)
                all_preds.extend(cleaned_pred_labels)

    # Calculate metrics using only valid labels and predictions
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    print('*' * 45)
    print(f'Test Accuracy: {accuracy*100:.2f}%')
    print(f'Test Precision: {precision*100:.2f}%')
    print(f'Test Recall: {recall*100:.2f}%')
    print(f'Test F1 Score: {f1*100:.2f}%')
    print('*' * 45)

    return all_labels, all_preds

# Plot metrics function
def plot_metrics(train_losses, val_losses, val_accuracies, val_precisions, val_recalls, val_f1s):
    """
    Plot training and validation metrics over epochs.

    Args:
        train_losses (list): List of training losses.
        val_losses (list): List of validation losses.
        val_accuracies (list): List of validation accuracies.
        val_precisions (list): List of validation precisions.
        val_recalls (list): List of validation recalls.
        val_f1s (list): List of validation F1 scores.

    Returns:
        None
    """
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(18, 12))
    
    plt.subplot(2, 3, 1)
    plt.plot(epochs, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs, val_losses, label='Validation Loss', marker='o')
    plt.legend()
    plt.title('Loss')
    
    plt.subplot(2, 3, 2)
    plt.plot(epochs, val_accuracies, label='Validation Accuracy', marker='o')
    plt.legend()
    plt.title('Accuracy')
    
    plt.subplot(2, 3, 3)
    plt.plot(epochs, val_precisions, label='Validation Precision', marker='o')
    plt.legend()
    plt.title('Precision')
    
    plt.subplot(2, 3, 4)
    plt.plot(epochs, val_recalls, label='Validation Recall', marker='o')
    plt.legend()
    plt.title('Recall')
    
    plt.subplot(2, 3, 5)
    plt.plot(epochs, val_f1s, label='Validation F1 Score', marker='o')
    plt.legend()
    plt.title('F1 Score')
    
    plt.tight_layout()
    plt.savefig('./outputs/prosody_raw_transformer_multiclass_metrics.png')
    plt.close()

# Evaluate on a new dataset
def evaluate_new_set(model, new_dataset_path, scaler_prosody, scaler_acoustic):
    """
    Evaluate the model on a new dataset.

    Args:
        model (nn.Module): The Transformer model to evaluate.
        new_dataset_path (str): Path to the new dataset JSON file.
        scaler_prosody (StandardScaler): Scaler for normalizing the prosodic features.
        scaler_acoustic (StandardScaler): Scaler for normalizing the acoustic features.

    Returns:
        tuple: A tuple containing all labels and predictions for the new dataset.
    """
    # Load new data
    new_data = load_data(new_dataset_path)
    new_dataset = ProsodyDataset(new_data, scaler_prosody=scaler_prosody, scaler_acoustic=scaler_acoustic)
    new_loader = DataLoader(new_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
    
    # Test the model on the new dataset and get predictions
    print('\n\nEvaluation on Held Out Set Dataset:')
    all_labels, all_preds = test_model(model, new_loader)

    return all_labels, all_preds

# Validate label integrity
def validate_labels(datasets, num_classes):
    """
    Validate that all labels are within the expected range.

    Args:
        datasets (list): A list of datasets to validate.
        num_classes (int): The number of output classes.

    Raises:
        ValueError: If any labels are found that are outside the expected range.
    """
    for dataset in datasets:
        for idx, (_, _, _, labels) in enumerate(dataset):
            invalid_mask = (labels >= num_classes) | (labels < 0)
            if torch.any(invalid_mask):
                invalid_labels = labels[invalid_mask].unique().tolist()
                raise ValueError(f"Found invalid labels {invalid_labels} in dataset at index {idx}. Labels should be in the range [0, {num_classes - 1}].")
    print("All labels are valid.")

# Get all unique labels across multiple datasets
def get_all_unique_labels(datasets):
    """
    Get all unique labels across multiple datasets.

    Args:
        datasets (list): A list of datasets.

    Returns:
        set: A set of unique labels across all datasets.
    """
    unique_labels = set()
    for dataset in datasets:
        for _, _, _, labels in dataset:
            unique_labels.update(labels.numpy().flatten())
    return unique_labels

# Function to get unique labels from data
def get_labels_from_data(data):
    """
    Get all unique labels from the data.

    Args:
        data (dict): The data containing labels.

    Returns:
        list: A sorted list of unique labels.
    """
    unique_labels = set()
    for key, item in data.items():
        labels = item['labels']
        unique_labels.update(labels)
    return sorted(unique_labels)

# Function to compute class weights
def compute_class_weights(dataset, num_classes):
    """
    Compute class weights for handling label imbalance.

    Args:
        dataset (Dataset): The dataset to compute class weights from.
        num_classes (int): The number of output classes.

    Returns:
        Tensor: A tensor containing the class weights.
    """
    all_labels = []
    for _, _, _, labels in dataset:
        labels_np = labels.numpy().flatten()
        labels_np = labels_np[labels_np != PADDING_VALUE]  # Exclude padding labels
        all_labels.extend(labels_np)
    classes = np.arange(num_classes)
    class_weights = compute_class_weight('balanced', classes=classes, y=all_labels)
    weight_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    return weight_tensor

# Function to fit feature scaler
def fit_feature_scaler(dataset, feature_type='prosodic'):
    """
    Fit a feature scaler on the dataset.

    Args:
        dataset (Dataset): The dataset to fit the scaler on.
        feature_type (str): Type of features to scale ('prosodic' or 'acoustic').

    Returns:
        StandardScaler: The fitted scaler.
    """
    all_features = []
    for _, prosodic_features, acoustic_features, _ in dataset:
        if feature_type == 'prosodic':
            all_features.append(prosodic_features.numpy())
        elif feature_type == 'acoustic':
            all_features.append(acoustic_features.numpy())
    all_features = np.concatenate(all_features, axis=0)
    scaler = StandardScaler()
    scaler.fit(all_features)
    return scaler

# ==============================
# Main Execution Block
# ==============================

if __name__ == "__main__":
    # Set seed for reproducibility
    set_seed(42)
    json_path = '../prosody/data/ambiguous_prosodic_raw_acoustic_ml_features_train.json'
    data = load_data(json_path)

    # Split data
    train_data, val_data, test_data = split_data(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
    check_overlap(train_data, val_data, test_data)

    # Create temporary dataset to fit scalers
    temp_train_dataset = ProsodyDataset(train_data)
    scaler_prosody = fit_feature_scaler(temp_train_dataset, feature_type='prosodic')
    scaler_acoustic = fit_feature_scaler(temp_train_dataset, feature_type='acoustic')

    # Create datasets
    train_dataset = ProsodyDataset(train_data, scaler_prosody=scaler_prosody, scaler_acoustic=scaler_acoustic)
    val_dataset = ProsodyDataset(val_data, scaler_prosody=scaler_prosody, scaler_acoustic=scaler_acoustic)
    test_dataset = ProsodyDataset(test_data, scaler_prosody=scaler_prosody, scaler_acoustic=scaler_acoustic)

    # Determine number of classes based on all datasets
    all_unique_labels = get_all_unique_labels([train_dataset, val_dataset, test_dataset])
    NUM_CLASSES = len(all_unique_labels)
    print(f"All unique labels across datasets: {sorted(all_unique_labels)}")
    print(f"Model Training with {NUM_CLASSES} classes")

    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Validate labels before training
    validate_labels([train_dataset, val_dataset, test_dataset], NUM_CLASSES)

    # Define the objective function for Optuna
    def objective(trial):
        # Sample hyperparameters
        HIDDEN_DIM = trial.suggest_int('hidden_dim', 128, 512, step=64)
        NUM_LAYERS = trial.suggest_int('num_layers', 1, 6)
        DROPOUT = trial.suggest_float('dropout', 0.1, 0.5, step=0.1)
        NUM_HEADS = trial.suggest_categorical('num_heads', [2, 4, 8])
        lr = trial.suggest_loguniform('lr', 1e-5, 1e-3)
        weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-3)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])

        # Set seed for reproducibility per trial
        set_seed(42 + trial.number)

        # Create data loaders with the sampled batch size
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        # Retrieve feature dimensions
        sample_words, sample_prosodic_features, sample_acoustic_features, sample_labels, sample_lengths = next(iter(train_loader))
        prosodic_dim = sample_prosodic_features.shape[2]
        acoustic_dim = sample_acoustic_features.shape[2]

        # Initialize the model
        model = TransformerClassifier(
            prosodic_dim=prosodic_dim,
            acoustic_dim=acoustic_dim,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            num_classes=NUM_CLASSES,
            dropout=DROPOUT,
            num_heads=NUM_HEADS
        ).to(device)

        # Compute class weights
        class_weights = compute_class_weights(train_dataset, NUM_CLASSES)

        # Define optimizer
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        # Optionally define scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-6)

        # Define loss function
        criterion = nn.CrossEntropyLoss(ignore_index=PADDING_VALUE, weight=class_weights)

        # Training parameters
        N_EPOCHS = 50  # Limit number of epochs to keep training time reasonable
        early_stopping = EarlyStopping(patience=5, min_delta=0.0001)
        best_valid_f1 = 0.0  # To track best F1 score

        for epoch in range(N_EPOCHS):
            train_loss = train_model(model, train_loader, optimizer, criterion, num_classes=NUM_CLASSES)
            valid_loss, valid_acc, valid_precision, valid_recall, valid_f1 = evaluate_model(model, val_loader, criterion, num_classes=NUM_CLASSES)

            # Update the learning rate scheduler
            scheduler.step(valid_f1)

            # Report intermediate objective value
            trial.report(valid_f1, epoch)

            # Handle pruning based on the intermediate value
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            # Save the best F1 score
            if valid_f1 > best_valid_f1:
                best_valid_f1 = valid_f1

            # Early stopping check
            early_stopping(valid_f1)
            if early_stopping.early_stop:
                break

        return best_valid_f1

    # Create the study and optimize
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)

    # Print best hyperparameters
    print('Number of finished trials:', len(study.trials))
    print('Best trial:')
    trial = study.best_trial

    print('  Best F1 Score: {:.4f}'.format(trial.value))
    print('  Best Hyperparameters:')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))

    # After hyperparameter tuning, retrain the model with the best hyperparameters
    # Optionally, you can save and test the final model here
