"""
Prosody Classification using Transformer with Multi-Feature Integration and Hyperparameter Optimization

This script implements a Transformer-based sequence labeling model for multiclass classification on prosody data.
It integrates both prosodic and raw acoustic features, includes data preprocessing, model definition,
training with hyperparameter optimization using Optuna, evaluation, testing, and visualization of training metrics.

Dependencies:
    - Python 3.x
    - PyTorch
    - NumPy
    - scikit-learn
    - matplotlib
    - pandas
    - torchinfo
    - Optuna

Author: Your Name
Date: 2024-04-27
"""

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
import optuna  # Import Optuna

# ==============================
# Utility Functions
# ==============================

def set_seed(seed):
    """
    Set the random seed for reproducibility across various libraries.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# Custom padding value for labels
PADDING_VALUE = -100  # Use -100 for ignore_index in CrossEntropyLoss

# ==============================
# Early Stopping Class
# ==============================

class EarlyStopping:
    """
    Early stopping to terminate training when validation loss doesn't improve.

    Attributes:
        patience (int): Number of epochs to wait after last improvement.
        min_delta (float): Minimum change to qualify as an improvement.
        best_score (float): Best validation score observed.
        counter (int): Counter for epochs without improvement.
        early_stop (bool): Flag indicating whether to stop early.
    """
    def __init__(self, patience=10, min_delta=0.0001):
        """
        Initializes the EarlyStopping instance.

        Args:
            patience (int, optional): Patience for early stopping. Defaults to 10.
            min_delta (float, optional): Minimum delta for improvement. Defaults to 0.0001.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, score):
        """
        Call method to update early stopping state based on validation score.

        Args:
            score (float): Current epoch's validation score.
        """
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score - self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# ==============================
# Data Loading and Preprocessing
# ==============================

def load_data(json_path):
    """
    Load data from a JSON file.

    Args:
        json_path (str): Path to the JSON data file.

    Returns:
        dict: The loaded data.
    """
    with open(json_path, 'r') as file:
        data = json.load(file)
    return data

def split_data(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Split the data into training, validation, and testing sets.

    Args:
        data (dict): The dataset to split.
        train_ratio (float, optional): Proportion of data for training. Defaults to 0.8.
        val_ratio (float, optional): Proportion of data for validation. Defaults to 0.1.
        test_ratio (float, optional): Proportion of data for testing. Defaults to 0.1.

    Returns:
        tuple: Datasets for training, validation, and testing.
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

def check_overlap(train_data, val_data, test_data):
    """
    Check for overlapping keys between training, validation, and testing datasets.

    Args:
        train_data (dict): Training dataset.
        val_data (dict): Validation dataset.
        test_data (dict): Testing dataset.

    Raises:
        AssertionError: If any overlaps are found.

    Prints:
        Confirmation message if no overlaps are detected.
    """
    train_keys = set(train_data.keys())
    val_keys = set(val_data.keys())
    test_keys = set(test_data.keys())

    assert train_keys.isdisjoint(val_keys), "Train and Validation sets overlap!"
    assert train_keys.isdisjoint(test_keys), "Train and Test sets overlap!"
    assert val_keys.isdisjoint(test_keys), "Validation and Test sets overlap!"

    print("No overlap between datasets.")

def preprocess_text(words):
    """
    Preprocess words by removing punctuation and tokenizing.

    Args:
        words (list): List of words to preprocess.

    Returns:
        list: Tokenized words without punctuation.
    """
    processed_words = []
    for word in words:
        processed_words.extend(re.findall(r"[\w']+|[.,!?;]", word))
    return processed_words

def clean_up_sentence(words, gold_labels, pred_labels, padding_value):
    """
    Clean up the sentence by removing padding and punctuation tokens.

    Args:
        words (list): List of words in the sentence.
        gold_labels (list): Corresponding gold labels.
        pred_labels (list): Corresponding predicted labels.
        padding_value (int): The padding value to ignore.

    Returns:
        tuple: Cleaned words, gold labels, and predicted labels.
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

# ==============================
# Dataset Class
# ==============================

class ProsodyDataset(Dataset):
    """
    Custom Dataset for Prosody Data.

    Each sample consists of processed words, features, and labels.
    """
    def __init__(self, data, scaler=None):
        """
        Initializes the ProsodyDataset.

        Args:
            data (dict): Dictionary containing prosody data with sentence keys.
            scaler (StandardScaler, optional): Scaler for feature normalization. Defaults to None.
        """
        self.entries = list(data.items())
        self.scaler = scaler

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.entries)

    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (words, features, labels)
                - words (list): List of tokenized words.
                - features (torch.Tensor): Tensor of features.
                - labels (torch.Tensor): Tensor of labels.
        """
        key, item = self.entries[idx]
        words = preprocess_text(item['words'])
        features = torch.tensor(item['features'], dtype=torch.float32)
        if self.scaler is not None:
            features_np = features.numpy()
            features_np = self.scaler.transform(features_np)
            features = torch.tensor(features_np, dtype=torch.float32)
        labels = torch.tensor(item['labels'], dtype=torch.long)
        return words, features, labels

# ==============================
# Collate Function
# ==============================

def collate_fn(batch):
    """
    Custom collate function to handle batches with variable sequence lengths.

    Pads sequences and prepares tensors for words, features, and labels.

    Args:
        batch (list): List of samples from the dataset.

    Returns:
        tuple: (words, features_padded, labels_padded, feature_lengths)
            - words (list): List of word lists.
            - features_padded (torch.Tensor): Padded feature tensors (batch_size, max_seq_len, feature_dim).
            - labels_padded (torch.Tensor): Padded label tensors (batch_size, max_seq_len).
            - feature_lengths (torch.Tensor): Original lengths of each sequence (batch_size).
    """
    words = [item[0] for item in batch]  # List of lists of words
    features = [item[1] for item in batch]
    labels = [item[2] for item in batch]

    # Pad feature sequences with 0.0
    features_padded = torch.nn.utils.rnn.pad_sequence(features, batch_first=True, padding_value=0.0)
    # Pad label sequences with PADDING_VALUE (-100)
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=PADDING_VALUE)
    # Record the lengths of each sequence before padding
    feature_lengths = torch.tensor([len(f) for f in features])

    return words, features_padded, labels_padded, feature_lengths

# ==============================
# Positional Encoding for Transformer
# ==============================

class PositionalEncoding(nn.Module):
    """
    Positional Encoding module for adding positional information to input embeddings.

    Attributes:
        pe (torch.Tensor): Tensor containing positional encodings.
    """
    def __init__(self, d_model, max_len=10000):
        """
        Initializes the PositionalEncoding module.

        Args:
            d_model (int): Dimension of the embeddings.
            max_len (int, optional): Maximum length of sequences. Defaults to 10000.
        """
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))  # [d_model/2]
        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sine to even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cosine to odd indices
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)  # Register as buffer to avoid updating during training

    def forward(self, x):
        """
        Forward pass to add positional encoding to input embeddings.

        Args:
            x (torch.Tensor): Input tensor (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Tensor with positional encodings added.
        """
        x = x + self.pe[:, :x.size(1), :]
        return x

# ==============================
# Transformer Encoder with Padding Masking
# ==============================

class TransformerEncoder(nn.Module):
    """
    Transformer Encoder module with padding mask handling.

    Attributes:
        input_fc (nn.Linear): Linear layer to project input features to hidden dimension.
        positional_encoding (PositionalEncoding): Positional encoding module.
        encoder_layer (nn.TransformerEncoderLayer): Single Transformer encoder layer.
        transformer_encoder (nn.TransformerEncoder): Stack of Transformer encoder layers.
        dropout (nn.Dropout): Dropout layer.
    """
    def __init__(self, feature_dim, hidden_dim, num_layers, dropout, num_heads=8):
        """
        Initializes the TransformerEncoder.

        Args:
            feature_dim (int): Dimension of input features.
            hidden_dim (int): Dimension of Transformer hidden states.
            num_layers (int): Number of Transformer encoder layers.
            dropout (float): Dropout rate.
            num_heads (int, optional): Number of attention heads. Defaults to 8.
        """
        super(TransformerEncoder, self).__init__()
        self.input_fc = nn.Linear(feature_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(d_model=hidden_dim, max_len=5000)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True  # Align with data format
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features, src_key_padding_mask=None):
        """
        Forward pass of the TransformerEncoder.

        Args:
            features (torch.Tensor): Input features (batch_size, seq_len, feature_dim).
            src_key_padding_mask (torch.Tensor, optional): Padding mask (batch_size, seq_len). Defaults to None.

        Returns:
            torch.Tensor: Encoder outputs (batch_size, seq_len, hidden_dim).
        """
        src = self.input_fc(features)  # Project input features
        src = self.positional_encoding(src)  # Add positional encoding
        src = self.dropout(src)
        memory = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)  # Transformer encoding
        return memory

# ==============================
# Padding Mask Creation
# ==============================

def create_src_padding_mask(lengths, max_len):
    """
    Creates a padding mask for the encoder based on the lengths of the input sequences.

    Args:
        lengths (torch.Tensor): Lengths of each sequence in the batch (batch_size).
        max_len (int): Maximum sequence length in the batch.

    Returns:
        torch.Tensor: Padding mask (batch_size, max_len), where True indicates padding positions.
    """
    batch_size = lengths.size(0)
    device = lengths.device  # Get the device of lengths tensor
    mask = torch.arange(max_len, device=device).expand(batch_size, max_len) >= lengths.unsqueeze(1)
    return mask  # [batch_size, max_len]

# ==============================
# Transformer-based Sequence Labeling Model
# ==============================

class TransformerClassifier(nn.Module):
    """
    Transformer-based Sequence Labeling Model for multiclass classification.

    Attributes:
        encoder (TransformerEncoder): Transformer encoder module.
        fc_out (nn.Linear): Fully connected layer to generate output predictions.
    """
    def __init__(self, feature_dim, hidden_dim, num_layers, num_classes, dropout, num_heads=8):
        """
        Initializes the TransformerClassifier.

        Args:
            feature_dim (int): Dimension of input features.
            hidden_dim (int): Dimension of Transformer hidden states.
            num_layers (int): Number of Transformer encoder layers.
            num_classes (int): Number of output classes.
            dropout (float): Dropout rate.
            num_heads (int, optional): Number of attention heads. Defaults to 8.
        """
        super(TransformerClassifier, self).__init__()
        self.encoder = TransformerEncoder(feature_dim, hidden_dim, num_layers, dropout, num_heads)
        self.fc_out = nn.Linear(hidden_dim, num_classes)

    def forward(self, features, lengths):
        """
        Forward pass of the TransformerClassifier.

        Args:
            features (torch.Tensor): Input features (batch_size, seq_len, feature_dim).
            lengths (torch.Tensor): Lengths of each sequence in the batch (batch_size).

        Returns:
            torch.Tensor: Output predictions (batch_size, seq_len, num_classes).
        """
        max_len = features.size(1)
        src_key_padding_mask = create_src_padding_mask(lengths, max_len)  # Create padding mask
        memory = self.encoder(features, src_key_padding_mask=src_key_padding_mask)  # Encoder output
        outputs = self.fc_out(memory)  # Generate predictions
        return outputs

# ==============================
# Training and Evaluation Functions
# ==============================

def train_model(model, iterator, optimizer, criterion, num_classes):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The Transformer model.
        iterator (DataLoader): DataLoader for training data.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        criterion (nn.Module): Loss function.
        num_classes (int): Number of output classes.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    epoch_loss = 0
    for batch_idx, (words, features, labels, lengths) in enumerate(iterator):
        features = features.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)

        optimizer.zero_grad()
        outputs = model(features, lengths)  # [batch_size, seq_len, num_classes]
        outputs = outputs.view(-1, num_classes)  # [batch_size * seq_len, num_classes]
        labels_flat = labels.view(-1)  # [batch_size * seq_len]

        # Compute loss
        loss = criterion(outputs, labels_flat)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)  # Gradient clipping
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

def evaluate_model(model, iterator, criterion, num_classes):
    """
    Evaluates the model on validation or test data.

    Args:
        model (nn.Module): The Transformer model.
        iterator (DataLoader): DataLoader for validation or test data.
        criterion (nn.Module): Loss function.
        num_classes (int): Number of output classes.

    Returns:
        tuple: (average_loss, accuracy, precision, recall, f1_score)
    """
    model.eval()
    epoch_loss = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for batch_idx, (words, features, labels, lengths) in enumerate(iterator):
            features = features.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)

            outputs = model(features, lengths)  # [batch_size, seq_len, num_classes]
            outputs = outputs.view(-1, num_classes)  # [batch_size * seq_len, num_classes]
            labels_flat = labels.view(-1)  # [batch_size * seq_len]

            preds = torch.argmax(outputs, dim=1)  # [batch_size * seq_len]

            # Compute loss
            loss = criterion(outputs, labels_flat)
            epoch_loss += loss.item()

            # Exclude padding indices from metrics
            non_pad_indices = labels_flat != PADDING_VALUE
            labels_np = labels_flat[non_pad_indices].cpu().numpy()
            preds_np = preds[non_pad_indices].cpu().numpy()
            all_labels.extend(labels_np)
            all_preds.extend(preds_np)

    # Calculate evaluation metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    return epoch_loss / len(iterator), accuracy, precision, recall, f1

# ==============================
# Testing Function
# ==============================

def test_model(model, iterator):
    """
    Tests the model on the test dataset and saves detailed results.

    Args:
        model (nn.Module): The trained Transformer model.
        iterator (DataLoader): DataLoader for test data.

    Returns:
        tuple: (all_labels, all_preds)
            - all_labels (list): List of true labels.
            - all_preds (list): List of predicted labels.
    """
    model.eval()
    all_labels = []
    all_preds = []

    # Create the output directory if it doesn't exist
    os.makedirs('./outputs', exist_ok=True)
    results_filepath = './outputs/prosody_transformer_multiclass_results.txt'
    with open(results_filepath, 'w') as file:
        file.write("")

    with torch.no_grad():
        for batch_idx, (words, features, labels, lengths) in enumerate(iterator):
            features = features.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)

            outputs = model(features, lengths)  # [batch_size, seq_len, num_classes]
            preds = torch.argmax(outputs, dim=2)  # [batch_size, seq_len]

            for i in range(features.size(0)):
                word_sentence = words[i]  # List of words
                gold_labels = labels[i].cpu().numpy().flatten()
                pred_labels = preds[i].cpu().numpy().flatten()

                PADDING_VALUE_EVAL = PADDING_VALUE  # Use the same padding value

                # Clean up the sentence by excluding padding positions and punctuation
                cleaned_words, cleaned_gold_labels, cleaned_pred_labels = clean_up_sentence(
                    word_sentence, gold_labels, pred_labels, padding_value=PADDING_VALUE_EVAL
                )

                # Create DataFrame with words and their corresponding labels
                data = {
                    'Word': cleaned_words,
                    'Gold Label': cleaned_gold_labels,
                    'Predicted Label': cleaned_pred_labels
                }

                df = pd.DataFrame(data)
                # Append the DataFrame to the results file
                with open(results_filepath, 'a') as file:
                    file.write(df.to_string(index=False))
                    file.write("\n" + "-" * 50 + "\n")

                # Collect valid labels and predictions for metric calculations
                all_labels.extend(cleaned_gold_labels)
                all_preds.extend(cleaned_pred_labels)

    # Calculate metrics using only valid labels and predictions
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    # Print evaluation metrics
    print('*' * 45)
    print(f'Test Accuracy: {accuracy*100:.2f}%')
    print(f'Test Precision: {precision*100:.2f}%')
    print(f'Test Recall: {recall*100:.2f}%')
    print(f'Test F1 Score: {f1*100:.2f}%')
    print('*' * 45)

    return all_labels, all_preds

# ==============================
# Evaluation on New Dataset
# ==============================

def evaluate_new_set(model, new_dataset_path, scaler):
    """
    Evaluates the model on a new held-out dataset.

    Args:
        model (nn.Module): The trained Transformer model.
        new_dataset_path (str): Path to the new dataset JSON file.
        scaler (StandardScaler): Scaler used for feature normalization.

    Returns:
        tuple: (all_labels, all_preds)
            - all_labels (list): List of true labels.
            - all_preds (list): List of predicted labels.
    """
    # Load new data
    new_data = load_data(new_dataset_path)
    new_dataset = ProsodyDataset(new_data, scaler=scaler)
    new_loader = DataLoader(new_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    # Test the model on the new dataset and get predictions
    print('\n\nEvaluation on Held Out Set Dataset:')
    all_labels, all_preds = test_model(model, new_loader)

    return all_labels, all_preds

# ==============================
# Label Validation
# ==============================

def validate_labels(datasets, num_classes):
    """
    Validates that all labels in the datasets are within the valid range.

    Args:
        datasets (list): List of Dataset instances to validate.
        num_classes (int): Number of output classes.

    Raises:
        ValueError: If any invalid labels are found.

    Prints:
        Confirmation message if all labels are valid.
    """
    for dataset in datasets:
        for idx, (_, _, labels) in enumerate(dataset):
            invalid_mask = (labels >= num_classes) | (labels < 0)
            if torch.any(invalid_mask):
                invalid_labels = labels[invalid_mask].unique().tolist()
                raise ValueError(f"Found invalid labels {invalid_labels} in dataset at index {idx}. Labels should be in the range [0, {num_classes - 1}].")
    print("All labels are valid.")

# ==============================
# Label Extraction and Class Weights
# ==============================

def get_all_unique_labels(datasets):
    """
    Retrieves all unique labels from the given datasets.

    Args:
        datasets (list): List of Dataset instances.

    Returns:
        set: Set of unique labels.
    """
    unique_labels = set()
    for dataset in datasets:
        for _, _, labels in dataset:
            unique_labels.update(labels.numpy().flatten())
    return unique_labels

def get_labels_from_data(data):
    """
    Retrieves sorted unique labels from the dataset.

    Args:
        data (dict): The dataset containing entries with labels.

    Returns:
        list: Sorted list of unique labels.
    """
    unique_labels = set()
    for key, item in data.items():
        labels = item['labels']
        unique_labels.update(labels)
    return sorted(unique_labels)

def compute_class_weights(dataset, num_classes):
    """
    Computes class weights to handle class imbalance.

    Args:
        dataset (Dataset): The dataset to compute class weights from.
        num_classes (int): Number of output classes.

    Returns:
        torch.Tensor: Tensor of class weights.
    """
    all_labels = []
    for _, _, labels in dataset:
        labels_np = labels.numpy().flatten()
        labels_np = labels_np[labels_np != PADDING_VALUE]  # Exclude padding labels
        all_labels.extend(labels_np)
    classes = np.arange(num_classes)
    class_weights = compute_class_weight('balanced', classes=classes, y=all_labels)
    weight_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    return weight_tensor

# ==============================
# Feature Scaling
# ==============================

def fit_feature_scaler(dataset):
    """
    Fits a StandardScaler on the dataset's features for normalization.

    Args:
        dataset (Dataset): The dataset to fit the scaler on.

    Returns:
        StandardScaler: The fitted scaler.
    """
    all_features = []
    for _, features, _ in dataset:
        all_features.append(features.numpy())
    all_features = np.concatenate(all_features, axis=0)
    scaler = StandardScaler()
    scaler.fit(all_features)
    return scaler

# ==============================
# Hyperparameter Optimization with Optuna
# ==============================

def objective(trial):
    """
    Objective function for Optuna hyperparameter optimization.

    Args:
        trial (optuna.trial.Trial): A trial object provided by Optuna.

    Returns:
        float: Validation F1 score to be maximized.
    """
    set_seed(42)

    # Hyperparameters to tune
    HIDDEN_DIM = trial.suggest_int('hidden_dim', 128, 512, step=64)
    NUM_LAYERS = trial.suggest_int('num_layers', 2, 16)
    DROPOUT = trial.suggest_float('dropout', 0.1, 0.4, step=0.05)
    NUM_HEADS = trial.suggest_categorical('num_heads', [4, 8, 16])
    LR = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    WEIGHT_DECAY = trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True)
    

    # Load data
    am_json_path = '../prosody/data/ambiguous_prosody_multi_label_features_train.json'
    un_json_path = '../prosody/data/prosody_multi_label_features_train.json'
    
    am_data = load_data(am_json_path)
    un_data = load_data(un_json_path)

    # Combine the two datasets
    data = {**am_data, **un_data}

    # Split data
    train_data, val_data, test_data = split_data(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
    check_overlap(train_data, val_data, test_data)

    # Fit scaler on training data
    temp_train_dataset = ProsodyDataset(train_data)
    scaler = fit_feature_scaler(temp_train_dataset)

    # Create datasets
    train_dataset = ProsodyDataset(train_data, scaler=scaler)
    val_dataset = ProsodyDataset(val_data, scaler=scaler)
    test_dataset = ProsodyDataset(test_data, scaler=scaler)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # Retrieve a sample batch to determine feature dimensions
    sample_words, sample_features, sample_labels, sample_lengths = next(iter(train_loader))
    feature_dim = sample_features.shape[2]

    # Determine number of classes
    all_unique_labels = get_all_unique_labels([train_dataset, val_dataset, test_dataset])
    NUM_CLASSES = len(all_unique_labels)

    # Define device
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the model
    model = TransformerClassifier(
        feature_dim=feature_dim,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT,
        num_heads=NUM_HEADS
    ).to(device)

    # Validate labels before training
    validate_labels([train_dataset, val_dataset], NUM_CLASSES)

    # Compute class weights
    class_weights = compute_class_weights(train_dataset, NUM_CLASSES)

    # Define optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-6)

    # Define loss function
    criterion = nn.CrossEntropyLoss(ignore_index=PADDING_VALUE, weight=class_weights)

    # Training parameters
    N_EPOCHS = 50

    # Initialize early stopping
    early_stopping = EarlyStopping(patience=5, min_delta=0.0001)
    best_valid_f1 = 0.0

    # Training loop
    for epoch in range(N_EPOCHS):
        train_loss = train_model(model, train_loader, optimizer, criterion, num_classes=NUM_CLASSES)
        valid_loss, valid_acc, valid_precision, valid_recall, valid_f1 = evaluate_model(
            model, val_loader, criterion, num_classes=NUM_CLASSES)

        # Update the learning rate scheduler
        scheduler.step(valid_f1)

        # Save the model if validation F1 score has improved
        if valid_f1 > best_valid_f1:
            best_valid_f1 = valid_f1
            # Save model state dict to a temporary file
            torch.save(model.state_dict(), 'temp_model.pt')

        # Early stopping check
        early_stopping(-valid_f1)  # Since EarlyStopping expects a value to minimize
        if early_stopping.early_stop:
            break

        # Report intermediate metrics to Optuna
        trial.report(valid_f1, epoch)

        # Handle pruning based on the intermediate value
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # Load the best model
    model.load_state_dict(torch.load('temp_model.pt'))

    # Evaluate on validation set
    valid_loss, valid_acc, valid_precision, valid_recall, valid_f1 = evaluate_model(
        model, val_loader, criterion, num_classes=NUM_CLASSES)

    # Remove temporary model file
    os.remove('temp_model.pt')

    # Return validation F1 score directly (since we're maximizing)
    return valid_f1

# ==============================
# Main Script
# ==============================

if __name__ == "__main__":
    """
    Main execution block for training, hyperparameter optimization, and evaluation of the Transformer model.
    """
    # ==============================
    # Configuration and Setup
    # ==============================

    # Set random seed for reproducibility
    set_seed(42)

    # Create Optuna study with direction 'maximize' for F1 score
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)

    # Print best hyperparameters found by Optuna
    print("Best trial:")
    trial = study.best_trial

    print(f"  Validation F1 Score: {trial.value}")
    print("  Best hyperparameters: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    