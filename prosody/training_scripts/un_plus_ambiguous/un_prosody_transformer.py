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
    def __init__(self, patience=10, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        """
        Call method to check if training should stop based on validation loss.

        Args:
            val_loss (float): Current validation loss.
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
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
# def clean_up_sentence(words, gold_labels, pred_labels, padding_value):
#     """
#     Remove padding from words, gold labels, and predicted labels.

#     Args:
#         words (list): A list of words.
#         gold_labels (list): A list of gold labels.
#         pred_labels (list): A list of predicted labels.
#         padding_value (int): The padding value to ignore.

#     Returns:
#         tuple: A tuple containing filtered words, gold labels, and predicted labels.
#     """
#     filtered_words = []
#     filtered_gold_labels = []
#     filtered_pred_labels = []

#     for i in range(len(words)):
#         if gold_labels[i] != padding_value:
#             filtered_words.append(words[i])
#             filtered_gold_labels.append(int(gold_labels[i]))
#             filtered_pred_labels.append(int(pred_labels[i]))
#     return filtered_words, filtered_gold_labels, filtered_pred_labels

def clean_up_sentence(words, gold_labels, pred_labels, padding_value):
    """
    Remove padding from words, gold labels, and predicted labels.

    Args:
        words (list): A list of words.
        gold_labels (array-like): An array or list of gold labels.
        pred_labels (array-like): An array or list of predicted labels.
        padding_value (int): The padding value to ignore.

    Returns:
        tuple: A tuple containing filtered words, gold labels, and predicted labels.
    """
    filtered_words = []
    filtered_gold_labels = []
    filtered_pred_labels = []

    # Use zip to iterate over sequences up to the shortest length
    for word, gold_label, pred_label in zip(words, gold_labels, pred_labels):
        if gold_label != padding_value:
            filtered_words.append(word)
            filtered_gold_labels.append(int(gold_label))
            filtered_pred_labels.append(int(pred_label))
    return filtered_words, filtered_gold_labels, filtered_pred_labels


# Custom dataset class for prosody features
class ProsodyDataset(Dataset):
    """
    Custom Dataset for handling prosody features for sequence labeling.

    Args:
        data (dict): The dataset containing words, features, and labels.
        scaler (StandardScaler, optional): A scaler for normalizing the features.
    """
    def __init__(self, data, scaler=None):
        self.entries = list(data.items())
        self.scaler = scaler

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        key, item = self.entries[idx]
        words = preprocess_text(item['words'])
        features = torch.tensor(item['features'], dtype=torch.float32)
        if self.scaler is not None:
            features_np = features.numpy()
            features_np = self.scaler.transform(features_np)
            features = torch.tensor(features_np, dtype=torch.float32)
        labels = torch.tensor(item['labels'], dtype=torch.long)
        return words, features, labels

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
    features = [item[1] for item in batch]
    labels = [item[2] for item in batch]

    features_padded = torch.nn.utils.rnn.pad_sequence(features, batch_first=True, padding_value=0.0)
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=PADDING_VALUE)
    feature_lengths = torch.tensor([len(f) for f in features])

    return words, features_padded, labels_padded, feature_lengths

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
        x = x + self.pe[:, :x.size(1), :]
        return x

# Transformer Encoder with Padding Masking
class TransformerEncoder(nn.Module):
    """
    Implements a Transformer encoder for sequence modeling.

    Args:
        feature_dim (int): Dimension of the input features.
        hidden_dim (int): Dimension of the hidden state.
        num_layers (int): Number of encoder layers.
        dropout (float): Dropout rate.
        num_heads (int): Number of attention heads.
    """
    def __init__(self, feature_dim, hidden_dim, num_layers, dropout, num_heads=8):
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
        Forward pass for the Transformer encoder.

        Args:
            features (Tensor): Input features of shape [batch_size, seq_len, feature_dim].
            src_key_padding_mask (Tensor, optional): Padding mask.

        Returns:
            Tensor: Output memory of the encoder.
        """
        src = self.input_fc(features)  # [batch_size, seq_len, hidden_dim]
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
        feature_dim (int): Dimension of the input features.
        hidden_dim (int): Dimension of the hidden state.
        num_layers (int): Number of encoder layers.
        num_classes (int): Number of output classes.
        dropout (float): Dropout rate.
        num_heads (int): Number of attention heads.
    """
    def __init__(self, feature_dim, hidden_dim, num_layers, num_classes, dropout, num_heads=8):
        super(TransformerClassifier, self).__init__()
        self.encoder = TransformerEncoder(feature_dim, hidden_dim, num_layers, dropout, num_heads)
        self.fc_out = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, features, lengths):
        """
        Forward pass for the Transformer-based classifier.

        Args:
            features (Tensor): Input features of shape [batch_size, seq_len, feature_dim].
            lengths (Tensor): Lengths of each sequence in the batch.

        Returns:
            Tensor: Output logits of shape [batch_size, seq_len, num_classes].
        """
        max_len = features.size(1)
        src_key_padding_mask = create_src_padding_mask(lengths, max_len)  # Now handled inside the function
        memory = self.encoder(features, src_key_padding_mask=src_key_padding_mask)  # [batch_size, seq_len, hidden_dim]
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
        for batch_idx, (words, features, labels, lengths) in enumerate(iterator):
            features = features.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)

            outputs = model(features, lengths)  # [batch_size, seq_len, num_classes]
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
    with open('./outputs/prosody_transformer_multiclass_results.txt', 'w') as file:
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
                with open('./outputs/prosody_transformer_multiclass_results.txt', 'a') as file:
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
    plt.savefig('./outputs/transformer_multiclass_metrics.png')
    plt.close()

# Evaluate on a new dataset
def evaluate_new_set(model, new_path_un, new_path_am, scaler):
    """
    Evaluate the model on a new dataset.

    Args:
        model (nn.Module): The Transformer model to evaluate.
        new_dataset_path (str): Path to the new dataset JSON file.
        scaler (StandardScaler): Scaler for normalizing the features.

    Returns:
        tuple: A tuple containing all labels and predictions for the new dataset.
    """
    
    # combine the two datasets
    eval_data = {**load_data(new_path_un), **load_data(new_path_am)}
    new_dataset = ProsodyDataset(eval_data, scaler=scaler)
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
        for idx, (_, _, labels) in enumerate(dataset):
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
        for _, _, labels in dataset:
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
    for _, _, labels in dataset:
        labels_np = labels.numpy().flatten()
        labels_np = labels_np[labels_np != PADDING_VALUE]  # Exclude padding labels
        all_labels.extend(labels_np)
    classes = np.arange(num_classes)
    class_weights = compute_class_weight('balanced', classes=classes, y=all_labels)
    weight_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    return weight_tensor

# Function to fit feature scaler
def fit_feature_scaler(dataset):
    """
    Fit a feature scaler on the dataset.

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

# Main script
if __name__ == "__main__":
    set_seed(42)

    am_json_path = '../prosody/data/ambiguous_prosody_multi_label_features_train.json'
    un_json_path = '../prosody/data/prosody_multi_label_features_train.json'
    am_data = load_data(am_json_path)
    un_data = load_data(un_json_path)

    # Combine the two datasets
    data = {**am_data, **un_data}
    

    # Sanity check
    print(f'Total number of Unambiguous entries: {len(un_data)}')
    print(f'Total number of Ambiguous entries: {len(am_data)}')
    print(f'Total number of entries: {len(data)}')

    # Create a descriptive filename for the model
    dataset_name = "un_ambiguous_instructions"
    task_name = "prosody-multiclass"
    best_model_filename = f"models/best-transformer-model-{dataset_name}-{task_name}.pt"

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

    print(f'Train size: {len(train_dataset)}')
    print(f'Validation size: {len(val_dataset)}')
    print(f'Test size: {len(test_dataset)}')

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # Retrieve a sample batch to determine feature dimensions
    sample_words, sample_features, sample_labels, sample_lengths = next(iter(train_loader))
    feature_dim = sample_features.shape[2]

    # Determine number of classes based on all datasets
    all_unique_labels = get_all_unique_labels([train_dataset, val_dataset, test_dataset])
    NUM_CLASSES = len(all_unique_labels)
    print(f"All unique labels across datasets: {sorted(all_unique_labels)}")
    print(f"Model Training with {NUM_CLASSES} classes")

    # Define model hyperparameters
    HIDDEN_DIM = 256
    NUM_LAYERS = 3
    DROPOUT = 0.1
    NUM_HEADS = 16

    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

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
    validate_labels([train_dataset, val_dataset, test_dataset], NUM_CLASSES)

    # Compute class weights
    class_weights = compute_class_weights(train_dataset, NUM_CLASSES)
    print(f"Class Weights: {class_weights}")

    # Define optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), 
                           lr= 0.0004425940999056446, 
                           weight_decay= 7.621907763317249e-06)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)

    # Define loss function
    criterion = nn.CrossEntropyLoss(ignore_index=PADDING_VALUE, ) #weight=class_weights

    # Initialize lists to store metrics
    train_losses = []
    val_losses = []
    val_accuracies = []
    val_precisions = []
    val_recalls = []
    val_f1s = []

    # Define training parameters
    N_EPOCHS = 100
    CLIP = 1

    # Initialize early stopping
    early_stopping = EarlyStopping(patience=15, min_delta=0.0001)
    best_valid_loss = float('inf')

    # Ensure the models directory exists
    os.makedirs('models', exist_ok=True)

    # Training loop
    for epoch in range(N_EPOCHS):
        train_loss = train_model(model, train_loader, optimizer, criterion, num_classes=NUM_CLASSES)
        valid_loss, valid_acc, valid_precision, valid_recall, valid_f1 = evaluate_model(model, val_loader, criterion, num_classes=NUM_CLASSES)

        train_losses.append(train_loss)
        val_losses.append(valid_loss)
        val_accuracies.append(valid_acc)
        val_precisions.append(valid_precision)
        val_recalls.append(valid_recall)
        val_f1s.append(valid_f1)

        # Print the current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch: {epoch+1:02} | Learning Rate: {current_lr:.6f} | '
              f'Train Loss: {train_loss:.4f} | Val. Loss: {valid_loss:.4f} | '
              f'Val. Acc: {valid_acc*100:.2f}% | Precision: {valid_precision:.4f} | '
              f'Recall: {valid_recall:.4f} | F1 Score: {valid_f1:.4f}')

        # Update the learning rate scheduler
        scheduler.step(valid_loss)

        # Save the model if validation loss has decreased
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), best_model_filename)

        # Early stopping check
        early_stopping(valid_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    # Load the best model and evaluate on the test set
    model.load_state_dict(torch.load(best_model_filename))
    test_loss, test_acc, test_precision, test_recall, test_f1 = evaluate_model(model, test_loader, criterion, num_classes=NUM_CLASSES)
    print(f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.2f}% | '
          f'Precision: {test_precision:.4f} | Recall: {test_recall:.4f} | F1 Score: {test_f1:.4f}')

    # Generate detailed test results
    test_model(model, test_loader)

    # Plot training and validation metrics
    plot_metrics(train_losses, val_losses, val_accuracies, val_precisions, val_recalls, val_f1s)

    # Evaluate model on held out set

    am_eval_json = "../prosody/data/ambiguous_prosody_multi_label_features_eval.json"
    un_eval_json = "../prosody/data/prosody_multi_label_features_eval.json"
    
    # Evaluate the model on the new dataset
    true_labels, predicted_labels = evaluate_new_set(model, un_eval_json, am_eval_json, scaler)

    # Log directory
    log_dir = "../prosody/outputs"

    # Load evaluation data to extract class names
    eval_data = load_data(un_eval_json)

    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Class names 
    class_names = get_labels_from_data(eval_data)

    # Compute precision, recall, f1-score, and support for each class
    class_precision, class_recall, class_f1, class_support = precision_recall_fscore_support(
        true_labels, predicted_labels, average=None, zero_division=0)

    print("Class Support:", class_support)
    # Compute the multilabel confusion matrix
    confusion_matrices = multilabel_confusion_matrix(true_labels, predicted_labels)

    # Write class-wise metrics to a file
    with open(f"{log_dir}/classwise_metrics.txt", "w") as f:
        for i, class_name in enumerate(class_names):
            f.write(f"{class_name}:\n")
            f.write(f"  Precision: {class_precision[i]:.4f}\n")
            f.write(f"  Recall: {class_recall[i]:.4f}\n")
            f.write(f"  F1-Score: {class_f1[i]:.4f}\n")
            f.write(f"  Support (True instances in eval data): {class_support[i]}\n")
            f.write("-" * 40 + "\n")

    # Write confusion matrix to a file
    with open(f"{log_dir}/confusion_matrix.txt", "w") as f:
        for i, class_name in enumerate(class_names):
            tn, fp, fn, tp = confusion_matrices[i].ravel()  # Extract the values from the confusion matrix

            f.write(f"\nConfusion Matrix for {class_name}:\n")
            f.write(f"True Negatives (TN): {tn}\n")
            f.write(f"False Positives (FP): {fp}\n")
            f.write(f"False Negatives (FN): {fn}\n")
            f.write(f"True Positives (TP): {tp}\n")

            f.write("\nInterpretation:\n")
            f.write(f"  - The model correctly predicted that '{class_name}' is NOT present {tn} times (TN).\n")
            f.write(f"  - The model incorrectly predicted that '{class_name}' is present {fp} times when it was actually NOT present (FP).\n")
            f.write(f"  - The model incorrectly predicted that '{class_name}' is NOT present {fn} times when it was actually present (FN).\n")
            f.write(f"  - The model correctly predicted that '{class_name}' is present {tp} times (TP).\n")
            f.write("-" * 40 + "\n")
