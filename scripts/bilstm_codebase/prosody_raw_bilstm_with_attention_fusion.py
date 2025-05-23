"""
Prosody Multiclass Classification using BiLSTM with Attention and Word Embeddings

This module implements a sequence-to-sequence model using Bidirectional LSTM (BiLSTM)
with attention mechanisms for multiclass classification on prosody data. It includes
data preprocessing, model definition, training, evaluation, and testing functionalities.
Additionally, it provides functionality to evaluate the model on a held-out dataset.

Dependencies:
    - Python 3.x
    - PyTorch
    - NumPy
    - scikit-learn
    - matplotlib
    - pandas

Author: Kweku Andoh Yamoah
Date: 2025-01-20
"""

import json
import random
import string
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import multilabel_confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import pandas as pd
import re
import torch.nn.utils.rnn as rnn_utils

# ==============================
# Configuration Constants
# ==============================

# Custom padding value for labels
PADDING_VALUE = -100  # Using -100 as it's the default ignore_index in CrossEntropyLoss

# Gradient clipping value
CLIP = 1

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
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def preprocess_text(words):
    """
    Preprocesses a list of words by tokenizing them.

    Args:
        words (list): List of words to preprocess.

    Returns:
        list: Tokenized words.
    """
    processed_words = []
    for word in words:
        # Tokenize words and punctuation
        processed_words.extend(re.findall(r"[\w']+|[.,!?;]", word))
    return processed_words

def load_data(json_path):
    """
    Loads data from a JSON file.

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
    Splits the data into training, validation, and testing sets.

    Args:
        data (dict): The dataset to split.
        train_ratio (float, optional): Proportion of data for training. Defaults to 0.8.
        val_ratio (float, optional): Proportion of data for validation. Defaults to 0.1.
        test_ratio (float, optional): Proportion of data for testing. Defaults to 0.1.

    Returns:
        tuple: Datasets for training, validation, and testing.
    """
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1"
    total = len(data)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    test_size = total - train_size - val_size
    return random_split(list(data.items()), [train_size, val_size, test_size])

def clean_up_sentence(words, gold_labels, pred_labels):
    """
    Cleans up the sentence by removing padding tokens and punctuation.

    Args:
        words (list): List of words in the sentence.
        gold_labels (list): Corresponding gold labels.
        pred_labels (list): Corresponding predicted labels.

    Returns:
        tuple: Cleaned words, gold labels, and predicted labels.
    """
    filtered_words = []
    filtered_gold_labels = []
    filtered_pred_labels = []

    # Remove punctuation from words list
    words = [word for word in words if word not in string.punctuation]
    for i in range(len(words)):
        if gold_labels[i] != PADDING_VALUE:
            filtered_words.append(words[i])
            filtered_gold_labels.append(int(gold_labels[i]))
            filtered_pred_labels.append(int(pred_labels[i]))

    return filtered_words, filtered_gold_labels, filtered_pred_labels

# ==============================
# Early Stopping Class
# ==============================

class EarlyStopping:
    """
    Early stopping to terminate training when validation loss doesn't improve.

    Attributes:
        patience (int): Number of epochs to wait after last improvement.
        min_delta (float): Minimum change to qualify as an improvement.
        best_loss (float): Best validation loss observed.
        counter (int): Counter for epochs without improvement.
        early_stop (bool): Flag indicating whether to stop early.
    """

    def __init__(self, patience=5, min_delta=0):
        """
        Initializes the EarlyStopping instance.

        Args:
            patience (int, optional): Patience for early stopping. Defaults to 5.
            min_delta (float, optional): Minimum delta for improvement. Defaults to 0.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        """
        Call method to update early stopping state based on validation loss.

        Args:
            val_loss (float): Current epoch's validation loss.
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        elif val_loss > self.best_loss + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# ==============================
# Dataset Class
# ==============================

class ProsodyDataset(Dataset):
    """
    Custom Dataset for Prosody Data.

    Each sample consists of tokenized words, associated features, word embeddings, and labels.
    """

    def __init__(self, data, scaler=None):
        """
        Initializes the dataset.

        Args:
            data (dict): The dataset containing entries with words, features, word_embeddings, and labels.
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
            tuple: (key, processed_words, combined_features_tensor, word_embeddings_tensor, labels_tensor)
        """
        key, item = self.entries[idx]
        processed_words = preprocess_text(item['words'])

        # Load prosody features
        prosodic_features = torch.tensor(item['prosodic_features'], dtype=torch.float32)  # Shape: (seq_len, prosody_dim)

        # Load raw acoustic features
        raw_acoustic_features = torch.tensor(item['raw_acoustic_features'], dtype=torch.float32)  # Shape: (seq_len, acoustic_dim)

        # Load word embeddings
        word_embeddings = torch.tensor(item['word_embeddings'], dtype=torch.float32)  # Shape: (seq_len, word_embedding_dim)

        # Concatenate prosodic_features, raw_acoustic_features, and word_embeddings
        combined_features = torch.cat((prosodic_features, raw_acoustic_features, word_embeddings), dim=1)  # Shape: (seq_len, prosody_dim + acoustic_dim + word_embedding_dim)

        if self.scaler is not None:
            combined_features_np = combined_features.numpy()
            combined_features_np = self.scaler.transform(combined_features_np)
            combined_features = torch.tensor(combined_features_np, dtype=torch.float32)

        labels = torch.tensor(item['labels'], dtype=torch.long)  # Changed to torch.long for class indices

        return key, processed_words, combined_features, word_embeddings, labels

# ==============================
# Collate Function
# ==============================

def collate_fn(batch):
    """
    Collate function to handle batches with variable sequence lengths.

    Pads sequences and prepares tensors for words, features, word_embeddings, and labels.

    Args:
        batch (list): List of samples from the dataset.

    Returns:
        tuple: Keys list, words list, padded features, padded word_embeddings, padded labels, and their lengths.
    """
    keys = [item[0] for item in batch]
    words = [item[1] for item in batch]  # List of lists of words
    combined_features = [item[2] for item in batch]
    word_embeddings = [item[3] for item in batch]
    labels = [item[4] for item in batch]

    # Pad features, word_embeddings, and labels
    features_padded = torch.nn.utils.rnn.pad_sequence(combined_features, batch_first=True, padding_value=0.0)
    word_embeddings_padded = torch.nn.utils.rnn.pad_sequence(word_embeddings, batch_first=True, padding_value=0.0)
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=PADDING_VALUE)

    # Record the lengths of each feature sequence
    lengths = torch.tensor([len(f) for f in combined_features])

    return keys, words, features_padded, word_embeddings_padded, labels_padded, lengths

# ==============================
# Attention Mechanisms
# ==============================

class AttentionLayer(nn.Module):
    """
    Attention layer that computes attention weights for encoder outputs.

    Attributes:
        attn (nn.Linear): Linear layer to compute energy.
        v (nn.Linear): Linear layer to project energy to attention scores.
    """

    def __init__(self, hidden_dim):
        """
        Initializes the AttentionLayer.

        Args:
            hidden_dim (int): Dimension of the hidden state.
        """
        super(AttentionLayer, self).__init__()
        self.attn = nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs, mask):
        """
        Forward pass to compute attention weights.

        Args:
            hidden (torch.Tensor): Current hidden state of the decoder (batch_size, hidden_dim).
            encoder_outputs (torch.Tensor): Outputs from the encoder (batch_size, seq_len, hidden_dim*2).
            mask (torch.Tensor): Mask to ignore padding tokens (batch_size, seq_len).

        Returns:
            torch.Tensor: Attention weights (batch_size, seq_len).
        """
        src_len = encoder_outputs.size(1)
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)  # Repeat hidden state across seq_len
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))  # (batch_size, seq_len, hidden_dim)
        attention = self.v(energy).squeeze(2)  # (batch_size, seq_len)

        # Apply the mask to ignore padding positions
        attention.masked_fill_(~mask, -1e10)

        return torch.softmax(attention, dim=1)  # (batch_size, seq_len)

class MultiAttention(nn.Module):
    """
    Multi-head attention mechanism using multiple AttentionLayers.

    Attributes:
        attention_layers (nn.ModuleList): List of AttentionLayer instances.
    """

    def __init__(self, hidden_dim, num_layers):
        """
        Initializes the MultiAttention.

        Args:
            hidden_dim (int): Dimension of the hidden state.
            num_layers (int): Number of attention layers.
        """
        super(MultiAttention, self).__init__()
        self.attention_layers = nn.ModuleList([AttentionLayer(hidden_dim) for _ in range(num_layers)])

    def forward(self, hidden, encoder_outputs, mask):
        """
        Forward pass to compute averaged attention weights from multiple layers.

        Args:
            hidden (torch.Tensor): Current hidden state of the decoder (batch_size, hidden_dim).
            encoder_outputs (torch.Tensor): Outputs from the encoder (batch_size, seq_len, hidden_dim*2).
            mask (torch.Tensor): Mask to ignore padding tokens (batch_size, seq_len).

        Returns:
            torch.Tensor: Averaged attention weights (batch_size, seq_len).
        """
        attn_weights = []
        for layer in self.attention_layers:
            attn_weight = layer(hidden, encoder_outputs, mask)
            attn_weights.append(attn_weight)
        attn_weights = torch.stack(attn_weights, dim=1).mean(dim=1)  # Average over attention layers
        return attn_weights  # (batch_size, seq_len)

# ==============================
# Encoder and Decoder Models
# ==============================

class ProjectedEncoder(nn.Module):
    """
    Encoder with feature projection before BiLSTM.
    """
    def __init__(self, combined_dim, projected_dim, hidden_dim, num_layers, dropout):
        super(ProjectedEncoder, self).__init__()
        self.proj = nn.Linear(combined_dim, projected_dim)
        self.lstm = nn.LSTM(projected_dim, hidden_dim, num_layers, 
                            dropout=dropout, batch_first=True, bidirectional=True)

    def forward(self, features, lengths):
        """
        Forward pass of the Projected Encoder.

        Args:
            features (torch.Tensor): Batch of combined features (batch_size, seq_len, combined_dim).
            lengths (torch.Tensor): Lengths of each sequence in the batch (batch_size).

        Returns:
            tuple: Encoder outputs, hidden states, and cell states.
        """
        # Project features
        projected = torch.relu(self.proj(features))  # (batch, seq_len, projected_dim)

        # Sort sequences by lengths in descending order for packing
        lengths_sorted, sorted_indices = lengths.sort(descending=True)
        projected = projected[sorted_indices]

        # Pack the sequences for efficient processing
        packed_input = rnn_utils.pack_padded_sequence(projected, lengths_sorted.cpu(), 
                                                      batch_first=True, enforce_sorted=True)

        # Encode
        packed_output, (hidden, cell) = self.lstm(packed_input)

        # Unpack the sequences
        outputs, _ = rnn_utils.pad_packed_sequence(packed_output, batch_first=True, padding_value=0.0)

        # Restore the original ordering
        _, original_indices = sorted_indices.sort()
        outputs = outputs[original_indices]
        hidden = hidden[:, original_indices]
        cell = cell[:, original_indices]

        return outputs, hidden, cell  # outputs: (batch_size, seq_len, hidden_dim*2)

class Decoder(nn.Module):
    """
    Decoder module with attention mechanism for sequence-to-sequence modeling.

    Attributes:
        lstm (nn.LSTM): Bidirectional LSTM layer.
        fc (nn.Linear): Fully connected layer for output predictions.
        attention (MultiAttention): Multi-head attention mechanism.
        word_proj (nn.Linear): Linear layer to project word embeddings.
    """

    def __init__(self, hidden_dim, output_dim, num_layers, dropout, num_attention_layers, word_embedding_dim=300):
        """
        Initializes the Decoder.

        Args:
            hidden_dim (int): Dimension of LSTM hidden states.
            output_dim (int): Number of output classes.
            num_layers (int): Number of LSTM layers.
            dropout (float): Dropout rate.
            num_attention_layers (int): Number of attention layers.
            word_embedding_dim (int, optional): Dimension of word embeddings. Defaults to 300.
        """
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_proj = nn.Linear(word_embedding_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim * 4 + hidden_dim, hidden_dim, num_layers, 
                            dropout=dropout, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.attention = MultiAttention(hidden_dim, num_attention_layers)

    def forward(self, encoder_outputs, word_embeddings, hidden, cell, mask):
        """
        Forward pass of the Decoder.

        Args:
            encoder_outputs (torch.Tensor): Outputs from the encoder (batch_size, seq_len, hidden_dim*2).
            word_embeddings (torch.Tensor): Word embeddings (batch_size, seq_len, word_embedding_dim).
            hidden (torch.Tensor): Hidden state from the encoder (num_layers*2, batch_size, hidden_dim).
            cell (torch.Tensor): Cell state from the encoder (num_layers*2, batch_size, hidden_dim).
            mask (torch.Tensor): Mask to ignore padding tokens (batch_size, seq_len).

        Returns:
            tuple: Predictions and updated hidden and cell states.
        """
        # Project word embeddings
        word_proj = torch.relu(self.word_proj(word_embeddings))  # (batch, seq_len, hidden_dim)

        # Compute attention weights using the last layer's hidden state
        attn_weights = self.attention(hidden[-1], encoder_outputs, mask)  # (batch_size, seq_len)

        # Compute context vector as weighted sum of encoder outputs
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)  # (batch_size, hidden_dim*2)

        # Prepare LSTM input by concatenating context with encoder outputs and projected word embeddings
        lstm_input = torch.cat(
            (context.unsqueeze(1).repeat(1, encoder_outputs.size(1), 1), 
             encoder_outputs,
             word_proj), dim=2
        )  # (batch_size, seq_len, hidden_dim*4 + hidden_dim)

        # Pass through LSTM
        outputs, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))  # outputs: (batch_size, seq_len, hidden_dim*2)

        # Generate predictions
        predictions = self.fc(outputs)  # (batch_size, seq_len, output_dim)

        return predictions, (hidden, cell)

class AttentionBasedFusion(nn.Module):
    """
    Attention layer to fuse combined features.
    """
    def __init__(self, combined_dim, hidden_dim):
        super(AttentionBasedFusion, self).__init__()
        self.attn = nn.Linear(combined_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, combined_features, mask):
        """
        Args:
            combined_features: (batch, seq_len, combined_dim)
            mask: (batch, seq_len)

        Returns:
            fused_features: (batch, seq_len, combined_dim)
        """
        energy = torch.tanh(self.attn(combined_features))  # (batch, seq_len, hidden_dim)
        attention = self.v(energy).squeeze(2)  # (batch, seq_len)
        attn_weights = torch.softmax(attention.masked_fill(~mask, -1e10), dim=1).unsqueeze(2)  # (batch, seq_len, 1)
        fused = combined_features * attn_weights  # (batch, seq_len, combined_dim)
        return fused  # (batch, seq_len, combined_dim)

class Seq2Seq(nn.Module):
    """
    Sequence-to-Sequence model combining Encoder, Attention-Based Fusion, and Decoder.

    Attributes:
        encoder (ProjectedEncoder): The encoder component.
        fusion (AttentionBasedFusion): Attention-based fusion mechanism.
        decoder (Decoder): The decoder component.
    """

    def __init__(self, encoder, fusion, decoder):
        """
        Initializes the Seq2Seq model.

        Args:
            encoder (ProjectedEncoder): Encoder instance.
            fusion (AttentionBasedFusion): Fusion instance.
            decoder (Decoder): Decoder instance.
        """
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.fusion = fusion
        self.decoder = decoder

    def forward(self, features, word_embeddings, lengths):
        """
        Forward pass of the Seq2Seq model.

        Args:
            features (torch.Tensor): Batch of combined features (batch_size, seq_len, combined_dim).
            word_embeddings (torch.Tensor): Batch of word embeddings (batch_size, seq_len, word_embedding_dim).
            lengths (torch.Tensor): Lengths of each sequence in the batch (batch_size).

        Returns:
            torch.Tensor: Output predictions (batch_size, seq_len, output_dim).
        """
        # Create mask based on lengths to ignore padding
        batch_size, seq_len, _ = features.size()
        mask = torch.arange(seq_len).unsqueeze(0).expand(batch_size, seq_len).to(features.device)
        mask = mask < lengths.unsqueeze(1)  # (batch_size, seq_len)

        # Apply attention-based fusion
        fused_features = self.fusion(features, mask)  # (batch, seq_len, combined_dim)

        # Encode the fused features
        encoder_outputs, hidden, cell = self.encoder(fused_features, lengths)

        # Decode the encoder outputs with word embeddings
        predictions, (hidden, cell) = self.decoder(encoder_outputs, word_embeddings, hidden, cell, mask)

        return predictions  # (batch_size, seq_len, output_dim)

# ==============================
# Training and Evaluation Functions
# ==============================

def train_model(model, iterator, optimizer, criterion, device, num_classes):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The Seq2Seq model.
        iterator (DataLoader): DataLoader for training data.
        optimizer (torch.optim.Optimizer): Optimizer.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to run the model on.
        num_classes (int): Number of output classes.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    epoch_loss = 0
    for keys, words, features, word_embeddings, labels, lengths in iterator:
        features = features.to(device)
        word_embeddings = word_embeddings.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)

        optimizer.zero_grad()
        output = model(features, word_embeddings, lengths)  # (batch_size, seq_len, num_classes)
        
        # Flatten outputs and labels for loss computation
        output = output.view(-1, num_classes)  # (batch_size * seq_len, num_classes)
        labels = labels.view(-1)  # (batch_size * seq_len)

        # Mask padding positions
        mask = labels != PADDING_VALUE
        masked_output = output[mask]
        masked_labels = labels[mask]

        # Compute loss
        loss = criterion(masked_output, masked_labels)
        loss.backward()

        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def evaluate_model(model, iterator, criterion, device, num_classes):
    """
    Evaluates the model on validation or test data.

    Args:
        model (nn.Module): The Seq2Seq model.
        iterator (DataLoader): DataLoader for validation or test data.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to run the model on.
        num_classes (int): Number of output classes.

    Returns:
        tuple: Average loss, accuracy, precision, recall, and F1 score.
    """
    model.eval()
    epoch_loss = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for keys, words, features, word_embeddings, labels, lengths in iterator:
            features = features.to(device)
            word_embeddings = word_embeddings.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)

            output = model(features, word_embeddings, lengths)  # (batch_size, seq_len, num_classes)
            
            # Flatten outputs and labels
            output = output.view(-1, num_classes)  # (batch_size * seq_len, num_classes)
            labels = labels.view(-1)  # (batch_size * seq_len)

            # Mask padding positions
            mask = labels != PADDING_VALUE
            masked_output = output[mask]
            masked_labels = labels[mask]

            # Compute loss
            loss = criterion(masked_output, masked_labels)
            epoch_loss += loss.item()

            # Predictions
            preds = torch.argmax(masked_output, dim=1)  # (num_valid_tokens)

            # Collect labels and predictions
            all_labels.extend(masked_labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # Compute evaluation metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    return epoch_loss / len(iterator), accuracy, precision, recall, f1

def test_model(model, iterator, device, num_classes):
    """
    Tests the model on the test dataset and saves detailed results.

    Args:
        model (nn.Module): The trained Seq2Seq model.
        iterator (DataLoader): DataLoader for test data.
        device (torch.device): Device to run the model on.
        num_classes (int): Number of output classes.

    Returns:
        tuple: All true labels and all predicted labels.
    """
    model.eval()
    all_labels = []
    all_preds = []

    # Initialize results file
    results_filepath = '../outputs/prosody_raw_acoustic_results.txt'
    os.makedirs('../outputs', exist_ok=True)
    with open(results_filepath, 'w') as file:
        file.write("")

    with torch.no_grad():
        for keys, words, features, word_embeddings, labels, lengths in iterator:
            features = features.to(device)
            word_embeddings = word_embeddings.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)

            output = model(features, word_embeddings, lengths)  # (batch_size, seq_len, num_classes)
            preds = torch.argmax(output, dim=2)  # (batch_size, seq_len)

            for i in range(features.shape[0]):
                key = keys[i]
                word_sentence = words[i]  # List of words
                gold_labels = labels[i].cpu().numpy().flatten()
                pred_labels = preds[i].cpu().numpy().flatten()

                # Clean up the sentence by excluding padding positions
                cleaned_words, cleaned_gold_labels, cleaned_pred_labels = clean_up_sentence(
                    word_sentence, gold_labels, pred_labels
                )

                # Create DataFrame
                data = {
                    'Word': cleaned_words,
                    'Gold Label': cleaned_gold_labels,
                    'Predicted Label': cleaned_pred_labels
                }

                df = pd.DataFrame(data)
                with open(results_filepath, 'a') as file:
                    file.write(f"Audio File: {key}\n")
                    file.write(df.to_string(index=False))
                    file.write("\n" + "-" * 50 + "\n")

                # Collect labels and predictions for metrics
                all_labels.extend(cleaned_gold_labels)
                all_preds.extend(cleaned_pred_labels)

                # Collect data for JSON
                data_json = {
                    'audio_file': key,
                    'words': cleaned_words,
                    'gold_labels': cleaned_gold_labels,
                    'predicted_labels': cleaned_pred_labels
                }

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

# ==============================
# Plotting Function
# ==============================

def plot_metrics(train_losses, val_losses, val_accuracies, val_precisions, val_recalls, val_f1s):
    """
    Plots training and validation metrics over epochs.

    Args:
        train_losses (list): List of training losses per epoch.
        val_losses (list): List of validation losses per epoch.
        val_accuracies (list): List of validation accuracies per epoch.
        val_precisions (list): List of validation precisions per epoch.
        val_recalls (list): List of validation recalls per epoch.
        val_f1s (list): List of validation F1 scores per epoch.
    """
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(18, 10))
    
    # Plot Losses
    plt.subplot(2, 3, 1)
    plt.plot(epochs, train_losses, label='Train Loss', color='blue')
    plt.plot(epochs, val_losses, label='Validation Loss', color='orange')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Accuracy
    plt.subplot(2, 3, 2)
    plt.plot(epochs, val_accuracies, label='Validation Accuracy', color='green')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot Precision
    plt.subplot(2, 3, 3)
    plt.plot(epochs, val_precisions, label='Validation Precision', color='red')
    plt.title('Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.legend()

    # Plot Recall
    plt.subplot(2, 3, 4)
    plt.plot(epochs, val_recalls, label='Validation Recall', color='purple')
    plt.title('Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.legend()

    # Plot F1 Score
    plt.subplot(2, 3, 5)
    plt.plot(epochs, val_f1s, label='Validation F1 Score', color='orange')
    plt.title('F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()

    plt.tight_layout()
    metrics_filepath = '../outputs/prosody_raw_acoustic_metrics.png'
    plt.savefig(metrics_filepath)
    plt.close()
    print(f'Metrics plot saved to {metrics_filepath}')

# ==============================
# Additional Evaluation Function
# ==============================

def evaluate_new_set(model, new_dataset_path, scaler=None, device='cpu', num_classes=3):
    """
    Evaluates the model on a new held-out dataset.

    Args:
        model (nn.Module): The trained Seq2Seq model.
        new_dataset_path (str): Path to the new JSON dataset.
        scaler (StandardScaler, optional): Scaler for feature normalization. Defaults to None.
        device (str, optional): Device to run the model on. Defaults to 'cpu'.
        num_classes (int, optional): Number of output classes. Defaults to 3.

    Returns:
        tuple: All true labels and all predicted labels.
    """
    # Load new data
    new_data = load_data(new_dataset_path)
    if scaler is not None:
        new_dataset = ProsodyDataset(new_data, scaler=scaler)
    else:
        new_dataset = ProsodyDataset(new_data)
    new_loader = DataLoader(new_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
    
    # Test the model on the new dataset and get predictions
    print('\n\nEvaluation on Held Out Set Dataset:')
    all_labels, all_preds = test_model(model, new_loader, device, num_classes)

    return all_labels, all_preds

def compute_class_weights(dataset, num_classes, device):
    """
    Compute class weights for handling label imbalance.

    Args:
        dataset (Dataset): The dataset to compute class weights from.
        num_classes (int): The number of output classes.
        device (torch.device): Device to run the model on.

    Returns:
        Tensor: A tensor containing the class weights.
    """
    all_labels = []
    for _, _, _, _, labels in dataset:
        labels_np = labels.numpy().flatten()
        labels_np = labels_np[labels_np != PADDING_VALUE]  # Exclude padding labels
        all_labels.extend(labels_np)
    classes = np.arange(num_classes)
    class_weights = compute_class_weight('balanced', classes=classes, y=all_labels)
    weight_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    return weight_tensor

# ==============================
# Main Execution
# ==============================

def main():
    # ==============================
    # Configuration and Setup
    # ==============================

    seed = 42
    set_seed(seed)  # Set seed for reproducibility

    # Path to training data
    json_path = '../../prosody/data/ambiguous_prosodic_raw_acoustic_ml_features_train_embeddings.json'
    data = load_data(json_path)  # Load data

    # Create a descriptive filename for the model
    dataset_name = "ambiguous_instructions"
    task_name = "prosody_raw_multiclass"
    best_model_filename = f"../models/best-model-{dataset_name}-{task_name}.pt"
    os.makedirs('../models', exist_ok=True)

    # Split data into train, validation, and test sets
    train_data, val_data, test_data = split_data(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)

    # Initialize datasets
    train_dataset = ProsodyDataset(dict(train_data))
    val_dataset = ProsodyDataset(dict(val_data))
    test_dataset = ProsodyDataset(dict(test_data))

    print(f'Train size: {len(train_dataset)}')
    print(f'Validation size: {len(val_dataset)}')
    print(f'Test size: {len(test_dataset)}')

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # Get feature dimension from the dataset
    sample_batch = next(iter(train_loader))
    _, _, sample_features, sample_word_embeddings, sample_labels, sample_lengths = sample_batch
    feature_dim = sample_features.shape[2]  # Combined feature dim (prosody + acoustic + word embeddings)
    word_embedding_dim = sample_word_embeddings.shape[2]  # Word embedding dimension (e.g., 300)

    # Determine number of classes dynamically
    all_labels = []
    for _, _, _, _, labels in train_dataset:
        labels_np = labels.numpy().flatten()
        labels_np = labels_np[labels_np != PADDING_VALUE]  # Exclude padding labels
        all_labels.extend(labels_np)
    num_classes = len(np.unique(all_labels))

    print(f'Model Training with {num_classes} classes, they are {np.unique(all_labels)}')

    # ==============================
    # Model Configuration
    # ==============================

    # Hyperparameters (can be tuned)
    PROJECTED_DIM = 512
    HIDDEN_DIM = 512
    OUTPUT_DIM = num_classes  # Updated to num_classes
    NUM_LAYERS = 1
    DROPOUT = 0.45  # Example dropout value
    NUM_ATTENTION_LAYERS = 1

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Initialize Encoder, Fusion, and Decoder
    encoder = ProjectedEncoder(combined_dim=feature_dim, projected_dim=PROJECTED_DIM, hidden_dim=HIDDEN_DIM, 
                               num_layers=NUM_LAYERS, dropout=DROPOUT).to(device)
    fusion = AttentionBasedFusion(combined_dim=feature_dim, hidden_dim=HIDDEN_DIM).to(device)
    decoder = Decoder(hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM, num_layers=NUM_LAYERS, 
                      dropout=DROPOUT, num_attention_layers=NUM_ATTENTION_LAYERS, 
                      word_embedding_dim=word_embedding_dim).to(device)
    model = Seq2Seq(encoder, fusion, decoder).to(device)

    # Print model summary
    summary(model, input_data=(sample_features.to(device), sample_word_embeddings.to(device), sample_lengths.to(device)), device=device)

    # Compute class weights
    class_weights = compute_class_weights(train_dataset, num_classes, device)
    print(f"Class Weights: {class_weights}")

    # Initialize optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), 
                           lr=0.004265807088782683, 
                           weight_decay=5.440811049353001e-05)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    # Alternative schedulers:
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    # Define loss function
    criterion = nn.CrossEntropyLoss(ignore_index=PADDING_VALUE, weight=class_weights)

    # Initialize lists to store metrics
    train_losses = []
    val_losses = []
    val_accuracies = []
    val_precisions = []
    val_recalls = []
    val_f1s = []

    # Training parameters
    N_EPOCHS = 100

    # Initialize early stopping
    early_stopping = EarlyStopping(patience=6, min_delta=0.0034379937547662745)
    best_valid_loss = float('inf')

    # ==============================
    # Training Loop
    # ==============================

    for epoch in range(N_EPOCHS):
        train_loss = train_model(model, train_loader, optimizer, criterion, device, num_classes)
        valid_loss, valid_acc, valid_precision, valid_recall, valid_f1 = evaluate_model(model, val_loader, criterion, device, num_classes)

        # Append metrics
        train_losses.append(train_loss)
        val_losses.append(valid_loss)
        val_accuracies.append(valid_acc)
        val_precisions.append(valid_precision)
        val_recalls.append(valid_recall)
        val_f1s.append(valid_f1)

        # Update the learning rate scheduler
        scheduler.step(valid_loss)

        # Save the model if validation loss improves
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), best_model_filename)
            print(f'\tBest model saved with validation loss: {best_valid_loss:.3f}')

        # Print epoch statistics
        print(f'Epoch: {epoch+1:02}')
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}% |  Precision: {valid_precision:.2f} |  Recall: {valid_recall:.2f} |  F1 Score: {valid_f1:.2f}')

        # Check for early stopping
        early_stopping(valid_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    # ==============================
    # Load Best Model and Test
    # ==============================

    # Load the best model
    model.load_state_dict(torch.load(best_model_filename))
    test_labels, test_preds = test_model(model, test_loader, device, num_classes)

    # ==============================
    # Plotting Metrics
    # ==============================

    plot_metrics(train_losses, val_losses, val_accuracies, val_precisions, val_recalls, val_f1s)

    # ==============================
    # Evaluation on Held-Out Set
    # ==============================

    # Path to held-out evaluation data
    eval_json = "../../prosody/data/ambiguous_prosodic_raw_acoustic_ml_features_eval_embeddings.json"

    # Evaluate the model on the new dataset
    true_labels, predicted_labels = evaluate_new_set(model, eval_json, scaler=None, device=device, num_classes=num_classes)

    # ==============================
    # Logging Metrics
    # ==============================

    # Log directory
    log_dir = "../../prosody/outputs"
    
    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Class names 
    class_names = [0,1,2] #[str(cls) for cls in range(num_classes)]  # Automatically generate class names based on num_classes

    # Compute per-class precision, recall, f1-score, and support
    class_precision, class_recall, class_f1, class_support = precision_recall_fscore_support(
        true_labels, predicted_labels, average=None, zero_division=0)
    
    # Compute weighted precision, recall, f1-score
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        true_labels, predicted_labels, average='weighted', zero_division=0)
    
    print("Class Support:", class_support)
    
    # Compute the multilabel confusion matrix
    confusion_matrices = multilabel_confusion_matrix(true_labels, predicted_labels)
    
    # Initialize a list to store accuracy for each class
    class_accuracy = []
    
    # Calculate accuracy for each class using the confusion matrices
    for i in range(len(class_names)):
        tn, fp, fn, tp = confusion_matrices[i].ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        class_accuracy.append(accuracy)
    
    # Compute weighted accuracy
    total_support = sum(class_support)
    weighted_accuracy = sum(acc * supp for acc, supp in zip(class_accuracy, class_support)) / total_support if total_support > 0 else 0.0
    
    # Write class-wise metrics to a file, including accuracy
    with open(f"{log_dir}/bilstm_prosody_raw_classwise_metrics.txt", "w") as f:
        for i, class_name in enumerate(class_names):
            f.write(f"{class_name}:\n")
            f.write(f"  Precision: {class_precision[i]:.4f}\n")
            f.write(f"  Recall: {class_recall[i]:.4f}\n")
            f.write(f"  F1-Score: {class_f1[i]:.4f}\n")
            f.write(f"  Accuracy: {class_accuracy[i]:.4f}\n")  # Added accuracy
            f.write(f"  Support (True instances in eval data): {class_support[i]}\n")
            f.write("-" * 40 + "\n")
        
        # Write weighted metrics
        f.write("\nWeighted Metrics:\n")
        f.write(f"  Weighted Precision: {weighted_precision:.4f}\n")
        f.write(f"  Weighted Recall: {weighted_recall:.4f}\n")
        f.write(f"  Weighted F1-Score: {weighted_f1:.4f}\n")
        f.write(f"  Weighted Accuracy: {weighted_accuracy:.4f}\n")
        f.write("-" * 40 + "\n")
    
    # Write confusion matrix to a file
    with open(f"{log_dir}/bilstm_prosody_raw_confusion_matrix.txt", "w") as f:
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
    
    print("Training and evaluation completed successfully.")

if __name__ == "__main__":
    main()