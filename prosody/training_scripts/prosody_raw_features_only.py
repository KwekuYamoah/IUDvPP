"""
Prosody Classification using BiLSTM with Attention and Multi-Feature Integration

This script implements a sequence-to-sequence model using Bidirectional LSTM (BiLSTM)
with attention mechanisms for multiclass classification on prosody data. It integrates
both prosodic and raw acoustic features, includes data preprocessing, model definition,
training, evaluation, testing, and visualization of training metrics.

Dependencies:
    - Python 3.x
    - PyTorch
    - NumPy
    - scikit-learn
    - matplotlib
    - pandas
    - torchinfo

Author: Your Name
Date: 2024-04-27
"""

import json
import random
import torch
import string
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import pandas as pd
import re
import torch.nn.utils.rnn as rnn_utils

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
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU setups
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_corpus(data):
    """
    Extract and tokenize words from the dataset to build a corpus.

    Args:
        data (dict): The dataset containing entries with words.

    Returns:
        list: A list of tokenized words for each entry.
    """
    corpus = []
    for entry in data.values():
        words = []
        for word in entry['words']:
            # Tokenize words and punctuation
            words.extend(re.findall(r"[\w']+|[.,!?;]", word))
        corpus.append(words)
    return corpus

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

def split_data(data, train_ratio=0.8, val_ratio=0.0, test_ratio=0.2):
    """
    Split the data into training, validation, and testing sets.

    Args:
        data (dict): The dataset to split.
        train_ratio (float, optional): Proportion of data for training. Defaults to 0.8.
        val_ratio (float, optional): Proportion of data for validation. Defaults to 0.0.
        test_ratio (float, optional): Proportion of data for testing. Defaults to 0.2.

    Returns:
        tuple: Datasets for training, validation, and testing.
    """
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1"
    total = len(data)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    test_size = total - train_size - val_size
    return random_split(list(data.items()), [train_size, val_size, test_size])

def preprocess_text(words):
    """
    Preprocess a list of words by tokenizing them.

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

def clean_up_sentence(words, gold_labels, pred_labels):
    """
    Clean up the sentence by removing padding and punctuation tokens.

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
    def __init__(self, patience=10, min_delta=0.001):
        """
        Initializes the EarlyStopping instance.

        Args:
            patience (int, optional): Patience for early stopping. Defaults to 10.
            min_delta (float, optional): Minimum delta for improvement. Defaults to 0.001.
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

    Each sample consists of processed words, prosodic features, raw acoustic features, and labels.
    """
    def __init__(self, data):
        """
        Initializes the ProsodyDataset.

        Args:
            data (dict): Dictionary containing prosody data with sentence keys.
        """
        self.entries = list(data.items())

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
            tuple: (processed_words, prosodic_features, raw_acoustic_features, labels)
                - processed_words (list): List of tokenized words.
                - prosodic_features (torch.Tensor): Tensor of prosodic features.
                - raw_acoustic_features (torch.Tensor): Tensor of raw acoustic features.
                - labels (torch.Tensor): Tensor of labels for each word.
        """
        key, item = self.entries[idx]
        processed_words = preprocess_text(item['words'])
        # Extract both prosodic and raw acoustic features
        prosodic_features = torch.tensor(item['prosodic_features'], dtype=torch.float32)
        raw_acoustic_features = torch.tensor(item['raw_acoustic_features'], dtype=torch.float32)
        labels = torch.tensor(item['labels'], dtype=torch.long)
        return processed_words, prosodic_features, raw_acoustic_features, labels

# ==============================
# Collate Function
# ==============================

def collate_fn(batch):
    """
    Collate function to handle batches with variable sequence lengths.

    Pads sequences and prepares tensors for words, prosodic features, raw acoustic features, and labels.

    Args:
        batch (list): List of samples from the dataset.

    Returns:
        tuple: (words, prosodic_features_padded, raw_acoustic_features_padded, labels_padded, lengths)
            - words (list): List of word lists.
            - prosodic_features_padded (torch.Tensor): Padded prosodic features (batch_size, max_seq_len, feature_dim).
            - raw_acoustic_features_padded (torch.Tensor): Padded raw acoustic features (batch_size, max_seq_len, feature_dim).
            - labels_padded (torch.Tensor): Padded labels (batch_size, max_seq_len).
            - lengths (torch.Tensor): Original lengths of each sequence (batch_size).
    """
    words = [item[0] for item in batch]  # List of lists of words
    prosodic_features = [item[1] for item in batch]
    raw_acoustic_features = [item[2] for item in batch]
    labels = [item[3] for item in batch]

    # Pad prosodic features with 0.0
    prosodic_features_padded = torch.nn.utils.rnn.pad_sequence(
        prosodic_features, batch_first=True, padding_value=0.0
    )
    # Pad raw acoustic features with 0.0
    raw_acoustic_features_padded = torch.nn.utils.rnn.pad_sequence(
        raw_acoustic_features, batch_first=True, padding_value=0.0
    )
    # Pad labels with PADDING_VALUE (-1)
    labels_padded = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=PADDING_VALUE
    )

    # Record the lengths of each sequence (assuming both prosodic and raw acoustic features have the same lengths)
    lengths = torch.tensor([len(f) for f in prosodic_features])

    return words, prosodic_features_padded, raw_acoustic_features_padded, labels_padded, lengths

# ==============================
# Feature Projection Layer
# ==============================

class FeatureProjection(nn.Module):
    """
    Feature projection layer to reduce feature dimensions.

    Applies a linear transformation followed by a ReLU activation.
    """
    def __init__(self, input_dim, output_dim):
        """
        Initializes the FeatureProjection layer.

        Args:
            input_dim (int): Dimension of input features.
            output_dim (int): Dimension after projection.
        """
        super(FeatureProjection, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass of the FeatureProjection layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor after linear layer and ReLU activation.
        """
        return self.relu(self.linear(x))

# ==============================
# Attention Mechanisms
# ==============================

class AttentionLayer(nn.Module):
    """
    Attention layer that computes attention weights for encoder outputs.

    Attributes:
        attn (nn.Linear): Linear layer to compute energy from hidden states and encoder outputs.
        v (nn.Linear): Linear layer to project energy to attention scores.
    """
    def __init__(self, hidden_dim):
        """
        Initializes the AttentionLayer.

        Args:
            hidden_dim (int): Dimension of the hidden state.
        """
        super(AttentionLayer, self).__init__()
        # Updated input dimension from hidden_dim * 4 to hidden_dim * 4
        self.attn = nn.Linear(hidden_dim * 4, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs, mask):
        """
        Forward pass to compute attention weights.

        Args:
            hidden (torch.Tensor): Concatenated hidden state from the encoder (batch_size, hidden_dim * 2).
            encoder_outputs (torch.Tensor): Outputs from the encoder (batch_size, seq_len, hidden_dim * 2).
            mask (torch.Tensor): Mask to ignore padding tokens (batch_size, seq_len).

        Returns:
            torch.Tensor: Attention weights (batch_size, seq_len).
        """
        src_len = encoder_outputs.size(1)

        # Concatenate the last forward and backward hidden states
        hidden_forward = hidden[-2]  # Last layer forward hidden state
        hidden_backward = hidden[-1]  # Last layer backward hidden state
        hidden = torch.cat((hidden_forward, hidden_backward), dim=1)  # Shape: [batch_size, hidden_dim * 2]

        # Repeat hidden state across the source sequence length
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)  # Shape: [batch_size, src_len, hidden_dim * 2]

        # Concatenate hidden and encoder outputs
        concatenated = torch.cat((hidden, encoder_outputs), dim=2)  # Shape: [batch_size, src_len, hidden_dim * 4]

        # Apply attention mechanism
        energy = torch.tanh(self.attn(concatenated))  # Shape: [batch_size, src_len, hidden_dim]
        attention = self.v(energy).squeeze(2)  # Shape: [batch_size, src_len]

        # Apply the mask to ignore padding positions by setting them to a very negative value
        attention.masked_fill_(~mask, -1e10)

        # Apply softmax to obtain attention weights
        return torch.softmax(attention, dim=1)  # Shape: [batch_size, src_len]

class MultiAttention(nn.Module):
    """
    Multi-head attention mechanism using multiple AttentionLayer instances.

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
            hidden (torch.Tensor): Concatenated hidden state from the encoder (batch_size, hidden_dim * 2).
            encoder_outputs (torch.Tensor): Outputs from the encoder (batch_size, seq_len, hidden_dim * 2).
            mask (torch.Tensor): Mask to ignore padding tokens (batch_size, seq_len).

        Returns:
            torch.Tensor: Averaged attention weights (batch_size, seq_len).
        """
        attn_weights = []
        for layer in self.attention_layers:
            attn_weight = layer(hidden, encoder_outputs, mask)
            attn_weights.append(attn_weight)
        # Stack attention weights from all layers and compute the mean
        attn_weights = torch.stack(attn_weights, dim=1).mean(dim=1)  # Shape: [batch_size, seq_len]
        return attn_weights

# ==============================
# Encoder Model
# ==============================

class Encoder(nn.Module):
    """
    Encoder module using a Bidirectional LSTM.

    Processes concatenated prosodic and raw acoustic features and encodes them into hidden representations.
    """
    def __init__(self, feature_dim, hidden_dim, num_layers, dropout):
        """
        Initializes the Encoder.

        Args:
            feature_dim (int): Dimension of input features (prosodic + raw acoustic).
            hidden_dim (int): Dimension of LSTM hidden states.
            num_layers (int): Number of LSTM layers.
            dropout (float): Dropout rate between LSTM layers.
        """
        super(Encoder, self).__init__()
        # Bidirectional LSTM with specified input and hidden dimensions
        self.lstm = nn.LSTM(feature_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True, bidirectional=True)

    def forward(self, features, lengths):
        """
        Forward pass of the Encoder.

        Args:
            features (torch.Tensor): Concatenated prosodic and raw acoustic features (batch_size, seq_len, feature_dim).
            lengths (torch.Tensor): Lengths of each sequence in the batch (batch_size).

        Returns:
            tuple: (encoder_outputs, hidden, cell)
                - encoder_outputs (torch.Tensor): LSTM outputs for each time step (batch_size, seq_len, hidden_dim * 2).
                - hidden (torch.Tensor): Final hidden states (num_layers * 2, batch_size, hidden_dim).
                - cell (torch.Tensor): Final cell states (num_layers * 2, batch_size, hidden_dim).
        """
        # Sort sequences by lengths in descending order for packing
        lengths_sorted, sorted_indices = lengths.sort(descending=True)
        features = features[sorted_indices]

        # Pack the padded sequences for efficient processing
        packed_input = rnn_utils.pack_padded_sequence(features, lengths_sorted.cpu(), batch_first=True, enforce_sorted=True)

        # Pass through the LSTM
        packed_output, (hidden, cell) = self.lstm(packed_input)

        # Unpack the LSTM outputs
        outputs, _ = rnn_utils.pad_packed_sequence(packed_output, batch_first=True, padding_value=0.0)

        # Restore the original ordering of sequences
        _, original_indices = sorted_indices.sort()
        outputs = outputs[original_indices]
        hidden = hidden[:, original_indices]
        cell = cell[:, original_indices]

        return outputs, hidden, cell

# ==============================
# Decoder Model
# ==============================

class Decoder(nn.Module):
    """
    Decoder module with attention mechanism.

    Generates predictions based on encoder outputs and current hidden state.
    """
    def __init__(self, hidden_dim, output_dim, num_layers, dropout, num_attention_layers):
        """
        Initializes the Decoder.

        Args:
            hidden_dim (int): Dimension of LSTM hidden states.
            output_dim (int): Number of output classes.
            num_layers (int): Number of LSTM layers.
            dropout (float): Dropout rate between LSTM layers.
            num_attention_layers (int): Number of attention layers.
        """
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        # Bidirectional LSTM that takes concatenated context and encoder outputs as input
        self.lstm = nn.LSTM(hidden_dim * 4, hidden_dim, num_layers, dropout=dropout, batch_first=True, bidirectional=True)
        # Fully connected layer to generate output predictions
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        # Multi-head attention mechanism
        self.attention = MultiAttention(hidden_dim, num_attention_layers)

    def forward(self, encoder_outputs, hidden, cell, lengths):
        """
        Forward pass of the Decoder.

        Args:
            encoder_outputs (torch.Tensor): Outputs from the encoder (batch_size, seq_len, hidden_dim * 2).
            hidden (torch.Tensor): Hidden state from the encoder (num_layers * 2, batch_size, hidden_dim).
            cell (torch.Tensor): Cell state from the encoder (num_layers * 2, batch_size, hidden_dim).
            lengths (torch.Tensor): Lengths of each sequence in the batch (batch_size).

        Returns:
            tuple: (predictions, (hidden, cell))
                - predictions (torch.Tensor): Predicted labels (batch_size, seq_len, output_dim).
                - hidden (torch.Tensor): Updated hidden states.
                - cell (torch.Tensor): Updated cell states.
        """
        max_len = encoder_outputs.size(1)
        batch_size = encoder_outputs.size(0)
        # Create mask based on lengths to ignore padding
        mask = torch.arange(max_len).unsqueeze(0).expand(batch_size, max_len).to(encoder_outputs.device)
        mask = mask < lengths.unsqueeze(1)  # Shape: [batch_size, seq_len]

        # Compute attention weights using the last layer's hidden state
        attn_weights = self.attention(hidden, encoder_outputs, mask)  # Shape: [batch_size, seq_len]
        # Compute context vector as weighted sum of encoder outputs
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)  # Shape: [batch_size, hidden_dim * 2]

        # Prepare LSTM input by concatenating context with encoder outputs
        lstm_input = torch.cat(
            (context.unsqueeze(1).repeat(1, encoder_outputs.size(1), 1), encoder_outputs),
            dim=2
        )  # Shape: [batch_size, seq_len, hidden_dim * 4]

        # Pass through the LSTM
        outputs, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))  # outputs: [batch_size, seq_len, hidden_dim * 2]

        # Generate predictions using the fully connected layer
        predictions = self.fc(outputs)  # Shape: [batch_size, seq_len, output_dim]

        return predictions, (hidden, cell)

# ==============================
# Sequence-to-Sequence Model
# ==============================

class Seq2Seq(nn.Module):
    """
    Sequence-to-Sequence model combining Encoder and Decoder.

    Handles the flow of data from input features through the encoder and decoder to generate predictions.
    """
    def __init__(self, encoder, decoder, prosodic_projection, acoustic_projection):
        """
        Initializes the Seq2Seq model.

        Args:
            encoder (Encoder): Encoder instance.
            decoder (Decoder): Decoder instance.
            prosodic_projection (FeatureProjection): Projection layer for prosodic features.
            acoustic_projection (FeatureProjection): Projection layer for raw acoustic features.
        """
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.prosodic_projection = prosodic_projection
        self.acoustic_projection = acoustic_projection

    def forward(self, prosodic_features, raw_acoustic_features, lengths):
        """
        Forward pass of the Seq2Seq model.

        Args:
            prosodic_features (torch.Tensor): Prosodic features (batch_size, seq_len, prosodic_dim).
            raw_acoustic_features (torch.Tensor): Raw acoustic features (batch_size, seq_len, acoustic_dim).
            lengths (torch.Tensor): Lengths of each sequence in the batch (batch_size).

        Returns:
            torch.Tensor: Output predictions (batch_size, seq_len, output_dim).
        """
        # Project prosodic and raw acoustic features to a lower dimension
        projected_prosodic = self.prosodic_projection(prosodic_features)  # Shape: [batch_size, seq_len, projected_dim]
        projected_acoustic = self.acoustic_projection(raw_acoustic_features)  # Shape: [batch_size, seq_len, projected_dim]
        # Concatenate projected prosodic and acoustic features along the feature dimension
        features = torch.cat((projected_prosodic, projected_acoustic), dim=2)  # Shape: [batch_size, seq_len, projected_dim * 2]
        # Pass the concatenated features through the encoder
        encoder_outputs, hidden, cell = self.encoder(features, lengths)
        # Pass the encoder outputs and hidden states to the decoder to get predictions
        outputs, (hidden, cell) = self.decoder(encoder_outputs, hidden, cell, lengths)
        return outputs

# ==============================
# Training Function
# ==============================

def train(model, iterator, optimizer, criterion):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The Seq2Seq model.
        iterator (DataLoader): DataLoader for training data.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        criterion (nn.Module): Loss function.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()  # Set model to training mode
    epoch_loss = 0
    for words, prosodic_features, raw_acoustic_features, labels, lengths in iterator:
        # Move tensors to the configured device
        prosodic_features = prosodic_features.to(device)
        raw_acoustic_features = raw_acoustic_features.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)

        optimizer.zero_grad()  # Reset gradients
        # Forward pass through the model
        output = model(prosodic_features, raw_acoustic_features, lengths)  # Shape: [batch_size, seq_len, num_classes]

        # Flatten outputs and labels for loss computation
        output = output.view(-1, num_classes)  # Shape: [batch_size * seq_len, num_classes]
        labels = labels.view(-1)  # Shape: [batch_size * seq_len]

        # Create mask to ignore padding positions
        mask = labels != PADDING_VALUE
        masked_output = output[mask]  # Shape: [num_valid_tokens, num_classes]
        masked_labels = labels[mask]  # Shape: [num_valid_tokens]

        # Compute loss using the masked outputs and labels
        loss = criterion(masked_output, masked_labels)
        loss.backward()  # Backpropagation
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        optimizer.step()  # Update model parameters

        epoch_loss += loss.item()  # Accumulate loss

    return epoch_loss / len(iterator)

# ==============================
# Evaluation Function
# ==============================

def evaluate(model, iterator, criterion):
    """
    Evaluates the model on validation or test data.

    Args:
        model (nn.Module): The Seq2Seq model.
        iterator (DataLoader): DataLoader for validation or test data.
        criterion (nn.Module): Loss function.

    Returns:
        tuple: (average_loss, accuracy, precision, recall, f1_score)
    """
    model.eval()  # Set model to evaluation mode
    epoch_loss = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():  # Disable gradient computation
        for words, prosodic_features, raw_acoustic_features, labels, lengths in iterator:
            # Move tensors to the configured device
            prosodic_features = prosodic_features.to(device)
            raw_acoustic_features = raw_acoustic_features.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)

            # Forward pass through the model
            output = model(prosodic_features, raw_acoustic_features, lengths)  # Shape: [batch_size, seq_len, num_classes]
            output = output.view(-1, num_classes)  # Shape: [batch_size * seq_len, num_classes]
            labels = labels.view(-1)  # Shape: [batch_size * seq_len]

            # Create mask to ignore padding positions
            mask = labels != PADDING_VALUE
            masked_output = output[mask]  # Shape: [num_valid_tokens, num_classes]
            masked_labels = labels[mask]  # Shape: [num_valid_tokens]

            # Compute loss using the masked outputs and labels
            loss = criterion(masked_output, masked_labels)
            epoch_loss += loss.item()  # Accumulate loss

            # Generate predictions by taking the argmax over class scores
            preds = torch.argmax(masked_output, dim=1)  # Shape: [num_valid_tokens]
            all_labels.extend(masked_labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

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
        model (nn.Module): The trained Seq2Seq model.
        iterator (DataLoader): DataLoader for test data.

    Returns:
        tuple: (all_labels, all_preds)
            - all_labels (list): List of true labels.
            - all_preds (list): List of predicted labels.
    """
    model.eval()  # Set model to evaluation mode
    all_labels = []
    all_preds = []

    # Path to save test results
    results_filepath = './outputs/prosody_raw_acoustic_results.txt'
    with open(results_filepath, 'w') as file:
        file.write("")

    with torch.no_grad():  # Disable gradient computation
        for words, prosodic_features, raw_acoustic_features, labels, lengths in iterator:
            # Move tensors to the configured device
            prosodic_features = prosodic_features.to(device)
            raw_acoustic_features = raw_acoustic_features.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)

            # Forward pass through the model
            output = model(prosodic_features, raw_acoustic_features, lengths)  # Shape: [batch_size, seq_len, num_classes]
            preds = torch.argmax(output, dim=2)  # Shape: [batch_size, seq_len]

            for i in range(prosodic_features.shape[0]):
                word_sentence = words[i]  # List of words in the sentence
                gold_labels = labels[i].cpu().numpy().flatten()  # True labels
                pred_labels = preds[i].cpu().numpy().flatten()  # Predicted labels

                # Clean up the sentence by excluding padding positions and punctuation
                cleaned_words, cleaned_gold_labels, cleaned_pred_labels = clean_up_sentence(
                    word_sentence, gold_labels, pred_labels
                )

                # Create a DataFrame with words and their corresponding labels
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

    # Calculate evaluation metrics
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
# Plotting Function
# ==============================

def plot_metrics(train_losses, val_losses, val_accuracies, val_precisions, val_recalls, val_f1s):
    """
    Plots training and validation metrics over epochs and saves the plot.

    Args:
        train_losses (list): List of training losses per epoch.
        val_losses (list): List of validation losses per epoch.
        val_accuracies (list): List of validation accuracies per epoch.
        val_precisions (list): List of validation precisions per epoch.
        val_recalls (list): List of validation recalls per epoch.
        val_f1s (list): List of validation F1 scores per epoch.
    """
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 8))

    # Plot Training and Validation Loss
    plt.subplot(2, 3, 1)
    plt.plot(epochs, train_losses, label='Train Loss', color='blue')
    plt.plot(epochs, val_losses, label='Validation Loss', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.legend()

    # Plot Validation Accuracy
    plt.subplot(2, 3, 2)
    plt.plot(epochs, val_accuracies, label='Validation Accuracy', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    plt.legend()

    # Plot Validation Precision
    plt.subplot(2, 3, 3)
    plt.plot(epochs, val_precisions, label='Validation Precision', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.title('Precision')
    plt.legend()

    # Plot Validation Recall
    plt.subplot(2, 3, 4)
    plt.plot(epochs, val_recalls, label='Validation Recall', color='purple')
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.title('Recall')
    plt.legend()

    # Plot Validation F1 Score
    plt.subplot(2, 3, 5)
    plt.plot(epochs, val_f1s, label='Validation F1 Score', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.title('F1 Score')
    plt.legend()

    plt.tight_layout()
    # Save the plot to a file
    plt.savefig('./outputs/prosody_raw_acoustic_metrics.png')
    plt.close()
    print(f'Metrics plot saved to ./outputs/prosody_raw_acoustic_metrics.png')

# ==============================
# Main Execution Block
# ==============================

if __name__ == "__main__":
    # ==============================
    # Configuration and Setup
    # ==============================

    # Set random seed for reproducibility
    seed = 42
    set_seed(seed)

    # Path to the training data JSON file
    json_path = '../prosody/data/ambiguous_prosodic_raw_acoustic_ml_features.json' 
    data = load_data(json_path)  # Load data from JSON

    # Create a descriptive filename for the best model based on dataset and task names
    dataset_name = "ambiguous_instructions"
    task_name = "prosody_raw_multiclass"
    best_model_filename = f"models/best-model-{dataset_name}-{task_name}.pt"

    # Split data into training, validation, and test sets with specified ratios
    train_data, val_data, test_data = split_data(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)

    # Create Dataset instances for training, validation, and testing
    train_dataset = ProsodyDataset(dict(train_data))
    val_dataset = ProsodyDataset(dict(val_data))
    test_dataset = ProsodyDataset(dict(test_data))

    # Print dataset sizes
    print(f'Train size: {len(train_dataset)}')
    print(f'Validation size: {len(val_dataset)}')
    print(f'Test size: {len(test_dataset)}')

    # Create DataLoader instances for batching and shuffling
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

    # Get feature dimensions from the dataset (after padding)
    sample_batch = next(iter(train_loader))
    sample_words, sample_prosodic_features, sample_raw_acoustic_features, sample_labels, sample_lengths = sample_batch
    prosodic_features_dim = sample_prosodic_features.shape[2]
    raw_acoustic_features_dim = sample_raw_acoustic_features.shape[2]
    print(f"Prosodic features dimension: {prosodic_features_dim}")
    print(f"Raw acoustic features dimension: {raw_acoustic_features_dim}")

    # Determine the number of classes based on training labels
    all_labels = []
    for _, _, _, labels in train_dataset:
        all_labels.extend(labels.numpy().flatten())

    num_classes = len(np.unique(all_labels))
    print(f'Model Training with {num_classes} classes')

    # Define model parameters
    PROJECTED_DIM = 128  # Dimension after feature projection
    INPUT_DIM = PROJECTED_DIM * 2  # Concatenated projected prosodic and acoustic features
    HIDDEN_DIM = 256      # Dimension of LSTM hidden states
    OUTPUT_DIM = num_classes  # Number of output classes for classification
    NUM_LAYERS = 8        # Number of LSTM layers
    DROPOUT = 0.13234009854266668  # Dropout rate between LSTM layers
    NUM_ATTENTION_LAYERS = 8  # Number of attention layers

    # Set the device to GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using Device: {device}')

    # Instantiate the feature projection layers for prosodic and raw acoustic features
    prosodic_projection = FeatureProjection(prosodic_features_dim, PROJECTED_DIM).to(device)
    acoustic_projection = FeatureProjection(raw_acoustic_features_dim, PROJECTED_DIM).to(device)

    # Instantiate the Encoder and Decoder
    encoder = Encoder(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT).to(device)
    decoder = Decoder(HIDDEN_DIM, OUTPUT_DIM, NUM_LAYERS, DROPOUT, NUM_ATTENTION_LAYERS).to(device)

    # Combine Encoder and Decoder into a Seq2Seq model with feature projections
    model = Seq2Seq(encoder, decoder, prosodic_projection, acoustic_projection).to(device)

    # Print a summary of the model architecture
    summary(model, input_data=(sample_prosodic_features.to(device), sample_raw_acoustic_features.to(device), sample_lengths.to(device)), device=device)

    # ==============================
    # Optimizer, Scheduler, and Loss Function
    # ==============================

    optimizer = optim.Adam(model.parameters(), 
                           lr=0.0005438229945889153, 
                           weight_decay=1e-5)  # Adam optimizer with learning rate and weight decay
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)  # Learning rate scheduler

    # Define the loss function with ignoring the padding index
    criterion = nn.CrossEntropyLoss(ignore_index=PADDING_VALUE)

    # ==============================
    # Initialize Metrics Storage
    # ==============================

    train_losses = []
    val_losses = []
    val_accuracies = []
    val_precisions = []
    val_recalls = []
    val_f1s = []

    # Training parameters
    N_EPOCHS = 500  # Maximum number of training epochs
    CLIP = 1         # Gradient clipping value to prevent exploding gradients

    # Initialize EarlyStopping instance with patience and minimum delta
    early_stopping = EarlyStopping(patience=10, min_delta=0.001)
    best_valid_loss = float('inf')  # Initialize the best validation loss

    # ==============================
    # Training Loop
    # ==============================

    for epoch in range(N_EPOCHS):
        # Train the model for one epoch and get the training loss
        train_loss = train(model, train_loader, optimizer, criterion)
        # Evaluate the model on the validation set and get validation metrics
        valid_loss, valid_acc, valid_precision, valid_recall, valid_f1 = evaluate(model, test_loader, criterion)

        # Append current epoch's metrics to the lists for later visualization
        train_losses.append(train_loss)
        val_losses.append(valid_loss)
        val_accuracies.append(valid_acc)
        val_precisions.append(valid_precision)
        val_recalls.append(valid_recall)
        val_f1s.append(valid_f1)

        # Update the learning rate scheduler based on validation loss
        scheduler.step(valid_loss)

        # Check if the current validation loss is the best so far
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            # Save the model state if validation loss has improved
            torch.save(model.state_dict(), best_model_filename)

        # Print epoch statistics
        print(f'Epoch: {epoch+1:02}')
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\tVal. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:.2f}%')
        print(f'\tPrecision: {valid_precision:.2f} | Recall: {valid_recall:.2f} | F1 Score: {valid_f1:.2f}')

        # Check for early stopping
        early_stopping(valid_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # ==============================
    # Load Best Model and Test
    # ==============================

    # Load the best model saved during training
    model.load_state_dict(torch.load(best_model_filename))
    # Test the model on the test set and print detailed results
    test_model(model, test_loader)
    # Plot and save the training and validation metrics
    plot_metrics(train_losses, val_losses, val_accuracies, val_precisions, val_recalls, val_f1s)

    # ==============================
    # Evaluate on a New Held-Out Set
    # ==============================

    eval_json = "../prosody/data/ambiguous_prosodic_raw_acoustic_ml_features_eval.json"
    # Evaluate the model on the new dataset and get predictions
    evaluate_new_set(model, eval_json)
