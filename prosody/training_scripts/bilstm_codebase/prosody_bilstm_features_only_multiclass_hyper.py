"""
Prosody Multiclass Classification using BiLSTM with Attention and Hyperparameter Optimization

This module implements a sequence-to-sequence model using Bidirectional LSTM (BiLSTM) 
with attention mechanisms for multiclass classification on prosody data. It includes 
data preprocessing, model definition, training with hyperparameter optimization using 
Optuna, evaluation, and testing functionalities.

Dependencies:
    - Python 3.x
    - PyTorch
    - NumPy
    - scikit-learn
    - matplotlib
    - pandas
    - Optuna

Author: Your Name
Date: 2024-11-08
"""

import json
import random
import string
import torch
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
import optuna  # Import Optuna for hyperparameter optimization

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

def get_corpus(data):
    """
    Extracts and tokenizes words from the dataset to build a corpus.

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
        processed_words.extend(re.findall(r"[\w']+|[.,!?;]", word))
    return processed_words

def clean_up_sentence(words, gold_labels, pred_labels):
    """
    Cleans up the sentence by removing unknown and padding tokens.

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

    Each sample consists of tokenized words, associated features, and labels.
    """

    def __init__(self, data):
        """
        Initializes the dataset.

        Args:
            data (dict): The dataset containing entries with words, features, and labels.
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
            tuple: (processed_words, features_tensor, labels_tensor)
        """
        key, item = self.entries[idx]
        processed_words = preprocess_text(item['words'])
        features = torch.tensor(item['features'], dtype=torch.float32)
        labels = torch.tensor(item['labels'], dtype=torch.long)  # Changed to torch.long for class indices
        
        return processed_words, features, labels

# ==============================
# Collate Function
# ==============================

def collate_fn(batch):
    """
    Collate function to handle batches with variable sequence lengths.

    Pads sequences and prepares tensors for words, features, and labels.

    Args:
        batch (list): List of samples from the dataset.

    Returns:
        tuple: Words list, padded features, padded labels, and their lengths.
    """
    words = [item[0] for item in batch]  # List of lists of words
    features = [item[1] for item in batch]
    labels = [item[2] for item in batch]

    # Pad features and labels
    features_padded = torch.nn.utils.rnn.pad_sequence(features, batch_first=True, padding_value=0.0)
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=PADDING_VALUE)

    # Record the lengths of each feature sequence
    lengths = torch.tensor([len(f) for f in features])

    return words, features_padded, labels_padded, lengths

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

class Encoder(nn.Module):
    """
    Encoder module using a Bidirectional LSTM with feature integration.

    Attributes:
        lstm (nn.LSTM): Bidirectional LSTM layer.
    """

    def __init__(self, feature_dim, hidden_dim, num_layers, dropout):
        """
        Initializes the Encoder.

        Args:
            feature_dim (int): Dimension of input features.
            hidden_dim (int): Dimension of LSTM hidden states.
            num_layers (int): Number of LSTM layers.
            dropout (float): Dropout rate.
        """
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(feature_dim, hidden_dim, num_layers, 
                            dropout=dropout, batch_first=True, bidirectional=True)

    def forward(self, features, lengths):
        """
        Forward pass of the Encoder.

        Args:
            features (torch.Tensor): Batch of features (batch_size, seq_len, feature_dim).
            lengths (torch.Tensor): Lengths of each sequence in the batch (batch_size).

        Returns:
            tuple: Encoder outputs, hidden states, and cell states.
        """
        # Sort sequences by lengths in descending order for packing
        lengths_sorted, sorted_indices = lengths.sort(descending=True)
        features = features[sorted_indices]

        # Pack the sequences for efficient processing
        packed_input = rnn_utils.pack_padded_sequence(features, lengths_sorted.cpu(), 
                                                      batch_first=True, enforce_sorted=True)

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
    """

    def __init__(self, hidden_dim, output_dim, num_layers, dropout, num_attention_layers):
        """
        Initializes the Decoder.

        Args:
            hidden_dim (int): Dimension of LSTM hidden states.
            output_dim (int): Number of output classes.
            num_layers (int): Number of LSTM layers.
            dropout (float): Dropout rate.
            num_attention_layers (int): Number of attention layers.
        """
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(hidden_dim * 4, hidden_dim, num_layers, 
                            dropout=dropout, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.attention = MultiAttention(hidden_dim, num_attention_layers)

    def forward(self, encoder_outputs, hidden, cell, mask):
        """
        Forward pass of the Decoder.

        Args:
            encoder_outputs (torch.Tensor): Outputs from the encoder (batch_size, seq_len, hidden_dim*2).
            hidden (torch.Tensor): Hidden state from the encoder (num_layers*2, batch_size, hidden_dim).
            cell (torch.Tensor): Cell state from the encoder (num_layers*2, batch_size, hidden_dim).
            mask (torch.Tensor): Mask to ignore padding tokens (batch_size, seq_len).

        Returns:
            tuple: Predictions and updated hidden and cell states.
        """
        # Compute attention weights using the last layer's hidden state
        attn_weights = self.attention(hidden[-1], encoder_outputs, mask)  # (batch_size, seq_len)
        # Compute context vector as weighted sum of encoder outputs
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)  # (batch_size, hidden_dim*2)

        # Prepare LSTM input by concatenating context with encoder outputs
        lstm_input = torch.cat(
            (context.unsqueeze(1).repeat(1, encoder_outputs.size(1), 1), 
             encoder_outputs), dim=2
        )  # (batch_size, seq_len, hidden_dim*4)

        # Pass through LSTM
        outputs, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))  # outputs: (batch_size, seq_len, hidden_dim*2)

        # Generate predictions
        predictions = self.fc(outputs)  # (batch_size, seq_len, output_dim)

        return predictions, (hidden, cell)

class Seq2Seq(nn.Module):
    """
    Sequence-to-Sequence model combining Encoder and Decoder.

    Attributes:
        encoder (Encoder): The encoder component.
        decoder (Decoder): The decoder component.
    """

    def __init__(self, encoder, decoder):
        """
        Initializes the Seq2Seq model.

        Args:
            encoder (Encoder): Encoder instance.
            decoder (Decoder): Decoder instance.
        """
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, features, lengths):
        """
        Forward pass of the Seq2Seq model.

        Args:
            features (torch.Tensor): Batch of features (batch_size, seq_len, feature_dim).
            lengths (torch.Tensor): Lengths of each sequence in the batch (batch_size).

        Returns:
            torch.Tensor: Output predictions (batch_size, seq_len, output_dim).
        """
        # Encode the input features
        encoder_outputs, hidden, cell = self.encoder(features, lengths)
        max_len = encoder_outputs.size(1)
        batch_size = encoder_outputs.size(0)

        # Create mask based on lengths to ignore padding
        mask = torch.arange(max_len).unsqueeze(0).expand(batch_size, max_len).to(features.device)
        mask = mask < lengths.unsqueeze(1)  # (batch_size, seq_len)

        # Decode the encoder outputs
        outputs, (hidden, cell) = self.decoder(encoder_outputs, hidden, cell, mask)

        return outputs  # (batch_size, seq_len, output_dim)

# ==============================
# Training and Evaluation Functions
# ==============================

def train(model, iterator, optimizer, criterion):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The Seq2Seq model.
        iterator (DataLoader): DataLoader for training data.
        optimizer (torch.optim.Optimizer): Optimizer.
        criterion (nn.Module): Loss function.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    epoch_loss = 0
    for words, features, labels, lengths in iterator:
        features = features.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)

        optimizer.zero_grad()
        output = model(features, lengths)  # (batch_size, seq_len, num_classes)
        
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

def evaluate(model, iterator, criterion):
    """
    Evaluates the model on validation data.

    Args:
        model (nn.Module): The Seq2Seq model.
        iterator (DataLoader): DataLoader for validation data.
        criterion (nn.Module): Loss function.

    Returns:
        tuple: Average validation loss, accuracy, precision, recall, and F1 score.
    """
    model.eval()
    epoch_loss = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for words, features, labels, lengths in iterator:
            features = features.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)

            output = model(features, lengths)  # (batch_size, seq_len, num_classes)
            
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

def test_model(model, iterator):
    """
    Tests the model on the test dataset and saves detailed results.

    Args:
        model (nn.Module): The trained Seq2Seq model.
        iterator (DataLoader): DataLoader for test data.

    Returns:
        tuple: All true labels and all predicted labels.
    """
    model.eval()
    all_labels = []
    all_preds = []

    # Initialize results file
    results_filepath = './outputs/prosody_bilstm_features_multiclass_results.txt'
    with open(results_filepath, 'w') as file:
        file.write("")

    with torch.no_grad():
        for words, features, labels, lengths in iterator:
            features = features.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)

            output = model(features, lengths)  # (batch_size, seq_len, num_classes)
            preds = torch.argmax(output, dim=2)  # (batch_size, seq_len)

            for i in range(features.shape[0]):
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

    print(f'Test Accuracy: {accuracy*100:.2f}%')
    print(f'Test Precision: {precision*100:.2f}%')
    print(f'Test Recall: {recall*100:.2f}%')
    print(f'Test F1 Score: {f1*100:.2f}%')

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
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
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
    metrics_filepath = './outputs/bilstm_features_multiclass_metrics.png'
    plt.savefig(metrics_filepath)
    plt.close()
    print(f'Metrics plot saved to {metrics_filepath}')

# ==============================
# Hyperparameter Optimization with Optuna
# ==============================

def objective(trial):
    """
    Objective function for Optuna hyperparameter optimization.

    Args:
        trial (optuna.trial.Trial): Optuna trial object.

    Returns:
        float: Validation F1 score to maximize.
    """
    # Suggest hyperparameters
    HIDDEN_DIM = trial.suggest_categorical('HIDDEN_DIM', [128, 256, 512])
    NUM_LAYERS = trial.suggest_int('NUM_LAYERS', 2, 8, step=2)
    DROPOUT = trial.suggest_float('DROPOUT', 0.1, 0.5)
    NUM_ATTENTION_LAYERS = trial.suggest_int('NUM_ATTENTION_LAYERS', 2, 8, step=2)
    LR = trial.suggest_float('LR', 1e-5, 1e-2, log=True)
    WEIGHT_DECAY = trial.suggest_float('WEIGHT_DECAY', 1e-6, 1e-3, log=True)
    BATCH_SIZE = trial.suggest_categorical('BATCH_SIZE', [32, 64, 128])

    # Create data loaders with the suggested batch size
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # Get feature dimension from the dataset
    sample_words, sample_features, sample_labels, sample_lengths = next(iter(train_loader))
    feature_dim = sample_features.shape[2]

    # Determine number of classes dynamically
    all_labels = []
    for _, _, labels in train_dataset:
        all_labels.extend(labels.numpy().flatten())
    global num_classes
    num_classes = len(np.unique(all_labels))

    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize Encoder and Decoder with suggested hyperparameters
    encoder = Encoder(feature_dim, HIDDEN_DIM, NUM_LAYERS, DROPOUT).to(device)
    decoder = Decoder(HIDDEN_DIM, num_classes, NUM_LAYERS, DROPOUT, NUM_ATTENTION_LAYERS).to(device)
    model = Seq2Seq(encoder, decoder).to(device)

    # Initialize optimizer and scheduler with suggested hyperparameters
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=False)

    # Define loss function
    criterion = nn.CrossEntropyLoss(ignore_index=PADDING_VALUE)

    # Training parameters
    N_EPOCHS = 50

    # Initialize early stopping
    early_stopping = EarlyStopping(patience=10, min_delta=0.001)
    best_valid_f1 = 0.0

    for epoch in range(N_EPOCHS):
        train_loss = train(model, train_loader, optimizer, criterion)
        valid_loss, valid_acc, valid_precision, valid_recall, valid_f1 = evaluate(model, val_loader, criterion)

        # Update the learning rate scheduler based on validation loss
        scheduler.step(valid_loss)

        # Update the best validation F1 score
        if valid_f1 > best_valid_f1:
            best_valid_f1 = valid_f1

        # Check for early stopping
        early_stopping(valid_loss)
        if early_stopping.early_stop:
            break

    return best_valid_f1  # Return validation F1 score to maximize

# ==============================
# Main Execution
# ==============================

if __name__ == "__main__":
    # ==============================
    # Configuration and Setup
    # ==============================

    seed = 42
    set_seed(seed)  # Set seed for reproducibility

    # Path to training data
    json_path = '../prosody/data/ambiguous_prosody_multi_label_features_train.json'
    data = load_data(json_path)  # Load data

    # Create a descriptive filename for the model
    dataset_name = "ambiguous_instructions"
    task_name = "prosody_multiclass"
    best_model_filename = f"models/best-model-{dataset_name}-{task_name}.pt"

    # Split data into train, validation, and test sets
    train_data, val_data, test_data = split_data(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)

    # Initialize datasets
    train_dataset = ProsodyDataset(dict(train_data))
    val_dataset = ProsodyDataset(dict(val_data))
    test_dataset = ProsodyDataset(dict(test_data))

    print(f'Train size: {len(train_dataset)}')
    print(f'Validation size: {len(val_dataset)}')
    print(f'Test size: {len(test_dataset)}')

    # ==============================
    # Hyperparameter Optimization with Optuna
    # ==============================

    print("Starting hyperparameter optimization with Optuna...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=200)  # Increase n_trials for more extensive search

    print('Best trial:')
    trial = study.best_trial
    print(f'  F1 Score: {trial.value:.4f}')
    print('  Params: ')
    for key, value in trial.params.items():
        print(f'    {key}: {value}')

    # ==============================
    # Retraining with Best Hyperparameters
    # ==============================

    # Use the best hyperparameters to retrain the model on the combined train and validation sets
    best_params = trial.params

    # Combine train and validation datasets
    combined_data = train_dataset.entries + val_dataset.entries
    combined_dataset = ProsodyDataset(dict(combined_data))

    # Create data loaders with the best batch size
    BATCH_SIZE = best_params['BATCH_SIZE']
    train_loader = DataLoader(combined_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # Get feature dimension from the dataset
    sample_words, sample_features, sample_labels, sample_lengths = next(iter(train_loader))
    feature_dim = sample_features.shape[2]

    # Determine number of classes dynamically
    all_labels = []
    for _, _, labels in train_dataset:
        all_labels.extend(labels.numpy().flatten())
    num_classes = len(np.unique(all_labels))

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Unpack best hyperparameters
    HIDDEN_DIM = best_params['HIDDEN_DIM']
    NUM_LAYERS = best_params['NUM_LAYERS']
    DROPOUT = best_params['DROPOUT']
    NUM_ATTENTION_LAYERS = best_params['NUM_ATTENTION_LAYERS']
    LR = best_params['LR']
    WEIGHT_DECAY = best_params['WEIGHT_DECAY']

    # Initialize Encoder and Decoder with best hyperparameters
    encoder = Encoder(feature_dim, HIDDEN_DIM, NUM_LAYERS, DROPOUT).to(device)
    decoder = Decoder(HIDDEN_DIM, num_classes, NUM_LAYERS, DROPOUT, NUM_ATTENTION_LAYERS).to(device)
    model = Seq2Seq(encoder, decoder).to(device)

    # Initialize optimizer and scheduler with best hyperparameters
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    # Define loss function
    criterion = nn.CrossEntropyLoss(ignore_index=PADDING_VALUE)

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
    early_stopping = EarlyStopping(patience=10, min_delta=0.001)
    best_valid_loss = float('inf')

    print("Starting training with best hyperparameters...")
    for epoch in range(N_EPOCHS):
        train_loss = train(model, train_loader, optimizer, criterion)
        valid_loss, valid_acc, valid_precision, valid_recall, valid_f1 = evaluate(model, train_loader, criterion)  # Using train_loader as validation since combined

        train_losses.append(train_loss)
        val_losses.append(valid_loss)
        val_accuracies.append(valid_acc)
        val_precisions.append(valid_precision)
        val_recalls.append(valid_recall)
        val_f1s.append(valid_f1)

        # Update the learning rate scheduler based on validation loss
        scheduler.step(valid_loss)

        # Save the model if validation loss improves
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), best_model_filename)

        print(f'Epoch: {epoch+1:02}')
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}% |  Precision: {valid_precision:.2f} |  Recall: {valid_recall:.2f} |  F1 Score: {valid_f1:.2f}')

        # Check for early stopping
        early_stopping(valid_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    # ==============================
    # Testing and Evaluation
    # ==============================

    print("Loading the best model and evaluating on the test set...")
    # Load the best model
    model.load_state_dict(torch.load(best_model_filename))
    test_model(model, test_loader)

    # ==============================
    # Plotting Metrics
    # ==============================

    print("Plotting training and validation metrics...")
    plot_metrics(train_losses, val_losses, val_accuracies, val_precisions, val_recalls, val_f1s)
