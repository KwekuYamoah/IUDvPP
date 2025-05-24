"""
Main script for training and evaluating a transformer-based prosody prediction model.

This version of the script trains with raw acoustic features, prosodic features and word embeddings.
The architecture remains essentially the same except that the encoder now concatenates
word embeddings, prosodic features, and raw acoustic features before projection.


Dependencies:
    - Python 3.x
    - PyTorch
    - NumPy
    - scikit-learn
    - matplotlib
    - pandas
    - Optuna
    - pynvml

Author: Kweku Andoh Yamoah
Date: 2025-02-05
"""

import os
import json
import random
import numpy as np
import re
import string
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import multilabel_confusion_matrix, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchinfo import summary

# =============================================================================
# Utility Functions and Constants
# =============================================================================

def set_seed(seed):
    """
    Sets the random seed for Python's `random` module, NumPy, and PyTorch to ensure reproducibility.
    Args:
        seed (int): The seed value to use for random number generators.
    Notes:
        - This function also sets PyTorch's cuDNN backend to be deterministic and disables the benchmark mode
          to further ensure reproducible results when using CUDA.
    """
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

PADDING_VALUE = 0  # Padding index

class EarlyStopping:
    """
    Implements early stopping to terminate training when validation loss stops improving.
    Attributes:
        patience (int): Number of epochs to wait after last improvement before stopping.
        min_delta (float): Minimum change in the monitored loss to qualify as an improvement.
        best_loss (float or None): Best recorded validation loss.
        counter (int): Number of epochs since last improvement.
        early_stop (bool): Flag indicating whether training should be stopped.
    Methods:
        __call__(val_loss):
            Call with the current validation loss after each epoch.
            Updates internal state and sets `early_stop` to True if stopping criteria are met.
    Args:
        patience (int, optional): How many epochs to wait after last time validation loss improved. Default is 5.
        min_delta (float, optional): Minimum change in the monitored loss to qualify as an improvement. Default is 0.
    """
   
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        elif val_loss > self.best_loss + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def load_data(json_path):
    """Load data from a JSON file."""
    with open(json_path, 'r') as file:
        data = json.load(file)
    return data

def split_data(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Splits a dictionary of data into training, validation, and test sets based on specified ratios.
    Args:
        data (dict): The data to split, where keys are identifiers and values are data samples.
        train_ratio (float, optional): Proportion of data to use for the training set. Defaults to 0.8.
        val_ratio (float, optional): Proportion of data to use for the validation set. Defaults to 0.1.
        test_ratio (float, optional): Proportion of data to use for the test set. Defaults to 0.1.
    Returns:
        list: A list of three subsets (train, validation, test), each as a list of (key, value) tuples.
    Raises:
        AssertionError: If the sum of train_ratio, val_ratio, and test_ratio does not equal 1.0.
    """
    
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1"
    total = len(data)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    test_size = total - train_size - val_size
    return random_split(list(data.items()), [train_size, val_size, test_size])

def preprocess_text(words):
    """
    Tokenize a list of words using regex.
    """
    processed_words = []
    for word in words:
        processed_words.extend(re.findall(r"[\w']+|[.,!?;]", word))
    return processed_words

def clean_up_sentence(words, gold_labels, pred_labels, padding_value):
    """
    Filters out elements from the input lists where the corresponding gold label equals the specified padding value.

    Args:
        words (list): List of words (tokens).
        gold_labels (list): List of gold (true) labels corresponding to each word.
        pred_labels (list): List of predicted labels corresponding to each word.
        padding_value (int or str): The value in gold_labels that indicates padding and should be filtered out.

    Returns:
        tuple: Three lists containing the filtered words, gold labels, and predicted labels, respectively, 
               with all elements where the gold label was equal to padding_value removed.
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

def compute_class_weights(dataset, num_classes, device):
    """
    Computes class weights for a dataset to address class imbalance, suitable for use in loss functions.
    Args:
        dataset (Iterable): An iterable dataset where each item contains labels as the last element.
        num_classes (int): The total number of classes, including padding and special tokens.
        device (torch.device): The device on which to place the resulting weight tensor.
    Returns:
        torch.Tensor: A tensor of shape (num_classes,) containing the computed class weights, 
                      with zero weight for the padding index and SOS token.
    Notes:
        - Assumes labels are already shifted (e.g., in {1, 2, 3, 4}).
        - Excludes padding values (PADDING_VALUE) from weight computation.
        - Sets the weight for the padding index (0) and SOS token (last index) to zero.
    """
    
    all_labels = []
    for _, _, _, _, _, labels in dataset:
        # labels are already shifted, so they are in {1,2,3,4} for example.
        labels_np = labels.numpy().flatten()
        labels_np = labels_np[labels_np != PADDING_VALUE]
        all_labels.extend(labels_np)
    classes = np.array(sorted(set(all_labels)))
    computed_weights = compute_class_weight('balanced', classes=classes, y=all_labels)
    weight_vector = np.ones(num_classes)
    weight_vector[0] = 0.0                     # Padding index weight is 0.
    weight_vector[1:len(computed_weights)+1] = computed_weights  # Computed weights.
    weight_vector[num_classes - 1] = 0.0         # SOS token weight.
    weight_tensor = torch.tensor(weight_vector, dtype=torch.float).to(device)
    return weight_tensor

# =============================================================================
# Dataset and Collate Function
# =============================================================================

class ProsodyDataset(Dataset):
    """
    A PyTorch Dataset for prosody data processing with three input feature types:
        - word embeddings
        - prosodic features
        - raw acoustic features
    """
    def __init__(self, data):
        self.entries = list(data.items())

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        key, item = self.entries[idx]
        words = preprocess_text(item['words'])
        word_embeddings = torch.tensor(item['word_embeddings'], dtype=torch.float32)
        prosodic_features = torch.tensor(item['prosodic_features'], dtype=torch.float32)
        raw_acoustic_features = torch.tensor(item['raw_acoustic_features'], dtype=torch.float32)
        # Shift labels by +1 so that 0 is reserved for padding.
        labels = torch.tensor(item['labels'], dtype=torch.long) + 1
        return key, words, word_embeddings, prosodic_features, raw_acoustic_features, labels

def collate_fn(batch):
    """
    Collate function for creating batches in a DataLoader.
    Pads the sequences for word embeddings, prosodic features, raw acoustic features, and labels.
    """
    keys = [item[0] for item in batch]
    words = [item[1] for item in batch]
    word_embeddings = [item[2] for item in batch]
    prosodic_features = [item[3] for item in batch]
    raw_acoustic_features = [item[4] for item in batch]
    labels = [item[5] for item in batch]

    word_embeddings_padded = torch.nn.utils.rnn.pad_sequence(word_embeddings, batch_first=True, padding_value=0.0)
    prosodic_features_padded = torch.nn.utils.rnn.pad_sequence(prosodic_features, batch_first=True, padding_value=0.0)
    raw_acoustic_features_padded = torch.nn.utils.rnn.pad_sequence(raw_acoustic_features, batch_first=True, padding_value=0.0)
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=PADDING_VALUE)
    lengths = torch.tensor([len(f) for f in prosodic_features])
    return keys, words, word_embeddings_padded, prosodic_features_padded, raw_acoustic_features_padded, labels_padded, lengths

# =============================================================================
# Model Components
# =============================================================================

class PositionalEncoding(nn.Module):
    """
    Positional Encoding module for Transformer models.
    """
    def __init__(self, d_model, max_len=10000):
        """
        Initializes the PositionalEncoding module.

        Args:
            d_model (int): The dimensionality of the model embeddings.
            max_len (int, optional): The maximum length of input sequences. Defaults to 10000.

        Description:
            Computes and stores positional encodings as a buffer for use in transformer models.
            The positional encodings are calculated using sine and cosine functions of different frequencies,
            enabling the model to incorporate information about the position of tokens in the sequence.
        """
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class ProjectedEncoder(nn.Module):
    """
    ProjectedEncoder is a neural network module that fuses word embeddings, prosodic features, and raw acoustic features
    using a projection layer followed by a Transformer encoder.
    Args:
        prosodic_dim (int): Dimension of the prosodic feature input.
        raw_acoustic_dim (int): Dimension of the raw acoustic feature input.
        embedding_dim (int): Dimension of the word embedding input.
        hidden_dim (int): Dimension of the hidden state in the projection and transformer layers.
        num_layers (int): Number of layers in the Transformer encoder.
        dropout (float): Dropout probability applied after projection and within the Transformer encoder.
        num_heads (int, optional): Number of attention heads in the Transformer encoder. Default is 8.
    Inputs:
        word_embeddings (Tensor): Tensor of shape (batch_size, seq_len, embedding_dim) containing word embeddings.
        prosodic_features (Tensor): Tensor of shape (batch_size, seq_len, prosodic_dim) containing prosodic features.
        raw_acoustic_features (Tensor): Tensor of shape (batch_size, seq_len, raw_acoustic_dim) containing raw acoustic features.
        src_key_padding_mask (Tensor, optional): Boolean tensor of shape (batch_size, seq_len) indicating padding positions.
    Returns:
        memory (Tensor): Output from the Transformer encoder of shape (batch_size, seq_len, hidden_dim).
    """
    
    def __init__(self, prosodic_dim, raw_acoustic_dim, embedding_dim, hidden_dim, num_layers, dropout, num_heads=8):
        super(ProjectedEncoder, self).__init__()
        # The projection input dimension is the sum of all three feature dimensions.
        self.projection = nn.Linear(embedding_dim + prosodic_dim + raw_acoustic_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(d_model=hidden_dim, max_len=5000)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, word_embeddings, prosodic_features, raw_acoustic_features, src_key_padding_mask=None):
        # Concatenate the three feature types along the last dimension.
        combined = torch.cat((word_embeddings, prosodic_features, raw_acoustic_features), dim=2)
        projected = self.projection(combined)
        projected = self.positional_encoding(projected)
        projected = self.dropout(projected)
        memory = self.transformer_encoder(projected, src_key_padding_mask=src_key_padding_mask)
        return memory

class TransformerDecoder(nn.Module):
    """
    A Transformer-based decoder module for sequence modeling tasks.
    Args:
        hidden_dim (int): Dimensionality of the hidden representations and embeddings.
        output_dim (int): Dimensionality of the output (not directly used, kept for compatibility).
        num_layers (int): Number of transformer decoder layers.
        dropout (float): Dropout probability applied after embedding and within transformer layers.
        num_classes (int): Number of output classes (including special tokens such as SOS and padding).
        num_heads (int, optional): Number of attention heads in each transformer decoder layer. Default is 8.
    Attributes:
        embedding (nn.Embedding): Embedding layer for target tokens.
        positional_encoding (PositionalEncoding): Module to add positional information to embeddings.
        transformer_decoder (nn.TransformerDecoder): Stacked transformer decoder layers.
        fc_out (nn.Linear): Linear layer projecting decoder outputs to class logits.
        dropout (nn.Dropout): Dropout layer.
    Forward Args:
        memory (Tensor): Encoder output of shape (batch_size, seq_len, hidden_dim).
        tgt (Tensor): Target sequence indices of shape (batch_size, tgt_seq_len).
        tgt_mask (Tensor, optional): Mask for target sequence (tgt_seq_len, tgt_seq_len).
        tgt_key_padding_mask (Tensor, optional): Padding mask for target sequence (batch_size, tgt_seq_len).
    Returns:
        Tensor: Output logits of shape (batch_size, tgt_seq_len, num_classes).
    """
    
    def __init__(self, hidden_dim, output_dim, num_layers, dropout, num_classes, num_heads=8):
        super(TransformerDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.embedding = nn.Embedding(
            num_embeddings=num_classes,  # includes SOS token
            embedding_dim=hidden_dim,
            padding_idx=PADDING_VALUE
        )
        self.positional_encoding = PositionalEncoding(d_model=hidden_dim, max_len=5000)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, memory, tgt, tgt_mask=None, tgt_key_padding_mask=None):
        tgt_embed = self.embedding(tgt)
        tgt_embed = self.positional_encoding(tgt_embed)
        tgt_embed = self.dropout(tgt_embed)
        output = self.transformer_decoder(
            tgt_embed,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        output = self.fc_out(output)
        return output

class Decoder(nn.Module):
    """
    Decoder module that integrates word embeddings with transformer-decoded memory representations.
    Args:
        hidden_dim (int): Dimensionality of the hidden state in the transformer decoder.
        output_dim (int): Output dimensionality of the decoder.
        num_layers (int): Number of layers in the transformer decoder.
        dropout (float): Dropout rate applied in the transformer decoder.
        num_classes (int): Number of output classes.
        num_heads (int, optional): Number of attention heads in the transformer decoder. Default is 8.
        embedding_dim (int, optional): Dimensionality of the input word embeddings. Default is 300.
    Attributes:
        transformer_decoder (nn.Module): The transformer decoder module.
        word_embedding_proj (nn.Linear): Linear layer to project word embeddings to hidden_dim.
        concat_proj (nn.Linear): Linear layer to project concatenated memory and word embeddings.
    Methods:
        forward(memory, tgt, word_embeddings, tgt_mask=None, tgt_key_padding_mask=None):
            Forward pass for the decoder.
            Args:
                memory (Tensor): Encoder output of shape (batch_size, seq_len, hidden_dim).
                tgt (Tensor): Target sequence input for the decoder.
                word_embeddings (Tensor): Word embeddings of shape (batch_size, seq_len, embedding_dim).
                tgt_mask (Tensor, optional): Mask for the target sequence.
                tgt_key_padding_mask (Tensor, optional): Padding mask for the target sequence.
            Returns:
                Tensor: Output of the transformer decoder.
    """
   
    def __init__(self, hidden_dim, output_dim, num_layers, dropout, num_classes, num_heads=8, embedding_dim=300):
        super(Decoder, self).__init__()
        self.transformer_decoder = TransformerDecoder(hidden_dim, output_dim, num_layers, dropout, num_classes, num_heads)
        self.word_embedding_proj = nn.Linear(embedding_dim, hidden_dim)
        self.concat_proj = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, memory, tgt, word_embeddings, tgt_mask=None, tgt_key_padding_mask=None):
        projected_word_embeddings = self.word_embedding_proj(word_embeddings)
        concatenated_memory = torch.cat([memory, projected_word_embeddings], dim=2)
        projected_concatenated_memory = self.concat_proj(concatenated_memory)
        output = self.transformer_decoder(
            projected_concatenated_memory,
            tgt,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        return output

class Seq2Seq(nn.Module):
    """
    A sequence-to-sequence (Seq2Seq) model combining an encoder and decoder for sequence transformation tasks.
    This model implements a transformer-based architecture that processes word embeddings along with 
    prosodic and acoustic features to generate output sequences.
    Args:
        encoder: The encoder module that processes input sequences
        decoder: The decoder module that generates output sequences
    Attributes:
        encoder: The encoder component of the Seq2Seq model
        decoder: The decoder component of the Seq2Seq model
    Methods:
        forward(word_embeddings, prosodic_features, raw_acoustic_features, labels, lengths):
            Performs the forward pass of the Seq2Seq model.
            Args:
                word_embeddings (Tensor): Embedded representation of input words
                prosodic_features (Tensor): Prosodic features of the input sequence
                raw_acoustic_features (Tensor): Raw acoustic features of the input sequence
                labels (Tensor): Target labels for the sequence
                lengths (Tensor): Lengths of the input sequences
            Returns:
                Tensor: Output predictions from the decoder
    """
    
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, word_embeddings, prosodic_features, raw_acoustic_features, labels, lengths):
        src_key_padding_mask = create_padding_mask(labels, padding_value=PADDING_VALUE)
        memory = self.encoder(word_embeddings, prosodic_features, raw_acoustic_features, src_key_padding_mask=src_key_padding_mask)
        batch_size, seq_len = labels.size()
        # Prepare decoder inputs: insert SOS token at the beginning and shift right.
        decoder_inputs = torch.full((batch_size, 1), SOS_IDX, dtype=labels.dtype, device=labels.device)
        decoder_inputs = torch.cat([decoder_inputs, labels[:, :-1]], dim=1)
        tgt_seq_len = decoder_inputs.size(1)
        tgt_mask = generate_causal_mask(tgt_seq_len).to(word_embeddings.device)
        tgt_key_padding_mask = create_padding_mask(decoder_inputs, padding_value=PADDING_VALUE).to(word_embeddings.device)
        output = self.decoder(memory, decoder_inputs, word_embeddings, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        return output

def generate_causal_mask(sz):
    """
    Generate a causal mask to prevent attention to future tokens.
    """
    mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
    return mask

def create_padding_mask(labels, padding_value=0):
    """Create a padding mask for sequences."""
    return (labels == padding_value)

# =============================================================================
# Training, Evaluation, and Testing Functions
# =============================================================================

def train_model(model, iterator, optimizer, criterion, num_classes):
    """
    Trains the model for one epoch using the provided iterator and optimization parameters.
    Args:
        model: The neural network model to train
        iterator: DataLoader containing the training data
        optimizer: The optimizer used for training
        criterion: The loss function
        num_classes: Number of output classes
    Returns:
        float: Average loss value for the epoch
    Notes:
        - Expects input data from iterator in format: (keys, words, word_embeddings, 
          prosodic_features, raw_acoustic_features, labels, lengths)
        - Moves all tensors to the configured device
        - Applies gradient clipping with max norm of 1.0
        - Returns average loss across all batches in the epoch
    """
    
    model.train()
    epoch_loss = 0
    for batch_idx, (keys, words, word_embeddings, prosodic_features, raw_acoustic_features, labels, lengths) in enumerate(iterator):
        word_embeddings = word_embeddings.to(device)
        prosodic_features = prosodic_features.to(device)
        raw_acoustic_features = raw_acoustic_features.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)

        optimizer.zero_grad()
        output = model(word_embeddings, prosodic_features, raw_acoustic_features, labels, lengths)  # [B, T, num_classes]
        output = output.view(-1, num_classes)
        labels_flat = labels.view(-1)
        loss = criterion(output, labels_flat)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

def evaluate_model(model, iterator, criterion, num_classes):
    """
    Evaluates the performance of a model on a given iterator.
    Args:
        model: The neural network model to evaluate.
        iterator: DataLoader containing the evaluation data.
        criterion: Loss function used for model evaluation.
        num_classes: Number of output classes in the classification task.
    Returns:
        tuple: Contains the following metrics:
            - epoch_loss (float): Average loss over the evaluation dataset
            - accuracy (float): Classification accuracy
            - precision (float): Weighted precision score
            - recall (float): Weighted recall score 
            - f1 (float): Weighted F1-score
    Notes:
        - The model is set to evaluation mode during this function
        - Predictions are made without gradient calculation
        - Padding values are excluded from metric calculations
        - All metrics except epoch_loss are calculated using sklearn's implementations
        - Precision, recall and F1 use weighted averaging and zero_division=0
    """
   
    model.eval()
    epoch_loss = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for batch_idx, (keys, words, word_embeddings, prosodic_features, raw_acoustic_features, labels, lengths) in enumerate(iterator):
            word_embeddings = word_embeddings.to(device)
            prosodic_features = prosodic_features.to(device)
            raw_acoustic_features = raw_acoustic_features.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)
            output = model(word_embeddings, prosodic_features, raw_acoustic_features, labels, lengths)
            output = output.view(-1, num_classes)
            labels_flat = labels.view(-1)
            preds = torch.argmax(output, dim=1)
            loss = criterion(output, labels_flat)
            epoch_loss += loss.item()
            non_pad_indices = labels_flat != PADDING_VALUE
            labels_np = labels_flat[non_pad_indices].cpu().numpy() - 1
            preds_np = preds[non_pad_indices].cpu().numpy() - 1
            all_labels.extend(labels_np)
            all_preds.extend(preds_np)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    return epoch_loss / len(iterator), accuracy, precision, recall, f1

def test_model(model, iterator):
    """
    Test the model on the given iterator and compute performance metrics.
    This function evaluates a trained model on test data, generating predictions and 
    comparing them with ground truth labels. It saves detailed results to a text file
    and computes overall accuracy, precision, recall and F1 scores.
    Args:
        model: The trained model to evaluate
        iterator: DataLoader containing test data batches with the following components:
            - keys: Audio file identifiers
            - words: Word sequences
            - word_embeddings: Tensor of word embeddings
            - prosodic_features: Tensor of prosodic features
            - raw_acoustic_features: Tensor of raw acoustic features
            - labels: Ground truth labels
            - lengths: Sequence lengths
    Returns:
        tuple: Two lists containing:
            - all_labels (list): Flattened list of ground truth labels
            - all_preds (list): Flattened list of predicted labels
    Notes:
        - Results are saved to '../outputs/prosody_raw_transformer_multiclass_results.txt'
        - Metrics are computed using weighted averaging to handle class imbalance
        - Padding tokens are removed before computing metrics
    """
    
    model.eval()
    all_labels = []
    all_preds = []
    os.makedirs('../outputs', exist_ok=True)
    results_txt_path = '../outputs/prosody_raw_transformer_multiclass_results.txt'
    with open(results_txt_path, 'w') as file:
        file.write("")
        
    with torch.no_grad():
        for batch_idx, (keys, words, word_embeddings, prosodic_features, raw_acoustic_features, labels, lengths) in enumerate(iterator):
            word_embeddings = word_embeddings.to(device)
            prosodic_features = prosodic_features.to(device)
            raw_acoustic_features = raw_acoustic_features.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)
            logits = model(word_embeddings, prosodic_features, raw_acoustic_features, labels, lengths)
            preds = torch.argmax(logits, dim=2)
            for i in range(labels.size(0)):
                key = keys[i]
                word_sentence = words[i]
                gold_labels = labels[i].cpu().numpy().flatten() - 1
                pred_labels = preds[i].cpu().numpy().flatten() - 1
                # Remove padded tokens (assuming padding value becomes -1 after shift)
                PADDING_VALUE_EVAL = -1
                cleaned_words, cleaned_gold_labels, cleaned_pred_labels = clean_up_sentence(
                    word_sentence, gold_labels, pred_labels, padding_value=PADDING_VALUE_EVAL
                )
                data = {
                    'Word': cleaned_words,
                    'Gold Label': cleaned_gold_labels,
                    'Predicted Label': cleaned_pred_labels
                }
                df = pd.DataFrame(data)
                with open(results_txt_path, 'a') as file:
                    file.write(f"Audio File: {key}\n")
                    file.write(df.to_string(index=False))
                    file.write("\n" + "-" * 50 + "\n")
                all_labels.extend(cleaned_gold_labels)
                all_preds.extend(cleaned_pred_labels)
                
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

def plot_metrics(train_losses, val_losses, val_accuracies, val_precisions, val_recalls, val_f1s):
    """
    Plots and saves training and validation metrics over epochs.
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
    plt.savefig('../outputs/prosody_raw_transformer_multiclass_metrics.png')
    plt.close()

def evaluate_new_set(model, new_dataset_path):
    """
    Evaluate the model on a new dataset.
    """
    new_data = load_data(new_dataset_path)
    new_dataset = ProsodyDataset(new_data)
    new_loader = DataLoader(new_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
    print('\n\nEvaluation on Held Out Set Dataset:')
    all_labels, all_preds = test_model(model, new_loader)
    return all_labels, all_preds

def validate_labels(datasets, num_classes):
    """
    Validates labels across datasets.
    """
    for dataset in datasets:
        for idx, (_, _, _, _, _, labels) in enumerate(dataset):
            invalid_mask = (labels >= num_classes) | (labels < 0)
            if torch.any(invalid_mask):
                invalid_labels = labels[invalid_mask].unique().tolist()
                raise ValueError(f"Found invalid labels {invalid_labels} in dataset at index {idx}. Labels should be in the range [0, {num_classes - 1}].")
    print("All labels are valid.")

def get_all_unique_labels(datasets):
    """
    Get all unique labels from a list of datasets.
    """
    unique_labels = set()
    for dataset in datasets:
        for _, _, _, _, _, labels in dataset:
            unique_labels.update(labels.cpu().numpy().flatten())
    return unique_labels

# =============================================================================
# Main Script
# =============================================================================

if __name__ == "__main__":
    set_seed(42)
    
    # Use the new data files with raw acoustic features
    train_json_path = '../../prosody/data/ambiguous_prosodic_raw_acoustic_ml_features_train_embeddings.json'
    eval_json_path = '../../prosody/data/ambiguous_prosodic_raw_acoustic_ml_features_eval_embeddings.json'
    
    data = load_data(train_json_path)
    dataset_name = "ambiguous_instructions"
    task_name = "prosody_raw_multiclass"
    best_model_filename = f"../models/best-transformer-model-{dataset_name}-{task_name}.pt"
    
    train_data, val_data, test_data = split_data(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
    train_dataset = ProsodyDataset(dict(train_data))
    val_dataset = ProsodyDataset(dict(val_data))
    test_dataset = ProsodyDataset(dict(test_data))
    
    print(f'Train size: {len(train_dataset)}')
    print(f'Validation size: {len(val_dataset)}')
    print(f'Test size: {len(test_dataset)}')
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    # Get a sample batch to compute dimensions.
    sample_batch = next(iter(train_loader))
    # Unpack: keys, words, word_embeddings, prosodic_features, raw_acoustic_features, labels, lengths
    _, _, sample_word_embeddings, sample_prosodic_features, sample_raw_acoustic_features, sample_labels, sample_lengths = sample_batch
    embedding_dim = sample_word_embeddings.shape[2]
    prosodic_dim = sample_prosodic_features.shape[2]
    raw_acoustic_dim = sample_raw_acoustic_features.shape[2]
    
    all_unique_labels = get_all_unique_labels([train_dataset, val_dataset, test_dataset])
    # NUM_CLASSES = (# unique labels) + 2 (for shifting and SOS token)
    NUM_CLASSES = len(all_unique_labels) + 2
    print(f"All unique labels across datasets: {sorted(all_unique_labels)}")
    print(f"Model Training with {NUM_CLASSES} classes")
    
    SOS_IDX = NUM_CLASSES - 1  # Start-of-Sequence token index
    
    # Model hyperparameters
    HIDDEN_DIM = 448
    OUTPUT_DIM = NUM_CLASSES
    NUM_LAYERS = 3
    DROPOUT = 0.25
    NUM_HEADS = 8
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Initialize encoder and decoder with new dimensions.
    encoder = ProjectedEncoder(
        prosodic_dim=prosodic_dim,
        raw_acoustic_dim=raw_acoustic_dim,
        embedding_dim=embedding_dim,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        num_heads=NUM_HEADS
    ).to(device)
    
    decoder = Decoder(
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        num_classes=NUM_CLASSES,
        num_heads=NUM_HEADS,
        embedding_dim=embedding_dim
    ).to(device)
    
    model = Seq2Seq(encoder, decoder).to(device)
    
    # Print model summary
    summary(model)
    
    validate_labels([train_dataset, val_dataset, test_dataset], NUM_CLASSES)
    
    # Compute class weights based on the training dataset.
    class_weights = compute_class_weights(train_dataset, NUM_CLASSES, device)
    
    # Use the computed class weights in the cross entropy loss if desired.
    criterion = nn.CrossEntropyLoss(ignore_index=PADDING_VALUE)  # Optionally: weight=class_weights
    
    optimizer = optim.Adam(model.parameters(), 
                           lr=0.00022254132061744368, 
                           weight_decay=2.255008416912481e-06)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    val_precisions = []
    val_recalls = []
    val_f1s = []
    
    N_EPOCHS = 100
    early_stopping = EarlyStopping(patience=10, min_delta=0.001)
    best_valid_loss = float('inf')
    
    os.makedirs('../models', exist_ok=True)
    
    for epoch in range(N_EPOCHS):
        train_loss = train_model(model, train_loader, optimizer, criterion, num_classes=NUM_CLASSES)
        valid_loss, valid_acc, valid_precision, valid_recall, valid_f1 = evaluate_model(model, val_loader, criterion, num_classes=NUM_CLASSES)
    
        train_losses.append(train_loss)
        val_losses.append(valid_loss)
        val_accuracies.append(valid_acc)
        val_precisions.append(valid_precision)
        val_recalls.append(valid_recall)
        val_f1s.append(valid_f1)
    
        scheduler.step(valid_loss)
    
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), best_model_filename)
    
        print(f'Epoch: {epoch+1:02} | Train Loss: {train_loss:.4f} | Val. Loss: {valid_loss:.4f} | '
              f'Val. Acc: {valid_acc*100:.2f}% | Precision: {valid_precision:.4f} | '
              f'Recall: {valid_recall:.4f} | F1 Score: {valid_f1:.4f}')
    
        early_stopping(valid_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
    
    model.load_state_dict(torch.load(best_model_filename))
    test_loss, test_acc, test_precision, test_recall, test_f1 = evaluate_model(model, test_loader, criterion, num_classes=NUM_CLASSES)
    print(f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.2f}% | '
          f'Precision: {test_precision:.4f} | Recall: {test_recall:.4f} | F1 Score: {test_f1:.4f}')
    
    test_model(model, test_loader)
    plot_metrics(train_losses, val_losses, val_accuracies, val_precisions, val_recalls, val_f1s)
    
    # Evaluate on held-out set.
    true_labels, predicted_labels = evaluate_new_set(model, eval_json_path)
    
    log_dir = "../../prosody/outputs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Define class names manually or compute dynamically.
    class_names = [0, 1, 2]
    class_precision, class_recall, class_f1, class_support = precision_recall_fscore_support(
        true_labels, predicted_labels, average=None, zero_division=0)
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        true_labels, predicted_labels, average='weighted', zero_division=0)
    print("Class Support:", class_support)
    
    confusion_matrices = multilabel_confusion_matrix(true_labels, predicted_labels)
    class_accuracy = []
    for i in range(len(class_names)):
        tn, fp, fn, tp = confusion_matrices[i].ravel()
        acc = (tp + tn) / (tp + tn + fp + fn)
        class_accuracy.append(acc)
    total_support = sum(class_support)
    weighted_accuracy = sum(acc * supp for acc, supp in zip(class_accuracy, class_support)) / total_support
    
    classwise_metrics_path = os.path.join(log_dir, "prosody_raw_classwise_metrics.txt")
    with open(classwise_metrics_path, "w") as f:
        for i, class_name in enumerate(class_names):
            f.write(f"{class_name}:\n")
            f.write(f"  Precision: {class_precision[i]:.4f}\n")
            f.write(f"  Recall: {class_recall[i]:.4f}\n")
            f.write(f"  F1-Score: {class_f1[i]:.4f}\n")
            f.write(f"  Accuracy: {class_accuracy[i]:.4f}\n")
            f.write(f"  Support (True instances in eval data): {class_support[i]}\n")
            f.write("-" * 40 + "\n")
        f.write("\nWeighted Metrics:\n")
        f.write(f"  Weighted Precision: {weighted_precision:.4f}\n")
        f.write(f"  Weighted Recall: {weighted_recall:.4f}\n")
        f.write(f"  Weighted F1-Score: {weighted_f1:.4f}\n")
        f.write(f"  Weighted Accuracy: {weighted_accuracy:.4f}\n")
        f.write("-" * 40 + "\n")
    
    confusion_matrix_path = os.path.join(log_dir, "prosody_raw_confusion_matrix.txt")
    with open(confusion_matrix_path, "w") as f:
        for i, class_name in enumerate(class_names):
            tn, fp, fn, tp = confusion_matrices[i].ravel()
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