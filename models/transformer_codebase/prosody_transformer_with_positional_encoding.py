"""Main script for training and evaluating a transformer-based prosody prediction model.

This script implements a transformer-based sequence-to-sequence model for prosody prediction 
with feature fusion and embedding integration. The model processes both acoustic features 
and word embeddings to predict prosodic labels.

Dependencies:
    - Python 3.x
    - PyTorch
    - NumPy
    - scikit-learn
    - matplotlib
    - pandas

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
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torchinfo import summary

# =============================================================================
# Utility Functions and Constants
# =============================================================================

def set_seed(seed):
    """
    Set the seed for reproducibility.

    This function sets the seed for various random number generators to ensure
    that the results are reproducible. It sets the seed for Python's built-in
    random module, NumPy, and PyTorch. Additionally, it configures PyTorch to
    use deterministic algorithms and disables the benchmark mode in cuDNN.

    Args:
        seed (int): The seed value to be set for the random number generators.
    """
    """Set the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

PADDING_VALUE = 0  # Padding index

class EarlyStopping:
    """
    Early stopping utility to stop training when a monitored metric has stopped improving.

    Attributes:
        patience (int): Number of epochs to wait after the last time the monitored metric improved.
        min_delta (float): Minimum change in the monitored metric to qualify as an improvement.
        best_loss (float or None): The best recorded value of the monitored metric.
        counter (int): Number of epochs since the last improvement.
        early_stop (bool): Flag to indicate whether early stopping should be triggered.

    Methods:
        __call__(val_loss):
            Checks if the validation loss has improved and updates the state accordingly.
            Args:
                val_loss (float): The current value of the monitored metric.
    """
    """Early stopping utility."""
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
    """
    Load data from a JSON file.

    Args:
        json_path (str): The path to the JSON file.

    Returns:
        dict: The data loaded from the JSON file.
    """
    """Load data from a JSON file."""
    with open(json_path, 'r') as file:
        data = json.load(file)
    return data

def split_data(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Splits the data into train, validation, and test sets.

    Parameters:
    data (dict): The data to be split, where keys are data identifiers and values are data points.
    train_ratio (float): The proportion of the data to include in the training set. Default is 0.8.
    val_ratio (float): The proportion of the data to include in the validation set. Default is 0.1.
    test_ratio (float): The proportion of the data to include in the test set. Default is 0.1.

    Returns:
    tuple: A tuple containing three lists: the training set, the validation set, and the test set.

    Raises:
    AssertionError: If the sum of train_ratio, val_ratio, and test_ratio is not equal to 1.
    """
    """Splits the data into train, validation, and test sets."""
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1"
    total = len(data)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    test_size = total - train_size - val_size
    return random_split(list(data.items()), [train_size, val_size, test_size])

def preprocess_text(words):
    """
    Tokenize a list of words using regular expressions.

    This function takes a list of words and tokenizes each word into smaller components
    based on a regular expression pattern. The pattern matches alphanumeric characters,
    apostrophes, and common punctuation marks (.,!?;).

    Args:
        words (list of str): A list of words to be tokenized.

    Returns:
        list of str: A list of tokenized words and punctuation marks.
    """
    """Tokenize words using regex."""
    processed_words = []
    for word in words:
        processed_words.extend(re.findall(r"[\w']+|[.,!?;]", word))
    return processed_words

def clean_up_sentence(words, gold_labels, pred_labels, padding_value):
    """
    Remove padded elements from the sentence and corresponding labels.
        Args:
            words (list): A list of words in the sentence.
            gold_labels (list): A list of gold standard labels corresponding to the words.
            pred_labels (list): A list of predicted labels corresponding to the words.
            padding_value (int): The value used for padding that needs to be removed.

        Returns:
            tuple: A tuple containing three lists:
                - filtered_words (list): The list of words with padding removed.
                - filtered_gold_labels (list): The list of gold labels with padding removed.
                - filtered_pred_labels (list): The list of predicted labels with padding removed.
    
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
    Compute class weights from the training dataset.
    
    Note: The original dataset has 4 unique labels: 0, 1, 2, 3.
          In __getitem__ each label is shifted by +1 so that the labels become 1,2,3,4.
          In addition, NUM_CLASSES is defined as 4 + 2 = 6 (with index 0 reserved for padding and
          index 5 reserved for the SOS token). We compute weights for classes 1-4 and then insert them
          into a weight vector of size 6.
    """
    all_labels = []
    for _, _, _, _, labels in dataset:
        # labels are already shifted, so they are in {1,2,3,4}
        labels_np = labels.numpy().flatten()
        # Exclude padding labels (value 0)
        labels_np = labels_np[labels_np != PADDING_VALUE]
        all_labels.extend(labels_np)
    # Compute balanced weights for classes 1,2,3,4.
    classes = np.array([1, 2, 3, 4])
    computed_weights = compute_class_weight('balanced', classes=classes, y=all_labels)
    # Create a weight vector of length num_classes (6).
    weight_vector = np.ones(num_classes)
    weight_vector[0] = 0.0         # Padding index weight is 0.
    weight_vector[1:5] = computed_weights  # Assign computed weights for classes 1-4.
    weight_vector[num_classes - 1] = 0.0     # SOS token (last index) gets default weight 1.0.
    weight_tensor = torch.tensor(weight_vector, dtype=torch.float).to(device)
    return weight_tensor

# =============================================================================
# Dataset and Collate Function
# =============================================================================

class ProsodyDataset(Dataset):
    """A PyTorch Dataset for prosody data processing.
    This dataset class handles prosodic data including words, word embeddings,
    acoustic features, and prosodic labels.
    Args:
        data (dict): A dictionary containing prosody data entries where each item has:
            - 'words': List of words
            - 'word_embeddings': Word embedding vectors
            - 'features': Acoustic features
            - 'labels': Prosodic labels
    Returns:
        tuple: A tuple containing:
            - key: Entry identifier
            - words: Preprocessed text
            - word_embeddings (torch.Tensor): Word embedding vectors
            - features (torch.Tensor): Acoustic features
            - labels (torch.Tensor): Prosodic labels (shifted by +1 for padding)
    Example:
        >>> dataset = ProsodyDataset(data_dict)
        >>> key, words, embeddings, features, labels = dataset[0]
    """
    
    def __init__(self, data):
        self.entries = list(data.items())

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        key, item = self.entries[idx]
        words = preprocess_text(item['words'])
        word_embeddings = torch.tensor(item['word_embeddings'], dtype=torch.float32)
        features = torch.tensor(item['features'], dtype=torch.float32)
        # Shift labels by +1 so that 0 is reserved for padding.
        labels = torch.tensor(item['labels'], dtype=torch.long) + 1
        return key, words, word_embeddings, features, labels

def collate_fn(batch):
    """
    Collate function for creating batches in a DataLoader.
    This function takes a batch of individual samples and combines them into a single batch by padding sequences
    to the same length.
    Args:
        batch (List[Tuple]): A list of tuples where each tuple contains:
            - keys (str): Identifier for the sample
            - words (List[str]): List of words
            - word_embeddings (torch.Tensor): Word embeddings
            - features (torch.Tensor): Features for each word
            - labels (torch.Tensor): Labels for each word
    Returns:
        Tuple containing:
            - keys (List[str]): List of sample identifiers
            - words (List[List[str]]): List of word lists
            - word_embeddings_padded (torch.Tensor): Padded word embeddings [batch_size, max_seq_len, embedding_dim]
            - features_padded (torch.Tensor): Padded features [batch_size, max_seq_len, feature_dim]
            - labels_padded (torch.Tensor): Padded labels [batch_size, max_seq_len]
            - lengths (torch.Tensor): Original sequence lengths before padding [batch_size]
    """
    
    keys = [item[0] for item in batch]
    words = [item[1] for item in batch]
    word_embeddings = [item[2] for item in batch]
    features = [item[3] for item in batch]
    labels = [item[4] for item in batch]

    word_embeddings_padded = torch.nn.utils.rnn.pad_sequence(word_embeddings, batch_first=True, padding_value=0.0)
    features_padded = torch.nn.utils.rnn.pad_sequence(features, batch_first=True, padding_value=0.0)
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=PADDING_VALUE)
    lengths = torch.tensor([len(f) for f in features])
    return keys, words, word_embeddings_padded, features_padded, labels_padded, lengths

# =============================================================================
# Model Components
# =============================================================================

class PositionalEncoding(nn.Module):
    """Positional Encoding module for Transformer models.
    This class implements positional encoding as described in "Attention Is All You Need"
    (Vaswaniak et al., 2017). It adds positional information to the input embeddings
    using sine and cosine functions of different frequencies.
    Args:
        d_model (int): The dimension of the model/embedding vector
        max_len (int, optional): Maximum sequence length. Defaults to 10000.
    Attributes:
        pe (Tensor): Positional encoding matrix of shape [1, max_len, d_model]
    Forward Args:
        x (Tensor): Input tensor to add positional encoding to.
                    Shape: [batch_size, seq_len, d_model]
    Returns:
        Tensor: Input with positional encoding added.
                Shape: [batch_size, seq_len, d_model]
    """
    
    def __init__(self, d_model, max_len=10000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class ProjectedEncoder(nn.Module):
    """Implements a transformer encoder with projection layer for combined features and word embeddings.
    This module projects concatenated word embeddings and prosodic features into a common space
    before applying positional encoding and transformer encoder layers.
    Args:
        feature_dim (int): Dimension of the prosodic feature vectors
        embedding_dim (int): Dimension of the word embeddings
        hidden_dim (int): Hidden dimension size for the transformer encoder
        num_layers (int): Number of transformer encoder layers
        dropout (float): Dropout probability
        num_heads (int, optional): Number of attention heads. Defaults to 8.
    Attributes:
        projection (nn.Linear): Linear projection layer
        positional_encoding (PositionalEncoding): Adds positional information to the sequences
        transformer_encoder (nn.TransformerEncoder): Multi-layer transformer encoder
        dropout (nn.Dropout): Dropout layer
    Forward Args:
        word_embeddings (torch.Tensor): Word embeddings tensor of shape [batch_size, seq_len, embedding_dim]
        features (torch.Tensor): Prosodic features tensor of shape [batch_size, seq_len, feature_dim]
        src_key_padding_mask (torch.Tensor, optional): Mask for padded elements. Defaults to None.
    Returns:
        torch.Tensor: Encoded sequence of shape [batch_size, seq_len, hidden_dim]
    """
    
    def __init__(self, feature_dim, embedding_dim, hidden_dim, num_layers, dropout, num_heads=8):
        super(ProjectedEncoder, self).__init__()
        self.projection = nn.Linear(feature_dim + embedding_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(d_model=hidden_dim, max_len=5000)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, word_embeddings, features, src_key_padding_mask=None):
        # Concatenate word embeddings and prosody features along the last dimension.
        combined = torch.cat((word_embeddings, features), dim=2)
        projected = self.projection(combined)
        projected = self.positional_encoding(projected)
        projected = self.dropout(projected)
        memory = self.transformer_encoder(projected, src_key_padding_mask=src_key_padding_mask)
        return memory

class TransformerDecoder(nn.Module):
    """TransformerDecoder is a decoder module that implements the transformer architecture.
    This class implements a transformer decoder that processes target sequences and memory
    from an encoder to generate output sequences. It includes embedding layers, positional
    encoding, and multi-head self-attention mechanisms.
    Args:
        hidden_dim (int): Dimension of the model's hidden layers
        output_dim (int): Dimension of the output 
        num_layers (int): Number of transformer decoder layers
        dropout (float): Dropout rate (0 to 1.0)
        num_classes (int): Number of possible output classes (vocabulary size)
        num_heads (int, optional): Number of attention heads. Defaults to 8
    Attributes:
        hidden_dim (int): Dimension of hidden layers
        num_classes (int): Size of output vocabulary
        embedding (nn.Embedding): Token embedding layer
        positional_encoding (PositionalEncoding): Adds positional information to embeddings
        transformer_decoder (nn.TransformerDecoder): Main transformer decoder layers
        fc_out (nn.Linear): Final linear layer for output projection
        dropout (nn.Dropout): Dropout layer
    Methods:
        forward(memory, tgt, tgt_mask=None, tgt_key_padding_mask=None):
            Forward pass of the decoder
            Args:
                memory (Tensor): Output from the encoder [batch_size, src_len, hidden_dim]
                tgt (Tensor): Target sequence [batch_size, tgt_len]
                tgt_mask (Tensor, optional): Mask for target sequence
                tgt_key_padding_mask (Tensor, optional): Mask for padding in target sequence
            Returns:
                Tensor: Output predictions [batch_size, tgt_len, num_classes]
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
    Decoder module that combines transformer decoding with word embeddings.
    This decoder implements a modified transformer decoder architecture that fuses word embeddings
    with the encoder memory before decoding. It projects word embeddings to the hidden dimension,
    concatenates them with encoder memory, and applies transformer decoding.
    Args:
        hidden_dim (int): Dimension of the hidden representations
        output_dim (int): Dimension of the output
        num_layers (int): Number of transformer decoder layers
        dropout (float): Dropout probability
        num_classes (int): Number of output classes
        num_heads (int, optional): Number of attention heads. Defaults to 8
        embedding_dim (int, optional): Dimension of input word embeddings. Defaults to 300
    Attributes:
        transformer_decoder (TransformerDecoder): Main transformer decoder component
        word_embedding_proj (nn.Linear): Projects word embeddings to hidden dimension
        concat_proj (nn.Linear): Projects concatenated features to hidden dimension
    Example:
        decoder = Decoder(512, 256, 6, 0.1, 10)
        output = decoder(memory, tgt, word_embeddings, tgt_mask, tgt_key_padding_mask)
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
    """A sequence-to-sequence model combining an encoder and decoder for prosody prediction.
    This class implements a sequence-to-sequence architecture that processes word embeddings
    and additional features through an encoder-decoder structure to predict prosodic labels.
    Args:
        encoder (nn.Module): The encoder module that processes input sequences
        decoder (nn.Module): The decoder module that generates output sequences
    Attributes:
        encoder (nn.Module): Stores the encoder component
        decoder (nn.Module): Stores the decoder component
    Forward Args:
        word_embeddings (torch.Tensor): Word embedding inputs
        features (torch.Tensor): Additional features for the model
        labels (torch.Tensor): Target labels for training
        lengths (torch.Tensor): Sequence lengths for proper padding handling
    Returns:
        torch.Tensor: The output predictions from the decoder
    Note:
        The forward pass handles padding masks and implements teacher forcing by using
        the ground truth labels shifted right during training. It also generates causal 
        attention masks to ensure the decoder can only attend to previous positions.
    """
    
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, word_embeddings, features, labels, lengths):
        src_key_padding_mask = create_padding_mask(labels, padding_value=PADDING_VALUE)
        memory = self.encoder(word_embeddings, features, src_key_padding_mask=src_key_padding_mask)
        batch_size, seq_len = labels.size()
        # Prepare decoder inputs: insert an SOS token at the beginning and shift right.
        decoder_inputs = torch.full((batch_size, 1), SOS_IDX, dtype=labels.dtype, device=labels.device)
        decoder_inputs = torch.cat([decoder_inputs, labels[:, :-1]], dim=1)
        tgt_seq_len = decoder_inputs.size(1)
        tgt_mask = generate_causal_mask(tgt_seq_len).to(features.device)
        tgt_key_padding_mask = create_padding_mask(decoder_inputs, padding_value=PADDING_VALUE).to(features.device)
        output = self.decoder(memory, decoder_inputs, word_embeddings, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        return output

def generate_causal_mask(sz):
    """Generate a causal mask for transformer self-attention.

    This function creates an upper triangular mask that prevents attention to future tokens
    in the sequence during self-attention. The mask ensures that position i can only attend
    to positions j ≤ i.

    Args:
        sz (int): The size of the square mask matrix to generate.

    Returns:
        torch.Tensor: A boolean tensor of shape (sz, sz) where True values indicate
                     positions that should be masked out (prevented from attending).
                     The lower triangle (including diagonal) contains False values,
                     while the upper triangle contains True values.

    Example:
        >>> mask = generate_causal_mask(3)
        >>> # Results in:
        >>> # [[False, True,  True ],
        >>> #  [False, False, True ],
        >>> #  [False, False, False]]
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
    Train the model for one epoch.

    This function performs training for one epoch by iterating through the data loader,
    computing loss, and updating model parameters through backpropagation.

    Args:
        model: The neural network model to be trained.
        iterator: DataLoader containing training data batches.
        optimizer: The optimizer used for updating model parameters.
        criterion: The loss function used for training.
        num_classes: Integer representing number of output classes.

    Returns:
        float: Average loss value for the epoch (total loss divided by number of batches).

    Note:
        - Assumes data is moved to appropriate device (CPU/GPU) within the function
        - Performs gradient clipping with max norm of 1.0
        - Expects batched inputs with the following structure:
            - keys: Batch of sample identifiers
            - words: Batch of input words
            - word_embeddings: Tensor of word embeddings
            - features: Tensor of additional features
            - labels: Tensor of target labels
            - lengths: Tensor of sequence lengths
    """
    model.train()
    epoch_loss = 0
    for batch_idx, (keys, words, word_embeddings, features, labels, lengths) in enumerate(iterator):
        word_embeddings = word_embeddings.to(device)
        features = features.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)

        optimizer.zero_grad()
        output = model(word_embeddings, features, labels, lengths)  # [B, T, num_classes]
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
    Evaluates the performance of a model on a given dataset iterator.

    This function runs the model in evaluation mode, computing various metrics including
    loss, accuracy, precision, recall, and F1 score. It handles padded sequences by
    excluding padding tokens from the metric calculations.

    Args:
        model: The neural network model to evaluate.
        iterator: DataLoader containing the evaluation data.
        criterion: Loss function used for model evaluation.
        num_classes: Number of classes in the classification task.

    Returns:
        tuple: A tuple containing:
            - float: Average loss per batch
            - float: Accuracy score
            - float: Weighted precision score
            - float: Weighted recall score
            - float: Weighted F1 score

    Notes:
        - The function assumes labels are 1-indexed and converts them to 0-indexed for metric calculation
        - Padding values are excluded from metric calculations
        - Precision, recall and F1 scores use weighted averaging to handle class imbalance
    """
    model.eval()
    epoch_loss = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for batch_idx, (keys, words, word_embeddings, features, labels, lengths) in enumerate(iterator):
            word_embeddings = word_embeddings.to(device)
            features = features.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)
            output = model(word_embeddings, features, labels, lengths)
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
    Tests the model on the given dataset iterator and computes performance metrics.

    Args:
        model: The transformer model to be evaluated
        iterator: DataLoader containing test data batches with keys, words, word embeddings,
                 features, labels and lengths

    Returns:
        tuple:
            all_labels (list): List of all gold (true) labels from test set
            all_preds (list): List of all predicted labels from the model
            
    Details:
        - Sets model to evaluation mode
        - Performs forward pass without gradient calculation
        - Saves detailed results to text file with word-level predictions
        - Saves complete results to JSON file
        - Computes and prints accuracy, precision, recall and F1 metrics
        - Results files are saved in '../outputs/' directory
        - Text output format includes audio file ID and word-level predictions
        - JSON output contains complete prediction details for each audio file

    File outputs:
        - '../outputs/prosody_transformer_multiclass_results.txt': Detailed word-level results
        - '../outputs/prosody_transformer_multiclass_results.json': Complete results in JSON format
    """
    model.eval()
    all_labels = []
    all_preds = []
    all_data = []
    os.makedirs('../outputs', exist_ok=True)
    results_txt_path = '../outputs/prosody_transformer_multiclass_results.txt'
    with open(results_txt_path, 'w') as file:
        file.write("")
    with torch.no_grad():
        for batch_idx, (keys, words, word_embeddings, features, labels, lengths) in enumerate(iterator):
            word_embeddings = word_embeddings.to(device)
            features = features.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)
            logits = model(word_embeddings, features, labels, lengths)
            preds = torch.argmax(logits, dim=2)
            for i in range(features.size(0)):
                key = keys[i]
                word_sentence = words[i]
                gold_labels = labels[i].cpu().numpy().flatten() - 1
                pred_labels = preds[i].cpu().numpy().flatten() - 1
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
                data_json = {
                    'audio_file': key,
                    'words': cleaned_words,
                    'gold_labels': cleaned_gold_labels,
                    'predicted_labels': cleaned_pred_labels
                }
                all_data.append(data_json)
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
    results_json_path = os.path.join('../outputs/prosody_transformer_multiclass_results.json')
    with open(results_json_path, 'w') as json_file:
        json.dump(all_data, json_file, indent=4)
    return all_labels, all_preds

def plot_metrics(train_losses, val_losses, val_accuracies, val_precisions, val_recalls, val_f1s):
    """
    Plots and saves training and validation metrics over epochs.

    This function creates a figure with 5 subplots showing the progression of different
    metrics during model training. The plots include training/validation loss, validation
    accuracy, precision, recall and F1 score. The resulting figure is saved as a PNG file.

    Parameters:
        train_losses (list): List of training loss values for each epoch
        val_losses (list): List of validation loss values for each epoch  
        val_accuracies (list): List of validation accuracy values for each epoch
        val_precisions (list): List of validation precision values for each epoch
        val_recalls (list): List of validation recall values for each epoch
        val_f1s (list): List of validation F1 score values for each epoch

    Returns:
        None. The function saves the plot to '../outputs/transformer_multiclass_metrics.png'

    Example:
        >>> plot_metrics([0.5, 0.3], [0.4, 0.2], [0.8, 0.9], [0.7, 0.8], [0.7, 0.8], [0.7, 0.8])
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
    plt.savefig('../outputs/transformer_multiclass_metrics.png')
    plt.close()

def evaluate_new_set(model, new_dataset_path):
    """
    Evaluates the model's performance on a new dataset.

    This function loads a new dataset, creates a DataLoader, and tests the model's performance
    on this held-out set. It returns both the true labels and model predictions.

    Args:
        model: The trained model to be evaluated
        new_dataset_path (str): Path to the new dataset file

    Returns:
        tuple: A tuple containing:
            - all_labels: True labels from the dataset
            - all_preds: Model predictions for the dataset

    Example:
        labels, predictions = evaluate_new_set(trained_model, "path/to/new/data.csv")
    """
    new_data = load_data(new_dataset_path)
    new_dataset = ProsodyDataset(new_data)
    new_loader = DataLoader(new_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
    print('\n\nEvaluation on Held Out Set Dataset:')
    all_labels, all_preds = test_model(model, new_loader)
    return all_labels, all_preds

def validate_labels(datasets, num_classes):
    """
    Validates labels across multiple datasets to ensure they fall within the valid range.

    This function checks if all labels in the provided datasets are within the valid range
    [0, num_classes - 1]. It raises a ValueError if any invalid labels are found.

    Args:
        datasets (list): A list of datasets, where each dataset yields tuples containing labels
                        at the 5th position (index 4).
        num_classes (int): The number of valid classes. Valid labels should be in range
                          [0, num_classes - 1].

    Raises:
        ValueError: If any labels are found that are either negative or >= num_classes,
                   including the invalid labels and their dataset index in the error message.

    Prints:
        "All labels are valid." if all labels across all datasets are within valid range.
    """
    for dataset in datasets:
        for idx, (_, _, _, _, labels) in enumerate(dataset):
            invalid_mask = (labels >= num_classes) | (labels < 0)
            if torch.any(invalid_mask):
                invalid_labels = labels[invalid_mask].unique().tolist()
                raise ValueError(f"Found invalid labels {invalid_labels} in dataset at index {idx}. Labels should be in the range [0, {num_classes - 1}].")
    print("All labels are valid.")

def get_all_unique_labels(datasets):
    """
    Get all unique labels from a list of datasets.

    Args:
        datasets (list): A list of datasets where each dataset contains tuples of 
                         (data, ..., labels) where labels is a tensor containing label values.

    Returns:
        set: A set of unique label values found across all datasets.

    Notes:
        - Labels are extracted from the 5th element (index 4) of each tuple in the datasets
        - Labels are converted from tensor to numpy array before processing
        - CPU values are used when converting tensors
    """
    unique_labels = set()
    for dataset in datasets:
        for _, _, _, _, labels in dataset:
            unique_labels.update(labels.cpu().numpy().flatten())
    return unique_labels

# =============================================================================
# Main Script
# =============================================================================

if __name__ == "__main__":
    set_seed(42)
    json_path = '../../prosody/data/ambiguous_prosody_multi_label_features_train_embeddings.json'
    data = load_data(json_path)

    dataset_name = "ambiguous_instructions"
    task_name = "prosody_multiclass"
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

    sample_batch = next(iter(train_loader))
    _, _, sample_word_embeddings, sample_features, sample_labels, sample_lengths = sample_batch
    feature_dim = sample_features.shape[2]
    embedding_dim = sample_word_embeddings.shape[2]

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

    # Initialize encoder and decoder
    encoder = ProjectedEncoder(
        feature_dim=feature_dim,
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

    # Use the computed class weights in the cross entropy loss.
    criterion = nn.CrossEntropyLoss(ignore_index=PADDING_VALUE, ) #weight=class_weights

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
    CLIP = 1

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

    eval_json = "../../prosody/data/ambiguous_prosody_multi_label_features_eval_embeddings.json"
    true_labels, predicted_labels = evaluate_new_set(model, eval_json)

    log_dir = "../../prosody/outputs"
    os.makedirs(log_dir, exist_ok=True)

    # Define class names manually (or compute dynamically)
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

    classwise_metrics_path = os.path.join(log_dir, "prosody_classwise_metrics.txt")
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
    
    confusion_matrix_path = os.path.join(log_dir, "prosody_confusion_matrix.txt")
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