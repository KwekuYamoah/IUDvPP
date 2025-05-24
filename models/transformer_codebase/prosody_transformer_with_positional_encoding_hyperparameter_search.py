"""
Main script for finding the best of parameters for a transformer-based prosody prediction model.

This version of the script trains with prosodic features and word embeddings.
The architecture remains essentially the same except that the encoder now concatenates
word embeddings, prosodic features, before projection.


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
import pandas as pd
import matplotlib.pyplot as plt
import pynvml  # For GPU monitoring
import optuna  # For hyperparameter tuning

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import multilabel_confusion_matrix, precision_recall_fscore_support

# Torch imports
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import amp
from torchinfo import summary

# Initialize pynvml for GPU monitoring
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Assuming single GPU; change index if multiple GPUs

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Define constants
PADDING_VALUE = 0  # Use 0 as the padding index
SOS_IDX = None  # Will be defined later based on NUM_CLASSES
MAX_GPU_TEMP = 75  # Celsius
MAX_GPU_MEM = 14 * 1024  # MB (14 GB)

def get_gpu_status():
    """
    Retrieves the current GPU temperature and memory usage.

    Returns:
        temp (int): Current GPU temperature in Celsius.
        mem_used (int): Memory used in MB.
    """
    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    mem_used = mem_info.used // 1024 // 1024  # Convert bytes to MB
    return temp, mem_used

def set_seed(seed):
    """
    Set the seed for generating random numbers to ensure reproducibility.

    Args:
        seed (int): The seed value to use for random number generation.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class EarlyStopping:
    """
    Implements early stopping to terminate training when validation loss stops improving.

    Attributes:
        patience (int): Number of epochs to wait after last improvement before stopping.
        min_delta (float): Minimum change in the monitored loss to qualify as an improvement.
        best_loss (float or None): Best validation loss observed so far.
        counter (int): Number of consecutive epochs without improvement.
        early_stop (bool): Flag indicating whether early stopping condition has been met.

    Methods:
        __call__(val_loss):
            Call method to update early stopping status based on the current validation loss.
            Args:
                val_loss (float): Current epoch's validation loss.
            Returns:
                None. Updates internal state and sets early_stop flag if stopping criteria are met.
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
    """ Loads data from a JSON file."""
    with open(json_path, 'r') as file:
        data = json.load(file)
    return data

def split_data(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """ Splits data into training, validation, and test sets."""
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1"
    total = len(data)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    test_size = total - train_size - val_size
    return random_split(list(data.items()), [train_size, val_size, test_size])

def preprocess_text(words):
    """ Preprocesses text by tokenizing and removing unwanted characters."""
    processed_words = []
    for word in words:
        processed_words.extend(re.findall(r"[\w']+|[.,!?;]", word))
    return processed_words

def clean_up_sentence(words, gold_labels, pred_labels, padding_value):
    """ Cleans up the sentence by removing padding values."""
    filtered_words = []
    filtered_gold_labels = []
    filtered_pred_labels = []

    for i in range(len(words)):
        if gold_labels[i] != padding_value:
            filtered_words.append(words[i])
            filtered_gold_labels.append(int(gold_labels[i]))
            filtered_pred_labels.append(int(pred_labels[i]))
    return filtered_words, filtered_gold_labels, filtered_pred_labels

class ProsodyDataset(Dataset):
    """
    A PyTorch Dataset for prosody data, providing access to word-level features and labels.

    Args:
        data (dict): A dictionary where each key maps to an item containing:
            - 'words' (str or list): The text or list of words to preprocess.
            - 'word_embeddings' (array-like): Precomputed word embeddings.
            - 'features' (array-like): Additional feature vectors.
            - 'labels' (array-like): Integer labels for each word.

    Methods:
        __len__(): Returns the number of entries in the dataset.
        __getitem__(idx): Retrieves the key, preprocessed words, word embeddings, features, and labels (shifted by +1) for the given index.

    Returns:
        tuple: (key, words, word_embeddings, features, labels)
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
        labels = torch.tensor(item['labels'], dtype=torch.long) + 1  # Shift labels by +1
        return key, words, word_embeddings, features, labels

def collate_fn(batch):
    """
    Collate function for batching data samples in a DataLoader.
    Args:
        batch (list): A list of tuples, where each tuple contains:
            - key: Identifier for the sample.
            - words: List of words in the sample.
            - word_embeddings (Tensor): Word embedding tensor for the sample (seq_len, embedding_dim).
            - features (Tensor): Feature tensor for the sample (seq_len, feature_dim).
            - labels (Tensor): Label tensor for the sample (seq_len,).
    Returns:
        tuple: A tuple containing:
            - keys (list): List of sample identifiers.
            - words (list): List of word lists for each sample.
            - word_embeddings_padded (Tensor): Padded word embeddings (batch_size, max_seq_len, embedding_dim).
            - features_padded (Tensor): Padded features (batch_size, max_seq_len, feature_dim).
            - labels_padded (Tensor): Padded labels (batch_size, max_seq_len).
            - lengths (Tensor): Original sequence lengths for each sample (batch_size,).
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

# Positional Encoding for Transformer
class PositionalEncoding(nn.Module):
    """
    Positional encoding module for adding positional information to the input embeddings.

    Args:
        d_model (int): The dimension of the model.
        max_len (int, optional): The maximum length of the input sequences. Default is 10000.

    Attributes:
        pe (torch.Tensor): The positional encoding matrix of shape (1, max_len, d_model).

    Methods:
        forward(x):
            Adds positional encoding to the input tensor.

            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).

            Returns:
                torch.Tensor: The input tensor with positional encoding added.
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
        # x: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return x



class TransformerEncoder(nn.Module):
    """
    Transformer Encoder module for processing sequential data.

    Args:
        feature_dim (int): The dimension of the input features.
        hidden_dim (int): The dimension of the hidden layer.
        num_layers (int): The number of transformer encoder layers.
        dropout (float): The dropout rate.
        num_heads (int, optional): The number of attention heads. Default is 8.

    Attributes:
        input_fc (nn.Linear): Linear layer to transform input features to hidden dimension.
        positional_encoding (PositionalEncoding): Positional encoding to add positional information to the input.
        encoder_layer (nn.TransformerEncoderLayer): A single transformer encoder layer.
        transformer_encoder (nn.TransformerEncoder): Stack of transformer encoder layers.
        dropout (nn.Dropout): Dropout layer.

    Methods:
        forward(features, src_key_padding_mask=None):
            Forward pass through the transformer encoder.
            Args:
                features (torch.Tensor): Input features of shape [batch_size, seq_len, feature_dim].
                src_key_padding_mask (torch.Tensor, optional): Mask for padding tokens. Default is None.
            Returns:
                torch.Tensor: Output memory of shape [batch_size, seq_len, hidden_dim].
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
        src = self.input_fc(features)  # [batch_size, seq_len, hidden_dim]
        src = self.positional_encoding(src)  # [batch_size, seq_len, hidden_dim]
        src = self.dropout(src)
        memory = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)  # [batch_size, seq_len, hidden_dim]
        return memory



class AttentionLayer(nn.Module):
    """
    AttentionLayer computes attention weights by evaluating the relevance of encoder outputs relative to the current hidden state.

    Args:
        hidden_dim (int): The dimension of the hidden state.

    Attributes:
        query_linear (nn.Linear): Linear layer to transform the query.
        key_linear (nn.Linear): Linear layer to transform the keys.
        value_linear (nn.Linear): Linear layer to transform the values.
        scale (float): Scaling factor for attention scores.

    Methods:
        forward(query, keys, values, mask=None):
            Computes attention weights and returns the context vector.

            Args:
                query (torch.Tensor): Query tensor of shape [batch_size, hidden_dim].
                keys (torch.Tensor): Keys tensor of shape [batch_size, seq_len, hidden_dim].
                values (torch.Tensor): Values tensor of shape [batch_size, seq_len, hidden_dim].
                mask (torch.Tensor, optional): Mask tensor of shape [batch_size, seq_len]. Default is None.

            Returns:
                torch.Tensor: Context vector of shape [batch_size, hidden_dim].
                torch.Tensor: Attention weights of shape [batch_size, seq_len].
    """
    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.query_linear = nn.Linear(hidden_dim, hidden_dim)
        self.key_linear = nn.Linear(hidden_dim, hidden_dim)
        self.value_linear = nn.Linear(hidden_dim, hidden_dim)
        self.scale = hidden_dim ** 0.5

    def forward(self, query, keys, values, mask=None):
        # query: [batch_size, hidden_dim]
        # keys: [batch_size, seq_len, hidden_dim]
        # values: [batch_size, seq_len, hidden_dim]
        Q = self.query_linear(query)  # [batch_size, hidden_dim]
        K = self.key_linear(keys)     # [batch_size, seq_len, hidden_dim]
        V = self.value_linear(values) # [batch_size, seq_len, hidden_dim]

        # Compute attention scores
        scores = torch.bmm(K, Q.unsqueeze(2)).squeeze(2) / self.scale  # [batch_size, seq_len]

        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))

        # Compute attention weights
        attn_weights = torch.softmax(scores, dim=1)  # [batch_size, seq_len]

        # Compute context vector
        context = torch.bmm(attn_weights.unsqueeze(1), V).squeeze(1)  # [batch_size, hidden_dim]

        return context, attn_weights



class MultiAttention(nn.Module):
    """
    MultiAttention implements a multi-head self-attention mechanism by aggregating outputs from multiple AttentionLayer instances.

    Args:
        hidden_dim (int): The dimension of the hidden state.
        num_heads (int): The number of attention heads.

    Attributes:
        attention_layers (nn.ModuleList): A list of AttentionLayer instances.

    Methods:
        forward(query, keys, values, mask=None):
            Applies multiple attention layers and aggregates their outputs.

            Args:
                query (torch.Tensor): Query tensor of shape [batch_size, hidden_dim].
                keys (torch.Tensor): Keys tensor of shape [batch_size, seq_len, hidden_dim].
                values (torch.Tensor): Values tensor of shape [batch_size, seq_len, hidden_dim].
                mask (torch.Tensor, optional): Mask tensor of shape [batch_size, seq_len]. Default is None.

            Returns:
                torch.Tensor: Aggregated context vector of shape [batch_size, hidden_dim].
                list of torch.Tensor: List of attention weights from each head.
    """
    def __init__(self, hidden_dim, num_heads):
        super(MultiAttention, self).__init__()
        self.num_heads = num_heads
        self.attention_layers = nn.ModuleList([AttentionLayer(hidden_dim) for _ in range(num_heads)])

    def forward(self, query, keys, values, mask=None):
        contexts = []
        attn_weights_all = []
        for attn in self.attention_layers:
            context, attn_weights = attn(query, keys, values, mask=mask)
            contexts.append(context)
            attn_weights_all.append(attn_weights)
        # Concatenate all context vectors
        aggregated_context = torch.cat(contexts, dim=1)  # [batch_size, hidden_dim * num_heads]
        return aggregated_context, attn_weights_all


# Attention-Based Fusion
class AttentionBasedFusion(nn.Module):
    """
    AttentionBasedFusion applies attention-based weighting to fuse combined features, emphasizing more informative parts of the input sequence.

    Args:
        hidden_dim (int): The dimension of the hidden state.
        num_heads (int): The number of attention heads.

    Attributes:
        multi_attention (MultiAttention): Multi-head attention mechanism.
        fusion_linear (nn.Linear): Linear layer to fuse the aggregated context.

    Methods:
        forward(query, keys, values, mask=None):
            Applies multi-head attention and fuses the context vectors.

            Args:
                query (torch.Tensor): Query tensor of shape [batch_size, hidden_dim].
                keys (torch.Tensor): Keys tensor of shape [batch_size, seq_len, hidden_dim].
                values (torch.Tensor): Values tensor of shape [batch_size, seq_len, hidden_dim].
                mask (torch.Tensor, optional): Mask tensor of shape [batch_size, seq_len]. Default is None.

            Returns:
                torch.Tensor: Fused context vector of shape [batch_size, hidden_dim].
    """
    def __init__(self, hidden_dim, num_heads):
        super(AttentionBasedFusion, self).__init__()
        self.multi_attention = MultiAttention(hidden_dim, num_heads)
        self.fusion_linear = nn.Linear(hidden_dim * num_heads, hidden_dim)

    def forward(self, query, keys, values, mask=None):
        aggregated_context, attn_weights_all = self.multi_attention(query, keys, values, mask=mask)
        fused_context = self.fusion_linear(aggregated_context)  # [batch_size, hidden_dim]
        return fused_context


# Projected Encoder: Projects combined features into lower-dimensional space and applies Transformer
class ProjectedEncoder(nn.Module):
    """
    ProjectedEncoder projects combined features (prosody features concatenated with word embeddings)
    into a lower-dimensional space using a linear layer followed by a Transformer layer.
    It processes the input sequences to capture context from past states.

    Args:
        feature_dim (int): The dimension of the prosody features.
        embedding_dim (int): The dimension of the word embeddings.
        hidden_dim (int): The dimension of the hidden layer after projection.
        num_layers (int): The number of transformer encoder layers.
        dropout (float): The dropout rate.
        num_heads (int): The number of attention heads.

    Attributes:
        projection (nn.Linear): Linear layer to project concatenated features.
        positional_encoding (PositionalEncoding): Positional encoding for the projected features.
        transformer_encoder (nn.TransformerEncoder): Transformer encoder to capture contextual information.
        dropout (nn.Dropout): Dropout layer.

    Methods:
        forward(word_embeddings, features, src_key_padding_mask=None):
            Forward pass through the ProjectedEncoder.

            Args:
                word_embeddings (torch.Tensor): Tensor of word embeddings [batch_size, seq_len, embedding_dim].
                features (torch.Tensor): Tensor of prosody features [batch_size, seq_len, feature_dim].
                src_key_padding_mask (torch.Tensor, optional): Padding mask [batch_size, seq_len].

            Returns:
                torch.Tensor: Encoded memory [batch_size, seq_len, hidden_dim].
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
        # Concatenate word embeddings and prosody features
        combined = torch.cat((word_embeddings, features), dim=2)  # [batch_size, seq_len, embedding_dim + feature_dim]
        projected = self.projection(combined)  # [batch_size, seq_len, hidden_dim]
        projected = self.positional_encoding(projected)
        projected = self.dropout(projected)
        memory = self.transformer_encoder(projected, src_key_padding_mask=src_key_padding_mask)  # [batch_size, seq_len, hidden_dim]
        return memory


# Transformer Decoder with Causal and Padding Masking
class TransformerDecoder(nn.Module):
    """
    TransformerDecoder is a neural network module that implements a transformer-based decoder.

    Args:
        hidden_dim (int): The dimension of the hidden layers.
        output_dim (int): The dimension of the output.
        num_layers (int): The number of decoder layers.
        dropout (float): The dropout rate.
        num_classes (int, optional): The number of classes for the embedding and output layers.
        num_heads (int, optional): The number of attention heads. Default is 8.

    Attributes:
        hidden_dim (int): The dimension of the hidden layers.
        num_classes (int): The number of classes for the embedding and output layers.
        embedding (nn.Embedding): The embedding layer.
        positional_encoding (PositionalEncoding): Positional encoding for the target embeddings.
        transformer_decoder (nn.TransformerDecoder): The transformer decoder composed of multiple decoder layers.
        fc_out (nn.Linear): The final fully connected layer that maps the hidden states to the output classes.

    Methods:
        forward(memory, tgt, tgt_mask=None, tgt_key_padding_mask=None):
            Performs a forward pass through the transformer decoder.

            Args:
                memory (Tensor): The memory tensor from the encoder. Shape [batch_size, seq_len, hidden_dim].
                tgt (Tensor): The target sequence tensor. Shape [batch_size, seq_len].
                tgt_mask (Tensor, optional): The target mask tensor. Default is None.
                tgt_key_padding_mask (Tensor, optional): The target key padding mask tensor. Default is None.

            Returns:
                Tensor: The output tensor. Shape [batch_size, seq_len, num_classes].
    """
    def __init__(self, hidden_dim, output_dim, num_layers, dropout, num_classes, num_heads=8):
        super(TransformerDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.embedding = nn.Embedding(
            num_embeddings=num_classes,  # Updated to include SOS token
            embedding_dim=hidden_dim,
            padding_idx=PADDING_VALUE
        )
        self.positional_encoding = PositionalEncoding(d_model=hidden_dim, max_len=5000)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True  # Align with data format
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(hidden_dim, num_classes)  # Ensure output_dim == num_classes
        self.dropout = nn.Dropout(dropout)

    def forward(self, memory, tgt, tgt_mask=None, tgt_key_padding_mask=None):
        tgt_embed = self.embedding(tgt)  # [batch_size, seq_len, hidden_dim]
        tgt_embed = self.positional_encoding(tgt_embed)
        tgt_embed = self.dropout(tgt_embed)
        output = self.transformer_decoder(
            tgt_embed,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )  # [batch_size, seq_len, hidden_dim]
        output = self.fc_out(output)  # [batch_size, seq_len, num_classes]
        return output


# Attention-Based Decoder incorporating Attention-Based Fusion
class Decoder(nn.Module):
    """
    Decoder utilizes the fused features and word embeddings to generate predictions.
    It incorporates the attention mechanism to focus on relevant encoder outputs and processes
    the information through another transformer.

    Args:
        hidden_dim (int): The dimension of the hidden layers.
        output_dim (int): The dimension of the output.
        num_layers (int): The number of transformer decoder layers.
        dropout (float): The dropout rate.
        num_classes (int, optional): The number of classes for the embedding and output layers.
        num_heads (int, optional): The number of attention heads. Default is 8.
        embedding_dim (int): The dimension of the word embeddings.

    Attributes:
        attention_fusion (AttentionBasedFusion): Attention-based fusion mechanism.
        transformer_decoder (TransformerDecoder): Transformer-based decoder.
        word_embedding_proj (nn.Linear): Linear layer to project word embeddings to hidden_dim.
        concat_proj (nn.Linear): Linear layer to project concatenated memory back to hidden_dim.

    Methods:
        forward(memory, tgt, fused_memory, word_embeddings, tgt_mask=None, tgt_key_padding_mask=None):
            Forward pass through the Decoder.

            Args:
                memory (torch.Tensor): Encoder outputs [batch_size, seq_len, hidden_dim].
                tgt (torch.Tensor): Target labels [batch_size, seq_len].
                fused_memory (torch.Tensor): Fused features from AttentionBasedFusion [batch_size, hidden_dim].
                word_embeddings (torch.Tensor): Word embeddings [batch_size, seq_len, embedding_dim].
                tgt_mask (torch.Tensor, optional): Causal mask for target sequence. Default is None.
                tgt_key_padding_mask (torch.Tensor, optional): Padding mask for target sequence. Default is None.

            Returns:
                torch.Tensor: Output logits [batch_size, seq_len, num_classes].
    """
    def __init__(self, hidden_dim, output_dim, num_layers, dropout, num_classes, num_heads=8, embedding_dim=300):
        super(Decoder, self).__init__()
        self.attention_fusion = AttentionBasedFusion(hidden_dim, num_heads)
        self.transformer_decoder = TransformerDecoder(hidden_dim, output_dim, num_layers, dropout, num_classes, num_heads)
        self.word_embedding_proj = nn.Linear(embedding_dim, hidden_dim)  # Projects word_embeddings to hidden_dim
        self.concat_proj = nn.Linear(hidden_dim * 2, hidden_dim)       # Projects concatenated memory back to hidden_dim

    def forward(self, memory, tgt, fused_memory, word_embeddings, tgt_mask=None, tgt_key_padding_mask=None):
        """
        Forward pass through the Decoder.

        Args:
            memory (torch.Tensor): Encoder outputs [batch_size, seq_len, hidden_dim].
            tgt (torch.Tensor): Target labels [batch_size, seq_len].
            fused_memory (torch.Tensor): Fused features from AttentionBasedFusion [batch_size, hidden_dim].
            word_embeddings (torch.Tensor): Word embeddings [batch_size, seq_len, embedding_dim].
            tgt_mask (torch.Tensor, optional): Causal mask for target sequence. Default is None.
            tgt_key_padding_mask (torch.Tensor, optional): Padding mask for target sequence. Default is None.

        Returns:
            torch.Tensor: Output logits [batch_size, seq_len, num_classes].
        """
        # Apply attention-based fusion to obtain the fused context
        fused_context = self.attention_fusion(fused_memory, memory, memory, mask=None)  # [batch_size, hidden_dim]

        # Project word_embeddings to hidden_dim
        projected_word_embeddings = self.word_embedding_proj(word_embeddings)  # [batch_size, seq_len, hidden_dim]

        # Concatenate memory and projected_word_embeddings along feature dimension
        # Resulting shape: [batch_size, seq_len, hidden_dim * 2]
        concatenated_memory = torch.cat([memory, projected_word_embeddings], dim=2)  # [batch_size, seq_len, hidden_dim * 2]

        # Project concatenated memory back to hidden_dim
        projected_concatenated_memory = self.concat_proj(concatenated_memory)  # [batch_size, seq_len, hidden_dim]

        # Pass the projected concatenated memory and target inputs through the transformer decoder
        output = self.transformer_decoder(
            projected_concatenated_memory,
            tgt,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )  # [batch_size, seq_len, num_classes]

        return output


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, word_embeddings, features, labels, lengths):
        # Generate padding mask for encoder (source)
        src_key_padding_mask = create_padding_mask(labels, padding_value=PADDING_VALUE)

        # Pass through encoder
        memory = self.encoder(word_embeddings, features, src_key_padding_mask=src_key_padding_mask)  # [batch_size, seq_len, hidden_dim]

        # Prepare decoder inputs by shifting labels and adding SOS token
        batch_size, seq_len = labels.size()
        decoder_inputs = torch.full((batch_size, 1), SOS_IDX, dtype=labels.dtype, device=labels.device)
        decoder_inputs = torch.cat([decoder_inputs, labels[:, :-1]], dim=1)  # Shifted labels

        # Generate causal mask for decoder
        tgt_seq_len = decoder_inputs.size(1)
        tgt_mask = generate_causal_mask(tgt_seq_len).to(features.device)  # [seq_len, seq_len]

        # Generate padding mask for decoder (target)
        tgt_key_padding_mask = create_padding_mask(decoder_inputs, padding_value=PADDING_VALUE).to(features.device)  # [batch_size, seq_len]

        # Fused features for attention-based fusion (can be a global context or any other representation)
        # Here, we'll use the mean of the memory as a simple fused representation
        fused_memory = memory.mean(dim=1)  # [batch_size, hidden_dim]

        # Pass through decoder, including word_embeddings
        outputs = self.decoder(
            memory,
            decoder_inputs,
            fused_memory,
            word_embeddings,        # Passing word_embeddings to the Decoder
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )  # [batch_size, seq_len, num_classes]

        return outputs

def generate_causal_mask(sz):
    mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()  # Upper triangular part (excluding diagonal)
    return mask  # [sz, sz]

def create_padding_mask(labels, padding_value=0):
    mask = (labels == padding_value)
    return mask  # [batch_size, seq_len]

def train_model(model, iterator, optimizer, criterion, scaler, accumulation_steps=1):
    model.train()
    epoch_loss = 0
    for batch_idx, (keys, words, word_embeddings, features, labels, lengths) in enumerate(iterator):
        word_embeddings = word_embeddings.to(device, non_blocking=True)
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)

        with amp.autocast(device_type='cuda'):
            output = model(word_embeddings, features, labels, lengths)  # [batch_size, seq_len, num_classes]
            output = output.view(-1, model.decoder.transformer_decoder.num_classes)  # [batch_size * seq_len, num_classes]
            labels_flat = labels.view(-1)  # [batch_size * seq_len]
            loss = criterion(output, labels_flat) / accumulation_steps

        scaler.scale(loss).backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        epoch_loss += loss.item() * accumulation_steps

        # GPU Monitoring
        temp, mem_used = get_gpu_status()
        if temp > MAX_GPU_TEMP or mem_used > MAX_GPU_MEM:
            print(f"GPU constraints exceeded: Temp={temp}°C, Memory Used={mem_used}MB")
            return None  # Indicate that the trial should be pruned

    return epoch_loss / len(iterator)

def evaluate_model(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for batch_idx, (keys, words, word_embeddings, features, labels, lengths) in enumerate(iterator):
            word_embeddings = word_embeddings.to(device, non_blocking=True)
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)

            with amp.autocast(device_type='cuda'):
                output = model(word_embeddings, features, labels, lengths)  # [batch_size, seq_len, num_classes]
                output = output.view(-1, model.decoder.transformer_decoder.num_classes)  # [batch_size * seq_len, num_classes]
                labels_flat = labels.view(-1)  # [batch_size * seq_len]
                loss = criterion(output, labels_flat)
                epoch_loss += loss.item()

                preds = torch.argmax(output, dim=1)  # [batch_size * seq_len]

                non_pad_indices = labels_flat != PADDING_VALUE
                labels_np = labels_flat[non_pad_indices].cpu().numpy() - 1  # Adjust labels back by -1
                preds_np = preds[non_pad_indices].cpu().numpy() - 1    # Adjust preds back by -1
                all_labels.extend(labels_np)
                all_preds.extend(preds_np)

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    return epoch_loss / len(iterator), accuracy, precision, recall, f1

def test_model(model, iterator, criterion, num_classes):
    """
    Evaluates the model on the test set and writes detailed results to files.
    """
    model.eval()
    epoch_loss = 0
    all_labels = []
    all_preds = []
    all_data = []

    os.makedirs('../outputs', exist_ok=True)
    results_txt_path = '../outputs/prosody_transformer_multiclass_results.txt'
    with open(results_txt_path, 'w') as file:
        file.write("")

    with torch.no_grad():
        for batch_idx, (keys, words, word_embeddings, features, labels, lengths) in enumerate(iterator):
            word_embeddings = word_embeddings.to(device, non_blocking=True)
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)

            with amp.autocast(device_type='cuda'):
                logits = model(word_embeddings, features, labels, lengths)  # [batch_size, seq_len, num_classes]
                preds = torch.argmax(logits, dim=2)  # [batch_size, seq_len]

            output = logits.view(-1, num_classes)
            labels_flat = labels.view(-1)
            loss = criterion(output, labels_flat)
            epoch_loss += loss.item()

            for i in range(features.size(0)):
                key = keys[i]
                word_sentence = words[i]
                gold_labels = labels[i].cpu().numpy().flatten()
                pred_labels = preds[i].cpu().numpy().flatten()

                gold_labels = gold_labels - 1
                pred_labels = pred_labels - 1
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
    new_data = load_data(new_dataset_path)
    new_dataset = ProsodyDataset(new_data)
    new_loader = DataLoader(new_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn,
                            num_workers=7, pin_memory=True)

    print('\n\nEvaluation on Held Out Set Dataset:')
    all_labels, all_preds = test_model(model, new_loader, criterion, num_classes=NUM_CLASSES)
    return all_labels, all_preds

def validate_labels(datasets, num_classes):
    for dataset in datasets:
        for idx, (_, _, _, _, labels) in enumerate(dataset):
            invalid_mask = (labels >= num_classes) | (labels < 0)
            if torch.any(invalid_mask):
                invalid_labels = labels[invalid_mask].unique().tolist()
                raise ValueError(f"Found invalid labels {invalid_labels} in dataset at index {idx}. Labels should be in the range [0, {num_classes - 1}].")
    print("All labels are valid.")

def get_all_unique_labels(datasets):
    unique_labels = set()
    for dataset in datasets:
        for _, _, _, _, labels in dataset:
            unique_labels.update(labels.cpu().numpy().flatten())
    return unique_labels

def objective(trial):
    # Suggest hyperparameters
    hidden_dim = trial.suggest_int('hidden_dim', 128, 512, step=64)
    num_layers =  trial.suggest_int('num_layers', 2, 16)
    dropout = trial.suggest_float('dropout', 0.1, 0.5, step=0.05)
    num_heads = trial.suggest_categorical('num_heads', [2, 4, 8, 16])
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    
    # Gradient accumulation steps
    accumulation_steps = trial.suggest_int('accumulation_steps', 1, 4)

    # Set seed for reproducibility
    set_seed(42)

    # Load data
    json_path = '../../prosody/data/ambiguous_prosody_multi_label_features_train_embeddings.json'
    data = load_data(json_path)

    # Split data
    train_data, val_data, test_data = split_data(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)

    # Create datasets
    train_dataset = ProsodyDataset(dict(train_data))
    val_dataset = ProsodyDataset(dict(val_data))
    test_dataset = ProsodyDataset(dict(test_data))

    # Create data loaders with efficient data loading
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn,
                              num_workers=7, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn,
                            num_workers=7, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn,
                             num_workers=7, pin_memory=True)

    # Determine number of classes
    all_unique_labels = get_all_unique_labels([train_dataset, val_dataset, test_dataset])
    num_classes = len(all_unique_labels) + 2  # +1 for label shifting, +1 for SOS token

    global SOS_IDX, NUM_CLASSES
    SOS_IDX = num_classes - 1
    NUM_CLASSES = num_classes

    # Validate labels
    try:
        validate_labels([train_dataset, val_dataset, test_dataset], NUM_CLASSES)
    except ValueError as e:
        print(e)
        return 0.0  # Invalid trial

    # Initialize Encoder and Decoder with suggested hyperparameters
    encoder = ProjectedEncoder(
        feature_dim=train_loader.dataset[0][3].shape[1],  # Corrected to shape[1]
        embedding_dim=train_loader.dataset[0][2].shape[1],
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        num_heads=num_heads
    ).to(device)

    decoder = Decoder(
        hidden_dim=hidden_dim,
        output_dim=NUM_CLASSES,
        num_layers=num_layers,
        dropout=dropout,
        num_classes=NUM_CLASSES,
        num_heads=num_heads,
        embedding_dim=train_loader.dataset[0][2].shape[1]
    ).to(device)

    model = Seq2Seq(encoder, decoder).to(device)

    # Define optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)

    # Define loss function
    criterion = nn.CrossEntropyLoss(ignore_index=PADDING_VALUE)

    # Initialize scaler for mixed precision
    scaler = amp.GradScaler()

    # Initialize early stopping
    early_stopping = EarlyStopping(patience=10, min_delta=0.001)

    # Training parameters
    N_EPOCHS = 100
    best_valid_f1 = 0.0

    for epoch in range(N_EPOCHS):
        train_loss = train_model(model, train_loader, optimizer, criterion, scaler, accumulation_steps)
        if train_loss is None:
            raise optuna.exceptions.TrialPruned()

        valid_loss, valid_acc, valid_precision, valid_recall, valid_f1 = evaluate_model(model, val_loader, criterion)

        scheduler.step(valid_loss)

        # Report intermediate objective value
        trial.report(valid_f1, epoch)

        # Prune trial if necessary
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        # Early stopping
        early_stopping(valid_loss)
        if early_stopping.early_stop:
            break

        # Check GPU constraints after each epoch
        temp, mem_used = get_gpu_status()
        if temp > MAX_GPU_TEMP or mem_used > MAX_GPU_MEM:
            print(f"GPU constraints exceeded: Temp={temp}°C, Memory Used={mem_used}MB")
            raise optuna.exceptions.TrialPruned()

        # Update best F1
        if valid_f1 > best_valid_f1:
            best_valid_f1 = valid_f1

    return best_valid_f1

def main():
    # Create Optuna study
    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
    
    # Optimize
    study.optimize(objective, n_trials=100, timeout=None)  # Adjust n_trials and timeout as needed

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # After finding best hyperparameters, train the final model
    best_params = trial.params
    hidden_dim = best_params['hidden_dim']
    num_layers = best_params['num_layers']
    dropout = best_params['dropout']
    num_heads = best_params['num_heads']
    lr = best_params['lr']
    weight_decay = best_params['weight_decay']
    accumulation_steps = best_params.get('accumulation_steps', 1)

    # Set seed
    set_seed(42)

    # Load data
    json_path = '../../prosody/data/ambiguous_prosody_multi_label_features_train_embeddings.json'
    data = load_data(json_path)

    # Split data
    train_data, val_data, test_data = split_data(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)

    # Create datasets
    train_dataset = ProsodyDataset(dict(train_data))
    val_dataset = ProsodyDataset(dict(val_data))
    test_dataset = ProsodyDataset(dict(test_data))

    # Create data loaders with efficient data loading
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn,
                              num_workers=7, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn,
                            num_workers=7, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn,
                             num_workers=7, pin_memory=True)

    # Determine number of classes
    all_unique_labels = get_all_unique_labels([train_dataset, val_dataset, test_dataset])
    num_classes = len(all_unique_labels) + 2  # +1 for label shifting, +1 for SOS token

    global SOS_IDX, NUM_CLASSES
    SOS_IDX = num_classes - 1
    NUM_CLASSES = num_classes

    # Validate labels
    try:
        validate_labels([train_dataset, val_dataset, test_dataset], NUM_CLASSES)
    except ValueError as e:
        print(e)
        return

    # Initialize Encoder and Decoder with best hyperparameters
    encoder = ProjectedEncoder(
        feature_dim=train_loader.dataset[0][3].shape[1],  # Corrected to shape[1]
        embedding_dim=train_loader.dataset[0][2].shape[1],
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        num_heads=num_heads
    ).to(device)

    decoder = Decoder(
        hidden_dim=hidden_dim,
        output_dim=num_classes,
        num_layers=num_layers,
        dropout=dropout,
        num_classes=num_classes,
        num_heads=num_heads,
        embedding_dim=train_loader.dataset[0][2].shape[1]
    ).to(device)

    model = Seq2Seq(encoder, decoder).to(device)

    # Define optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)

    # Define loss function
    criterion = nn.CrossEntropyLoss(ignore_index=PADDING_VALUE)

    # Initialize scaler for mixed precision
    scaler = amp.GradScaler()

    # Initialize early stopping
    early_stopping = EarlyStopping(patience=10, min_delta=0.001)

    # Training parameters
    N_EPOCHS = 100
    best_valid_f1 = 0.0

    # Ensure the models directory exists
    os.makedirs('../models', exist_ok=True)
    best_model_filename = f"../models/best-transformer-model-ambiguous_prosody_multiclass.pt"

    # Initialize lists to store metrics
    train_losses = []
    val_losses = []
    val_accuracies = []
    val_precisions = []
    val_recalls = []
    val_f1s = []

    for epoch in range(N_EPOCHS):
        train_loss = train_model(model, train_loader, optimizer, criterion, scaler, accumulation_steps)
        if train_loss is None:
            print("GPU constraints exceeded during training. Stopping.")
            break

        valid_loss, valid_acc, valid_precision, valid_recall, valid_f1 = evaluate_model(model, val_loader, criterion)

        scheduler.step(valid_loss)

        train_losses.append(train_loss)
        val_losses.append(valid_loss)
        val_accuracies.append(valid_acc)
        val_precisions.append(valid_precision)
        val_recalls.append(valid_recall)
        val_f1s.append(valid_f1)

        # Save the model if validation F1 improves
        if valid_f1 > best_valid_f1:
            best_valid_f1 = valid_f1
            torch.save(model.state_dict(), best_model_filename)

        print(f'Epoch: {epoch+1:02} | Train Loss: {train_loss:.4f} | Val. Loss: {valid_loss:.4f} | '
              f'Val. Acc: {valid_acc*100:.2f}% | Precision: {valid_precision:.4f} | '
              f'Recall: {valid_recall:.4f} | F1 Score: {valid_f1:.4f}')

        # Early stopping
        early_stopping(valid_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

        # Check GPU constraints after each epoch
        temp, mem_used = get_gpu_status()
        if temp > MAX_GPU_TEMP or mem_used > MAX_GPU_MEM:
            print(f"GPU constraints exceeded: Temp={temp}°C, Memory Used={mem_used}MB")
            break

    # Load the best model and evaluate on the test set
    model.load_state_dict(torch.load(best_model_filename))
    test_loss, test_acc, test_precision, test_recall, test_f1 = evaluate_model(model, test_loader, criterion)
    print(f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.2f}% | '
          f'Precision: {test_precision:.4f} | Recall: {test_recall:.4f} | F1 Score: {test_f1:.4f}')

    # Generate detailed test results
    test_model(model, test_loader, criterion, num_classes=NUM_CLASSES)

    # Plot training and validation metrics
    plot_metrics(train_losses, val_losses, val_accuracies, val_precisions, val_recalls, val_f1s)

    # Evaluate model on held out set
    eval_json = "../../prosody/data/ambiguous_prosody_multi_label_features_eval_embeddings.json"
    true_labels, predicted_labels = evaluate_new_set(model, eval_json)

    # Log directory
    log_dir = "../../prosody/outputs"
    os.makedirs(log_dir, exist_ok=True)

    # Class names (assuming labels are integers starting from 0)
    class_names = sorted(list(all_unique_labels))

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
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        class_accuracy.append(accuracy)

    # Compute weighted accuracy
    total_support = sum(class_support)
    weighted_accuracy = sum(acc * supp for acc, supp in zip(class_accuracy, class_support)) / total_support

    # Write class-wise metrics to a file, including accuracy
    classwise_metrics_path = os.path.join(log_dir, "prosody_classwise_metrics.txt")
    with open(classwise_metrics_path, "w") as f:
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

if __name__ == "__main__":
    main()
