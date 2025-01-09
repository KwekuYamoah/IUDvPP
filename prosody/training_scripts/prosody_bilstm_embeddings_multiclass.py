"""
Prosody Analysis using BiLSTM with Attention for Multiclass Classification

This module implements a sequence-to-sequence model using Bidirectional LSTM (BiLSTM) 
with attention mechanisms to perform multiclass classification on prosody data. 
It includes data preprocessing, embedding initialization with GloVe, model training, 
evaluation, and testing functionalities.

Dependencies:
    - Python 3.x
    - PyTorch
    - NumPy
    - scikit-learn
    - matplotlib
    - pandas

Author: Your Name
Date: 2024-04-27
"""

import json
import random
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

# ==============================
# Utility Functions
# ==============================
PADDING_LABEL = 0
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

def load_glove_embeddings(file_path, embedding_dim):
    """
    Loads GloVe embeddings from a file.

    Args:
        file_path (str): Path to the GloVe embeddings file.
        embedding_dim (int): Dimension of the embeddings.

    Returns:
        dict: A dictionary mapping words to their embedding vectors.
    """
    embeddings_index = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

def create_embedding_matrix(word2idx, embeddings_index, embedding_dim):
    """
    Creates an embedding matrix for the vocabulary using pre-trained embeddings.

    Args:
        word2idx (dict): Mapping from words to their indices.
        embeddings_index (dict): Pre-trained embeddings.
        embedding_dim (int): Dimension of the embeddings.

    Returns:
        np.ndarray: The embedding matrix.
    """
    vocab_size = len(word2idx) + 1  # +1 for the padding token
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, idx in word2idx.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[idx] = embedding_vector
        else:
            # Initialize with random vectors for words not in GloVe
            embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embedding_dim,))
    return embedding_matrix

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
        train_ratio (float): Proportion of data for training.
        val_ratio (float): Proportion of data for validation.
        test_ratio (float): Proportion of data for testing.

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

# ==============================
# Dataset and DataLoader
# ==============================

class ProsodyDataset(Dataset):
    """
    Custom Dataset for Prosody Data.

    Each sample consists of tokenized words, associated features, and labels.
    """

    def __init__(self, data, word2idx, unk_idx):
        """
        Initializes the dataset.

        Args:
            data (dict): The dataset containing entries with words, features, and labels.
            word2idx (dict): Mapping from words to their indices.
            unk_idx (int): Index for unknown words.
        """
        self.data = data
        self.word2idx = word2idx
        self.unk_idx = unk_idx
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
            tuple: (words_tensor, features_tensor, labels_tensor)
        """
        key, item = self.entries[idx]
        processed_words = preprocess_text(item['words'])
        words = [self.word2idx.get(word, self.unk_idx) for word in processed_words]
        features = torch.tensor(item['features'], dtype=torch.float32)
        labels = torch.tensor(item['labels'], dtype=torch.long)  # For CrossEntropyLoss
        return torch.tensor(words, dtype=torch.long), features, labels

def collate_fn(batch):
    """
    Collate function to handle batches with variable sequence lengths.

    Pads sequences and prepares tensors for words, features, and labels.

    Args:
        batch (list): List of samples from the dataset.

    Returns:
        tuple: Padded words, features, labels, and their lengths.
    """
    words = [item[0] for item in batch]
    features = [item[1] for item in batch]
    labels = [item[2] for item in batch]

    # Pad words with the padding index (PADDING_IDX)
    words_padded = torch.nn.utils.rnn.pad_sequence(words, batch_first=True, padding_value=PADDING_IDX)
    features_padded = torch.nn.utils.rnn.pad_sequence(features, batch_first=True, padding_value=0.0)
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=PADDING_LABEL)

    lengths = torch.tensor([len(w) for w in words])

    return words_padded, features_padded, labels_padded, lengths

# ==============================
# Early Stopping
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
            hidden (torch.Tensor): Current hidden state of the decoder.
            encoder_outputs (torch.Tensor): Outputs from the encoder.
            mask (torch.Tensor): Mask to ignore padding tokens.

        Returns:
            torch.Tensor: Attention weights.
        """
        src_len = encoder_outputs.size(1)
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)  # Repeat hidden state
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)

        # Apply the mask to ignore padding positions
        attention.masked_fill_(mask == 0, -1e10)

        return torch.softmax(attention, dim=1)

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
            hidden (torch.Tensor): Current hidden state of the decoder.
            encoder_outputs (torch.Tensor): Outputs from the encoder.
            mask (torch.Tensor): Mask to ignore padding tokens.

        Returns:
            torch.Tensor: Averaged attention weights.
        """
        attn_weights = []
        for layer in self.attention_layers:
            attn_weight = layer(hidden, encoder_outputs, mask)
            attn_weights.append(attn_weight)
        attn_weights = torch.stack(attn_weights, dim=1).mean(dim=1)  # Average over attention layers
        return attn_weights

# ==============================
# Encoder and Decoder Models
# ==============================

class Encoder(nn.Module):
    """
    Encoder module using a Bidirectional LSTM with embedding and feature integration.

    Attributes:
        embedding (nn.Embedding): Embedding layer initialized with pre-trained embeddings.
        lstm (nn.LSTM): Bidirectional LSTM layer.
    """

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout, embeddings, feature_dim):
        """
        Initializes the Encoder.

        Args:
            vocab_size (int): Size of the vocabulary.
            embedding_dim (int): Dimension of word embeddings.
            hidden_dim (int): Dimension of LSTM hidden states.
            num_layers (int): Number of LSTM layers.
            dropout (float): Dropout rate.
            embeddings (torch.Tensor): Pre-trained embedding weights.
            feature_dim (int): Dimension of additional features.
        """
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PADDING_IDX)
        self.embedding.weight = nn.Parameter(embeddings)
        self.embedding.weight.requires_grad = True  # Allow fine-tuning of embeddings
        self.lstm = nn.LSTM(embedding_dim + feature_dim, hidden_dim, num_layers, 
                            dropout=dropout, batch_first=True, bidirectional=True)

    def forward(self, words, features, lengths):
        """
        Forward pass of the Encoder.

        Args:
            words (torch.Tensor): Batch of word indices.
            features (torch.Tensor): Batch of additional features.
            lengths (torch.Tensor): Lengths of each sequence in the batch.

        Returns:
            tuple: Encoder outputs, hidden states, and cell states.
        """
        # Sort sequences by lengths in descending order for packing
        lengths_sorted, sorted_indices = lengths.sort(descending=True)
        words = words[sorted_indices]
        features = features[sorted_indices]

        embedded = self.embedding(words)  # Shape: (batch_size, seq_len, embedding_dim)
        combined = torch.cat((embedded, features), dim=2)  # Concatenate features

        # Pack the sequences for efficient processing
        packed_input = rnn_utils.pack_padded_sequence(combined, lengths_sorted.cpu(), 
                                                      batch_first=True, enforce_sorted=True)

        packed_output, (hidden, cell) = self.lstm(packed_input)

        # Unpack the sequences
        outputs, _ = rnn_utils.pad_packed_sequence(packed_output, batch_first=True, padding_value=0.0)

        # Restore the original ordering
        _, original_indices = sorted_indices.sort()
        outputs = outputs[original_indices]
        hidden = hidden[:, original_indices]
        cell = cell[:, original_indices]

        return outputs, hidden, cell

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
            encoder_outputs (torch.Tensor): Outputs from the encoder.
            hidden (torch.Tensor): Hidden state from the encoder.
            cell (torch.Tensor): Cell state from the encoder.
            mask (torch.Tensor): Mask to ignore padding tokens.

        Returns:
            tuple: Predictions and updated hidden and cell states.
        """
        # Compute attention weights
        attn_weights = self.attention(hidden[-1], encoder_outputs, mask)
        # Compute context vector as weighted sum of encoder outputs
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)

        # Prepare LSTM input by concatenating context with encoder outputs
        lstm_input = torch.cat((context.unsqueeze(1).repeat(1, encoder_outputs.size(1), 1), 
                                encoder_outputs), dim=2)
        outputs, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        predictions = self.fc(outputs)  # No activation for CrossEntropyLoss

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

    def forward(self, src, features, lengths):
        """
        Forward pass of the Seq2Seq model.

        Args:
            src (torch.Tensor): Source sequences (word indices).
            features (torch.Tensor): Additional features.
            lengths (torch.Tensor): Lengths of each sequence in the batch.

        Returns:
            torch.Tensor: Output predictions.
        """
        encoder_outputs, hidden, cell = self.encoder(src, features, lengths)
        mask = (src != PADDING_IDX)  # Create mask based on padding
        outputs, (hidden, cell) = self.decoder(encoder_outputs, hidden, cell, mask)
        return outputs

# ==============================
# Training and Evaluation
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
        words = words.to(device)
        features = features.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)

        optimizer.zero_grad()
        output = model(words, features, lengths)  # Shape: (batch_size, seq_len, output_dim)
        output = output.view(-1, OUTPUT_DIM)  # Flatten for loss computation
        labels = labels.view(-1)  # Flatten labels

        loss = criterion(output, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)  # Gradient clipping
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
            words = words.to(device)
            features = features.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)

            output = model(words, features, lengths)
            output = output.view(-1, OUTPUT_DIM)
            labels = labels.view(-1)

            loss = criterion(output, labels)
            epoch_loss += loss.item()

            preds = torch.argmax(output, dim=1)

            # Mask out padding tokens
            mask = labels != PADDING_LABEL
            masked_preds = preds[mask]
            masked_labels = labels[mask]

            all_labels.extend(masked_labels.cpu().numpy())
            all_preds.extend(masked_preds.cpu().numpy())

    # Compute evaluation metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    return epoch_loss / len(iterator), accuracy, precision, recall, f1

def test_model(model, iterator, word2idx):
    """
    Tests the model on the test dataset and saves detailed results.

    Args:
        model (nn.Module): The trained Seq2Seq model.
        iterator (DataLoader): DataLoader for test data.
        word2idx (dict): Mapping from words to indices.

    Returns:
        tuple: All true labels and all predicted labels.
    """
    model.eval()
    all_labels = []
    all_preds = []
    idx2word = {idx: word for word, idx in word2idx.items()}

    # Initialize results file
    with open('./outputs/prosody_bilstm_embeddings_multiclass_results.txt', 'w') as file:
        file.write("")

    with torch.no_grad():
        for words, features, labels, lengths in iterator:
            words = words.to(device)
            features = features.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)

            output = model(words, features, lengths)
            preds = torch.argmax(output, dim=2)  # Shape: (batch_size, seq_len)

            for i in range(words.shape[0]):
                word_indices = words[i].cpu().numpy()
                word_sentence = [idx2word.get(idx, '<UNK>') for idx in word_indices]
                gold_labels = labels[i].cpu().numpy().flatten()
                pred_labels = preds[i].cpu().numpy().flatten()

                # Clean up the sentence by excluding padding tokens
                cleaned_words, cleaned_gold_labels, cleaned_pred_labels = clean_up_sentence(
                    word_sentence, gold_labels, pred_labels
                )

                data = {
                    'Word': cleaned_words,
                    'Gold Label': cleaned_gold_labels,
                    'Predicted Label': cleaned_pred_labels
                }

                df = pd.DataFrame(data)
                with open('./outputs/prosody_bilstm_embeddings_multiclass_results.txt', 'a') as file:
                    file.write(df.to_string(index=False))
                    file.write("\n" + "-" * 50 + "\n")

                all_labels.extend(cleaned_gold_labels)
                all_preds.extend(cleaned_pred_labels)

    # Compute evaluation metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    print(f'Test Accuracy: {accuracy*100:.2f}%')
    print(f'Test Precision: {precision:.2f}')
    print(f'Test Recall: {recall:.2f}')
    print(f'Test F1 Score: {f1:.2f}')
    
    return all_labels, all_preds

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
    plt.figure(figsize=(12, 8))

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
    plt.savefig('./outputs/bilstm_embeddings_multiclass_metrics.png')
    plt.close()

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
    
    for i in range(len(words)):
        if words[i] != '<UNK>' and gold_labels[i] != PADDING_LABEL:
            filtered_words.append(words[i])
            filtered_gold_labels.append(gold_labels[i])
            filtered_pred_labels.append(pred_labels[i])

    return filtered_words, filtered_gold_labels, filtered_pred_labels

# ==============================
# Main Execution
# ==============================

if __name__ == "__main__":
    # ==============================
    # Configuration and Setup
    # ==============================

    seed = 42 
    set_seed(seed)  # Set seed for reproducibility

    json_path = '../prosody/data/multi_label_features.json'  # Path to data
    data = load_data(json_path)  # Load data

    # Compute number of classes dynamically
    all_labels = []
    for item in data.values():
        all_labels.extend(item['labels'])
    num_classes = len(set(all_labels))
    print(f'Number of classes: {num_classes}')

    # Split data into train, validation, and test sets
    train_data, val_data, test_data = split_data(data)
    combined_corpus = get_corpus(dict(train_data)) + get_corpus(dict(val_data)) + get_corpus(dict(test_data))

    # ==============================
    # Embedding Initialization
    # ==============================

    embedding_dim = 300  # Dimension of GloVe embeddings
    glove_path = '../prosody/glove_embeddings/glove.6B.300d.txt'  # Path to GloVe file
    glove_embeddings = load_glove_embeddings(glove_path, embedding_dim)  # Load GloVe embeddings

    vocab = list(glove_embeddings.keys())  # Vocabulary from GloVe
    word2idx = {word: idx for idx, word in enumerate(vocab)}  # Word to index mapping
    unk_idx = len(word2idx)  # Index for unknown words
    word2idx['<UNK>'] = unk_idx  # Add <UNK> token

    # Set padding index to the next index
    PADDING_IDX = len(word2idx)
    word2idx['<PAD>'] = PADDING_IDX  # Add <PAD> token

    embedding_matrix = create_embedding_matrix(word2idx, glove_embeddings, embedding_dim)  # Create embedding matrix

    # Adjust embedding matrix size to account for padding index
    embedding_matrix = np.vstack((embedding_matrix, np.zeros((1, embedding_dim))))  # Add zero vector for <PAD>

    # ==============================
    # Dataset and DataLoader Setup
    # ==============================

    train_dataset = ProsodyDataset(dict(train_data), word2idx, unk_idx)
    val_dataset = ProsodyDataset(dict(val_data), word2idx, unk_idx)
    test_dataset = ProsodyDataset(dict(test_data), word2idx, unk_idx)

    print(f'Train size: {len(train_dataset)}')
    print(f'Validation size: {len(val_dataset)}')
    print(f'Test size: {len(test_dataset)}')

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # ==============================
    # Model Initialization
    # ==============================

    VOCAB_SIZE = len(word2idx)
    EMBEDDING_DIM = 300
    HIDDEN_DIM = 128
    OUTPUT_DIM = num_classes  # Number of output classes
    NUM_LAYERS = 4
    DROPOUT = 0.5
    NUM_ATTENTION_LAYERS = 4

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available
    feature_dim = next(iter(train_loader))[1].shape[2]  # Dimension of additional features

    embedding_tensor = torch.tensor(embedding_matrix, dtype=torch.float32)  # Convert embedding matrix to tensor

    # Initialize Encoder and Decoder
    encoder = Encoder(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT, embedding_tensor, feature_dim).to(device)
    decoder = Decoder(HIDDEN_DIM, OUTPUT_DIM, NUM_LAYERS, DROPOUT, NUM_ATTENTION_LAYERS).to(device)
    model = Seq2Seq(encoder, decoder).to(device)  # Combine into Seq2Seq model

    # ==============================
    # Optimizer, Scheduler, and Loss
    # ==============================

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # Adam optimizer with weight decay
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # Learning rate scheduler

    criterion = nn.CrossEntropyLoss(ignore_index=PADDING_LABEL)  # Loss function with padding ignored

    # ==============================
    # Training Loop
    # ==============================

    train_losses = []
    val_losses = []
    val_accuracies = []
    val_precisions = []
    val_recalls = []
    val_f1s = []

    N_EPOCHS = 500  # Maximum number of epochs
    CLIP = 1  # Gradient clipping value

    early_stopping = EarlyStopping(patience=10, min_delta=0.001)  # Initialize early stopping
    best_valid_loss = float('inf')  # Track best validation loss

    for epoch in range(N_EPOCHS):
        train_loss = train(model, train_loader, optimizer, criterion)
        valid_loss, valid_acc, valid_precision, valid_recall, valid_f1 = evaluate(model, val_loader, criterion)

        train_losses.append(train_loss)
        val_losses.append(valid_loss)
        val_accuracies.append(valid_acc)
        val_precisions.append(valid_precision)
        val_recalls.append(valid_recall)
        val_f1s.append(valid_f1)

        # Update the learning rate based on the validation loss
        scheduler.step(valid_loss)

        # Save the model if validation loss improves
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'models/best-model-embeddings-multiclass-version.pt')

        print(f'Epoch: {epoch+1:02}')
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}% |  Precision: {valid_precision:.2f} |  Recall: {valid_recall:.2f} |  F1 Score: {valid_f1:.2f}')

        # Check for early stopping
        early_stopping(valid_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # ==============================
    # Testing and Evaluation
    # ==============================

    # Load the best model
    model.load_state_dict(torch.load('models/best-model-embeddings-multiclass-version.pt'))
    test_model(model, test_loader, word2idx)

    # ==============================
    # Plotting Metrics
    # ==============================

    plot_metrics(train_losses, val_losses, val_accuracies, val_precisions, val_recalls, val_f1s)
