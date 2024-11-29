import os
import json
import random
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import multilabel_confusion_matrix, precision_recall_fscore_support
import pandas as pd
import re
import matplotlib.pyplot as plt

# torch imports
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

def set_seed(seed):
    """
    Set the seed for generating random numbers to ensure reproducibility.

    This function sets the seed for Python's built-in random module, NumPy's random module,
    and PyTorch's random number generators. It also configures PyTorch to use deterministic
    algorithms and disables the benchmark mode in cuDNN to ensure reproducibility.

    Args:
        seed (int): The seed value to use for random number generation.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# Custom padding value for labels
PADDING_VALUE = 0  # Use 0 as the padding index

class EarlyStopping:
    """
    Early stopping to terminate training when validation loss does not improve after a given patience.

    Attributes:
        patience (int): Number of epochs to wait after last time validation loss improved.
        min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        best_loss (float or None): Best recorded validation loss.
        counter (int): Number of epochs since the last improvement in validation loss.
        early_stop (bool): Flag to indicate whether training should be stopped.

    Methods:
        __call__(val_loss):
            Checks if the validation loss has improved and updates the counter and early_stop flag accordingly.
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

# Load data from a JSON file
def load_data(json_path):
    """
    Load data from a JSON file.

    Args:
        json_path (str): The path to the JSON file.

    Returns:
        dict: The data loaded from the JSON file.
    """
    with open(json_path, 'r') as file:
        data = json.load(file)
    return data

# Split the data into train, validation, and test sets
def split_data(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Splits the given data into training, validation, and test sets based on the provided ratios.

    Args:
        data (dict): The data to be split, where keys are data identifiers and values are data points.
        train_ratio (float, optional): The proportion of the data to be used for training. Defaults to 0.8.
        val_ratio (float, optional): The proportion of the data to be used for validation. Defaults to 0.1.
        test_ratio (float, optional): The proportion of the data to be used for testing. Defaults to 0.1.

    Returns:
        tuple: A tuple containing three lists: the training set, the validation set, and the test set.

    Raises:
        AssertionError: If the sum of train_ratio, val_ratio, and test_ratio is not equal to 1.
    """
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1"
    total = len(data)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    test_size = total - train_size - val_size
    return random_split(list(data.items()), [train_size, val_size, test_size])

# Preprocess words (remove punctuation and tokenize)
def preprocess_text(words):
    """
    Preprocesses a list of words by splitting each word into individual tokens.

    This function uses a regular expression to split each word into tokens that
    include words, contractions, and punctuation marks such as periods, commas,
    exclamation points, question marks, and semicolons.

    Args:
        words (list of str): A list of words to be processed.

    Returns:
        list of str: A list of processed tokens.
    """
    processed_words = []
    for word in words:
        processed_words.extend(re.findall(r"[\w']+|[.,!?;]", word))
    return processed_words

# Clean up sentence by removing padding
def clean_up_sentence(words, gold_labels, pred_labels, padding_value):
    """
    Cleans up the input sentence by removing elements with padding values.
    Args:
        words (list): A list of words in the sentence.
        gold_labels (list): A list of gold label values corresponding to the words.
        pred_labels (list): A list of predicted label values corresponding to the words.
        padding_value (int): The value used for padding that should be removed.
    Returns:
        tuple: A tuple containing three lists:
            - filtered_words (list): The list of words with padding values removed.
            - filtered_gold_labels (list): The list of gold labels with padding values removed.
            - filtered_pred_labels (list): The list of predicted labels with padding values removed.
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

# Custom dataset class for prosody features
class ProsodyDataset(Dataset):
    """
    A custom Dataset class for handling prosody data.

    Args:
        data (dict): A dictionary where keys are identifiers and values are dictionaries 
                     containing 'words', 'prosodic_features', 'raw_acoustic_features', and 'labels'.

    Attributes:
        entries (list): A list of tuples where each tuple contains a key and its corresponding data.

    Methods:
        __len__(): Returns the number of entries in the dataset.
        __getitem__(idx): Retrieves the words, features, and labels for the given index.
    """
    def __init__(self, data):
        self.entries = list(data.items())

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        key, item = self.entries[idx]
        words = preprocess_text(item['words'])
        prosodic_features = torch.tensor(item['prosodic_features'], dtype=torch.float32)
        raw_acoustic_features = torch.tensor(item['raw_acoustic_features'], dtype=torch.float32)
        # Concatenate features along the feature dimension
        features = torch.cat((prosodic_features, raw_acoustic_features), dim=1)
        labels = torch.tensor(item['labels'], dtype=torch.long) + 1  # Shift labels by +1
        return words, features, labels

# Custom collate function to handle padding
def collate_fn(batch):
    """
    Collates a batch of data for the prosody transformer model.

    Args:
        batch (list of tuples): A list where each tuple contains:
            - item[0] (list of str): List of words.
            - item[1] (torch.Tensor): Tensor of combined features.
            - item[2] (torch.Tensor): Tensor of labels.

    Returns:
        tuple: A tuple containing:
            - words (list of list of str): List of lists of words.
            - features_padded (torch.Tensor): Padded tensor of features with shape (batch_size, max_seq_length, feature_dim).
            - labels_padded (torch.Tensor): Padded tensor of labels with shape (batch_size, max_seq_length).
            - lengths (torch.Tensor): Tensor containing the lengths of each sequence in the batch.
    """
    words = [item[0] for item in batch]  # List of lists of words
    features = [item[1] for item in batch]
    labels = [item[2] for item in batch]

    features_padded = torch.nn.utils.rnn.pad_sequence(features, batch_first=True, padding_value=0.0)
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=PADDING_VALUE)
    lengths = torch.tensor([len(f) for f in features])

    return words, features_padded, labels_padded, lengths

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

# Transformer Encoder with Padding Masking
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
        decoder_layer (nn.TransformerDecoderLayer): A single transformer decoder layer.
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
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True  # Align with data format
        )
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(hidden_dim, num_classes)  # Ensure output_dim == num_classes

    def forward(self, memory, tgt, tgt_mask=None, tgt_key_padding_mask=None):
        tgt_embed = self.embedding(tgt)  # [batch_size, seq_len, hidden_dim]
        tgt_embed = self.positional_encoding(tgt_embed)
        output = self.transformer_decoder(
            tgt_embed,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )  # [batch_size, seq_len, hidden_dim]
        output = self.fc_out(output)  # [batch_size, seq_len, num_classes]
        return output

# TransformerSeq2Seq combining Encoder and Decoder with Masking
class TransformerSeq2Seq(nn.Module):
    """
    Transformer Sequence-to-Sequence Model.
    This model consists of an encoder and a decoder, both of which are transformer-based.
    It is designed to handle sequence-to-sequence tasks.
    Args:
        encoder (nn.Module): The encoder module.
        decoder (nn.Module): The decoder module.
    Methods:
        forward(features, labels, lengths):
            Forward pass of the model.
            Args:
                features (torch.Tensor): Input features of shape [batch_size, seq_len, feature_dim].
                labels (torch.Tensor): Target labels of shape [batch_size, seq_len].
                lengths (torch.Tensor): Lengths of the sequences in the batch.
            Returns:
                torch.Tensor: Output predictions of shape [batch_size, seq_len, num_classes].
    """
    def __init__(self, encoder, decoder):
        super(TransformerSeq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, features, labels, lengths):
        # features: [batch_size, seq_len, feature_dim]
        # labels: [batch_size, seq_len]
        
        # Generate padding mask for encoder (source)
        src_key_padding_mask = create_padding_mask(features.sum(dim=2), padding_value=0.0)
        
        # Pass through encoder
        memory = self.encoder(features, src_key_padding_mask=src_key_padding_mask)  # [batch_size, seq_len, hidden_dim]
        
        # Prepare decoder inputs by shifting labels and adding SOS token
        batch_size, seq_len = labels.size()
        decoder_inputs = torch.full((batch_size, 1), SOS_IDX, dtype=labels.dtype, device=labels.device)
        decoder_inputs = torch.cat([decoder_inputs, labels[:, :-1]], dim=1)  # Shifted labels
        
        # Generate causal mask for decoder
        tgt_seq_len = decoder_inputs.size(1)
        tgt_mask = generate_causal_mask(tgt_seq_len).to(features.device)  # [seq_len, seq_len]
        
        # Generate padding mask for decoder (target)
        tgt_key_padding_mask = create_padding_mask(decoder_inputs, padding_value=PADDING_VALUE).to(features.device)  # [batch_size, seq_len]
        
        # Pass through decoder
        outputs = self.decoder(
            memory,
            decoder_inputs,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )  # [batch_size, seq_len, num_classes]
        
        return outputs

# Function to generate a causal mask
def generate_causal_mask(sz):
    """
    Generates a causal mask for a sequence of a given size.
    This mask is an upper triangular matrix with ones above the main diagonal
    and zeros on and below the main diagonal. It is used to prevent the model
    from attending to future tokens in a sequence during training.
    Args:
        sz (int): The size of the sequence.
    Returns:
        torch.Tensor: A boolean tensor of shape (sz, sz) representing the causal mask.
    """

    mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()  # Upper triangular part (excluding diagonal)
    return mask  # [sz, sz]

# Function to create padding masks
def create_padding_mask(seq, padding_value=0):
    """
    Creates a padding mask for sequences.
    Args:
        seq: Tensor of shape [batch_size, seq_len]
        padding_value: The value used for padding in labels
    Returns:
        mask: Tensor of shape [batch_size, seq_len], where True indicates padding positions
    """
    mask = (seq == padding_value)
    return mask  # [batch_size, seq_len]

# Training function
def train_model(model, iterator, optimizer, criterion, num_classes):
    """
    Trains the given model for one epoch.

    Args:
        model (torch.nn.Module): The model to be trained.
        iterator (iterable): An iterable that provides batches of data.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        criterion (torch.nn.Module): The loss function.
        num_classes (int): The number of classes.

    Returns:
        float: The average loss over the epoch.
    """
    model.train()
    epoch_loss = 0
    for batch_idx, (words, features, labels, lengths) in enumerate(iterator):
        features = features.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)

        optimizer.zero_grad()
        output = model(features, labels, lengths)  # [batch_size, seq_len, num_classes]
        output = output.view(-1, num_classes)  # [batch_size * seq_len, num_classes]
        labels_flat = labels.view(-1)  # [batch_size * seq_len]

        # Exclude padding indices from loss
        loss = criterion(output, labels_flat)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

# Evaluation function
def evaluate_model(model, iterator, criterion, num_classes):
    """
    Evaluates the performance of a given model on a dataset.

    Args:
        model (torch.nn.Module): The model to evaluate.
        iterator (DataLoader): DataLoader providing the dataset to evaluate on.
        criterion (torch.nn.Module): Loss function to use for evaluation.
        num_classes (int): Number of classes in the output.

    Returns:
        tuple: A tuple containing:
            - epoch_loss (float): The average loss over the dataset.
            - accuracy (float): The accuracy of the model on the dataset.
            - precision (float): The precision of the model on the dataset.
            - recall (float): The recall of the model on the dataset.
            - f1 (float): The F1 score of the model on the dataset.
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

            output = model(features, labels, lengths)  # [batch_size, seq_len, num_classes]
            output = output.view(-1, num_classes)  # [batch_size * seq_len, num_classes]
            labels_flat = labels.view(-1)  # [batch_size * seq_len]

            preds = torch.argmax(output, dim=1)  # [batch_size * seq_len]

            loss = criterion(output, labels_flat)
            epoch_loss += loss.item()

            # Exclude padding indices from metrics
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

# Test model function
def test_model(model, iterator):
    """
    Evaluates the given model on the provided data iterator and writes the results to a file.

    Args:
        model (torch.nn.Module): The model to be evaluated.
        iterator (iterable): An iterable that provides batches of data. Each batch should be a tuple containing:
            - words (list of list of str): The words in each sentence.
            - features (torch.Tensor): The input features for the model.
            - labels (torch.Tensor): The ground truth labels.
            - lengths (torch.Tensor): The lengths of each sequence in the batch.

    Returns:
        tuple: A tuple containing:
            - all_labels (list): The list of all ground truth labels.
            - all_preds (list): The list of all predicted labels.
    """
    model.eval()
    all_labels = []
    all_preds = []

    os.makedirs('./outputs', exist_ok=True)
    with open('./outputs/prosody_raw_transformer_multiclass_results.txt', 'w') as file:
        file.write("")

    with torch.no_grad():
        for batch_idx, (words, features, labels, lengths) in enumerate(iterator):
            features = features.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)

            logits = model(features, labels, lengths)  # [batch_size, seq_len, num_classes]
            preds = torch.argmax(logits, dim=2)  # [batch_size, seq_len]

            for i in range(features.size(0)):
                word_sentence = words[i]  # List of words
                gold_labels = labels[i].cpu().numpy().flatten()
                pred_labels = preds[i].cpu().numpy().flatten()

                # Adjust labels back by -1
                gold_labels = gold_labels - 1
                pred_labels = pred_labels - 1
                PADDING_VALUE_EVAL = -1  # Update padding value for evaluation

                # Clean up the sentence by excluding padding positions
                cleaned_words, cleaned_gold_labels, cleaned_pred_labels = clean_up_sentence(
                    word_sentence, gold_labels, pred_labels, padding_value=PADDING_VALUE_EVAL
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
    Plots training and validation metrics over epochs and saves the plot as an image file.
    Args:
        train_losses (list of float): List of training loss values for each epoch.
        val_losses (list of float): List of validation loss values for each epoch.
        val_accuracies (list of float): List of validation accuracy values for each epoch.
        val_precisions (list of float): List of validation precision values for each epoch.
        val_recalls (list of float): List of validation recall values for each epoch.
        val_f1s (list of float): List of validation F1 score values for each epoch.
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
def evaluate_new_set(model, new_dataset_path):
    """
    Evaluate the given model on a new dataset.
    Args:
        model (torch.nn.Module): The trained model to be evaluated.
        new_dataset_path (str): The file path to the new dataset.
    Returns:
        tuple: A tuple containing two lists:
            - all_labels (list): The true labels from the new dataset.
            - all_preds (list): The model's predictions on the new dataset.
    """
    # Load new data
    new_data = load_data(new_dataset_path)
    new_dataset = ProsodyDataset(new_data)
    new_loader = DataLoader(new_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
    
    # Test the model on the new dataset and get predictions
    print('\n\nEvaluation on Held Out Set Dataset:')
    all_labels, all_preds = test_model(model, new_loader)

    return all_labels, all_preds

# Validate label integrity
def validate_labels(datasets, num_classes):
    """
    Validates the labels in the given datasets to ensure they are within the specified range.

    Args:
        datasets (list of datasets): A list of datasets where each dataset is an iterable of tuples.
                                     Each tuple contains three elements, and the third element is expected to be the labels.
        num_classes (int): The number of classes. Labels should be in the range [0, num_classes - 1].

    Raises:
        ValueError: If any label in the datasets is found to be outside the range [0, {num_classes - 1}].

    Example:
        datasets = [
            [(input1, target1, labels1), (input2, target2, labels2)],
            [(input3, target3, labels3), (input4, target4, labels4)]
        ]
        validate_labels(datasets, num_classes=10)
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
    Extracts all unique labels from a list of datasets.

    Args:
        datasets (list): A list of datasets, where each dataset is an iterable of tuples.
                         Each tuple contains three elements, with the third element being
                         the labels.

    Returns:
        set: A set containing all unique labels found in the datasets.
    """
    unique_labels = set()
    for dataset in datasets:
        for _, _, labels in dataset:
            unique_labels.update(labels.numpy().flatten())
    return unique_labels

if __name__ == "__main__":
    """
    Main execution block for training and evaluation of the Transformer model.
    """
    # Set random seed for reproducibility
    set_seed(42)

    # Load data
    json_path = '../prosody/data/ambiguous_prosodic_raw_acoustic_ml_features_train.json'
    data = load_data(json_path)

    # Create a descriptive filename for the model
    dataset_name = "ambiguous_instructions"
    task_name = "prosody-raw-multiclass"
    best_model_filename = f"models/best-transformer-model-{dataset_name}-{task_name}.pt"

    # Split data
    train_data, val_data, test_data = split_data(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)

    # Create datasets
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

    # Retrieve a sample batch to determine feature dimensions
    sample_words, sample_features, sample_labels, sample_lengths = next(iter(train_loader))
    feature_dim = sample_features.shape[2]

    # Determine number of classes based on all datasets
    all_unique_labels = get_all_unique_labels([train_dataset, val_dataset, test_dataset])

    # Update the number of classes to include SOS token
    NUM_CLASSES = len(all_unique_labels) + 2  # +1 for label shifting, +1 for SOS token
    print(f"All unique labels across datasets: {sorted(all_unique_labels)}")
    print(f"Model Training with {NUM_CLASSES} classes")

    # Define special token indices
    SOS_IDX = NUM_CLASSES - 1  # Start-of-Sequence token index

    # Define model hyperparameters
    HIDDEN_DIM = 320
    OUTPUT_DIM = NUM_CLASSES
    NUM_LAYERS = 5
    DROPOUT = 0.1
    NUM_HEADS = 8
    LR = 7.09586457311747e-05
    WEIGHT_DECAY = 2.0597041763339934e-06

    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Initialize Encoder and Decoder
    encoder = TransformerEncoder(feature_dim, HIDDEN_DIM, NUM_LAYERS, DROPOUT, num_heads=NUM_HEADS).to(device)
    decoder = TransformerDecoder(HIDDEN_DIM, OUTPUT_DIM, NUM_LAYERS, DROPOUT, num_classes=NUM_CLASSES, num_heads=NUM_HEADS).to(device)
    model = TransformerSeq2Seq(encoder, decoder).to(device)

    # Validate labels before training
    validate_labels([train_dataset, val_dataset, test_dataset], NUM_CLASSES)

    # Define optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), 
                           lr=LR, 
                           weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Define loss function
    criterion = nn.CrossEntropyLoss(ignore_index=PADDING_VALUE)

    # Initialize lists to store metrics
    train_losses = []
    val_losses = []
    val_accuracies = []
    val_precisions = []
    val_recalls = []
    val_f1s = []

    # Define training parameters
    N_EPOCHS = 20
    CLIP = 1

    # Initialize early stopping
    early_stopping = EarlyStopping(patience=10, min_delta=0.001)
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

        # Update the learning rate scheduler
        scheduler.step()

        # Save the model if validation loss has decreased
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), best_model_filename)

        # Print training progress
        print(f'Epoch: {epoch+1:02} | Train Loss: {train_loss:.4f} | Val. Loss: {valid_loss:.4f} | '
              f'Val. Acc: {valid_acc*100:.2f}% | Precision: {valid_precision:.4f} | '
              f'Recall: {valid_recall:.4f} | F1 Score: {valid_f1:.4f}')

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
    eval_json = "../prosody/data/ambiguous_prosodic_raw_acoustic_ml_features_eval.json"
    # Evaluate the model on the new dataset
    true_labels, predicted_labels = evaluate_new_set(model, eval_json)

    # Log directory
    log_dir = "../prosody/outputs"

    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Class names
    class_names = sorted(list(set(true_labels)))

    # Compute precision, recall, f1-score, and support for each class
    class_precision, class_recall, class_f1, class_support = precision_recall_fscore_support(
        true_labels, predicted_labels, average=None, zero_division=0)

    print("Class Support:", class_support)
    # Compute the multilabel confusion matrix
    confusion_matrices = multilabel_confusion_matrix(true_labels, predicted_labels)

    # Write class-wise metrics to a file
    with open(f"{log_dir}/prosody_raw_classwise_metrics.txt", "w") as f:
        for i, class_name in enumerate(class_names):
            f.write(f"{class_name}:\n")
            f.write(f"  Precision: {class_precision[i]:.4f}\n")
            f.write(f"  Recall: {class_recall[i]:.4f}\n")
            f.write(f"  F1-Score: {class_f1[i]:.4f}\n")
            f.write(f"  Support (True instances in eval data): {class_support[i]}\n")
            f.write("-" * 40 + "\n")

    # Write confusion matrix to a file
    with open(f"{log_dir}/prosody_raw_confusion_matrix.txt", "w") as f:
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
