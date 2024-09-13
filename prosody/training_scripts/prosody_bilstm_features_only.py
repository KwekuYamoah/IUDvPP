"""
    This script contains various classes and functions for training and evaluating a prosody BiLSTM model.

    Classes:
        - EarlyStopping: Class for implementing early stopping during training.
        - ProsodyDataset: A dataset class for handling prosody data.
        - AttentionLayer: Class for implementing the attention layer in the decoder.
        - MultiAttention: Class for calculating attention weights.
        - Encoder: Class representing an Encoder module for a prosody BiLSTM model.
        - Decoder: Class representing a Decoder module for a prosody BiLSTM model.
        - Seq2Seq: Class representing a Seq2Seq model for sequence-to-sequence tasks.
    Functions:
        - set_seed(seed): Set the random seed for reproducibility.
        - load_data(json_path): Load data from a JSON file.
        - split_data(data, train_ratio, val_ratio, test_ratio): Splits the given data into training, validation, and test sets.
        - preprocess_text(words): Preprocesses a list of words by splitting them into individual tokens.
        - collate_fn(batch, pad_value): Collates a batch of data by padding the features and labels sequences.
        - train(model, iterator, optimizer, criterion, num_classes, pad_value): Trains the model using the given iterator and optimizer.
        - evaluate(model, iterator, criterion, num_classes, pad_value): Evaluate the performance of a model on a given iterator.
        - test_model(model, iterator, num_classes, pad_value): Test the given model on the provided iterator.
"""

import json
import random
import re
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight



def set_seed(seed):
    """
    Set the random seed for reproducibility.

    Args:
        seed (int): The seed value to set.

    Returns:
        None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class EarlyStopping:
    """
    Class for implementing early stopping during training.

    Attributes:
        patience (int): The number of epochs to wait for improvement before stopping.
        min_delta (float): The minimum change in validation loss required to be considered as improvement.
        best_loss (float): The best validation loss achieved so far.
        counter (int): The number of epochs without improvement.
        early_stop (bool): Flag indicating whether to stop training early.

    Methods:
        __call__(val_loss): Updates the best_loss and counter based on the given validation loss.
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
    """
    Load data from a JSON file.

    Parameters:
    json_path (str): The path to the JSON file.

    Returns:
    dict: The loaded data from the JSON file.
    """
    with open(json_path, 'r') as file:
        data = json.load(file)
    return data

def split_data(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Splits the given data into training, validation, and test sets based on the provided ratios.

    Args:
        data (list): The data to be split.
        train_ratio (float, optional): The ratio of training data. Defaults to 0.8.
        val_ratio (float, optional): The ratio of validation data. Defaults to 0.1.
        test_ratio (float, optional): The ratio of test data. Defaults to 0.1.

    Returns:
        tuple: A tuple containing three lists: training data, validation data, and test data.
    """
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1"
    total = len(data)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    test_size = total - train_size - val_size
    return random_split(list(data.items()), [train_size, val_size, test_size])

def preprocess_text(words):
    """
    Preprocesses a list of words by splitting them into individual tokens.

    Args:
        words (list): A list of words to be preprocessed.

    Returns:
        list: A list of processed tokens.

    Example:
        >>> words = ["Hello, world!", "This is a sentence."]
        >>> preprocess_text(words)
        ['Hello', ',', 'world', '!', 'This', 'is', 'a', 'sentence', '.']
    """
    processed_words = []
    for word in words:
        processed_words.extend(re.findall(r"[\w']+|[.,!?;]", word))
    return processed_words

class ProsodyDataset(Dataset):
    
    def __init__(self, data):
        self.data = data
        self.entries = list(data.items())

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        key, item = self.entries[idx]
        features = torch.tensor(item['features'], dtype=torch.float32)
        labels = torch.tensor(item['labels'], dtype=torch.float32)
        return features, labels


def collate_fn(batch, pad_value=-1):
    """
    Collates a batch of data by padding the features and labels sequences.

    Args:
        batch (list): A list of tuples containing the features and labels sequences.
        pad_value (int, optional): The value used for padding. Defaults to -1.

    Returns:
        tuple: A tuple containing the padded features sequence, padded labels sequence, and mask for valid labels.
    """
    pass
    features = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    features_padded = torch.nn.utils.rnn.pad_sequence(features, batch_first=True, padding_value=pad_value)
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=pad_value)

    # Create a mask for the labels (1 for valid labels, 0 for padding)
    mask = (labels_padded != pad_value).float()

    return features_padded, labels_padded, mask

class AttentionLayer(nn.Module):
    """
    Initializes the AttentionLayer module.

    Args:
        hidden_dim (int): The dimension of the hidden state.

    """

    """
    Performs the forward pass of the AttentionLayer module.

    Args:
        hidden (torch.Tensor): The hidden state.
        encoder_outputs (torch.Tensor): The encoder outputs.

    Returns:
        torch.Tensor: The attention weights.

    """
    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.attn = nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        src_len = encoder_outputs.size(1)
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        return torch.softmax(attention, dim=1)

class MultiAttention(nn.Module):
    """
    MultiAttention module for calculating attention weights.

    Args:
        hidden_dim (int): The dimension of the hidden state.
        num_layers (int): The number of attention layers.

    Attributes:
        attention_layers (nn.ModuleList): A list of AttentionLayer instances.

    Methods:
        forward(hidden, encoder_outputs): Calculates the attention weights for the given hidden state and encoder outputs.

    Returns:
        attn_weights (torch.Tensor): The attention weights.

    """
    def __init__(self, hidden_dim, num_layers):
        super(MultiAttention, self).__init__()
        self.attention_layers = nn.ModuleList([AttentionLayer(hidden_dim) for _ in range(num_layers)])

    def forward(self, hidden, encoder_outputs):
        attn_weights = []
        for layer in self.attention_layers:
            attn_weight = layer(hidden, encoder_outputs)
            attn_weights.append(attn_weight)
        attn_weights = torch.stack(attn_weights, dim=1).mean(dim=1)
        return attn_weights

class Encoder(nn.Module):
    """
    This class represents an Encoder module for a prosody BiLSTM model.

    Parameters:
    - feature_dim (int): The dimensionality of the input features.
    - hidden_dim (int): The dimensionality of the hidden states in the LSTM.
    - num_layers (int): The number of LSTM layers.
    - dropout (float): The dropout probability.

    Methods:
    - forward(features): Performs a forward pass through the LSTM encoder.

    Returns:
    - outputs (Tensor): The output sequence from the LSTM encoder.
    - hidden (Tensor): The hidden state of the last LSTM layer.
    - cell (Tensor): The cell state of the last LSTM layer.
    """
    def __init__(self, feature_dim, hidden_dim, num_layers, dropout):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(feature_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True, bidirectional=True)

    def forward(self, features):
        outputs, (hidden, cell) = self.lstm(features)
        return outputs, hidden, cell

class Decoder(nn.Module):
    """
    Initializes the Decoder module.
    Args:
        hidden_dim (int): The dimensionality of the hidden state in the LSTM.
        output_dim (int): The dimensionality of the output.
        num_layers (int): The number of LSTM layers.
        dropout (float): The dropout probability.
        num_attention_layers (int): The number of attention layers.
        num_classes (int, optional): The number of classes. Defaults to 2.
    """
    """
    Performs a forward pass of the Decoder module.
    Args:
        encoder_outputs (torch.Tensor): The output of the encoder module.
        hidden (torch.Tensor): The hidden state of the LSTM.
        cell (torch.Tensor): The cell state of the LSTM.
    Returns:
        torch.Tensor: The predictions of the Decoder module.
        tuple: The updated hidden and cell states.
    """
    def __init__(self, hidden_dim, output_dim, num_layers, dropout, num_attention_layers, num_classes=2):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.lstm = nn.LSTM(hidden_dim * 4, hidden_dim, num_layers, dropout=dropout, batch_first=True, bidirectional=True)

        # Always output logits for CrossEntropyLoss (both binary and multi-class)
        self.fc = nn.Linear(hidden_dim * 2, output_dim * num_classes)
        
        self.attention = MultiAttention(hidden_dim, num_attention_layers)

    def forward(self, encoder_outputs, hidden, cell):
        attn_weights = self.attention(hidden[-1], encoder_outputs)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        lstm_input = torch.cat((context.unsqueeze(1).repeat(1, encoder_outputs.size(1), 1), encoder_outputs), dim=2)
        outputs, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        predictions = self.fc(outputs)

        return predictions, (hidden, cell)

class Seq2Seq(nn.Module):
    """
    Seq2Seq class for sequence-to-sequence model.

    Args:
        encoder (nn.Module): The encoder module.
        decoder (nn.Module): The decoder module.

    Attributes:
        encoder (nn.Module): The encoder module.
        decoder (nn.Module): The decoder module.

    Methods:
        forward(features): Performs forward pass of the model.

    Returns:
        outputs (Tensor): The output tensor from the decoder.

    """
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, features):
        encoder_outputs, hidden, cell = self.encoder(features)
        outputs, (hidden, cell) = self.decoder(encoder_outputs, hidden, cell)
        return outputs


def train(model, iterator, optimizer, criterion, num_classes=2, pad_value=-1, CLIP=1):
    """
    Trains the model using the given iterator and optimizer for binary or multi-class
    classification.

    Args:
        model (torch.nn.Module): The model to be trained.
        iterator (torch.utils.data.DataLoader): The data iterator.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        criterion (torch.nn.Module): The loss function.
        num_classes (int, optional): The number of classes. Defaults to 2.
        pad_value (int, optional): The padding value. Defaults to -1.

    Returns:
        float: The average loss per epoch.
    """
    model.train()
    epoch_loss = 0
    for features, labels, mask in iterator:
        features = features.to(device)
        labels = labels.to(device)
        mask = mask.to(device)

        optimizer.zero_grad()
        output = model(features)

        output = output.view(-1, num_classes)  # Output shape: [batch_size * seq_len, num_classes]
        labels = labels.view(-1).long()  # labels are [batch_size * seq_len]

        loss = criterion(output, labels)
        loss = (loss * mask.view(-1)).sum() / mask.sum()  # masking for valid tokens

        loss.backward()
        # gradient clipping
        # torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)

        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion, num_classes=2, pad_value=-1):
    """
    Evaluate the performance of a model on a given iterator for classification.

    Args:
        model (torch.nn.Module): The model to evaluate.
        iterator (torch.utils.data.DataLoader): The data iterator.
        criterion (torch.nn.Module): The loss criterion.
        num_classes (int, optional): The number of classes. Defaults to 2.
        pad_value (int, optional): The padding value. Defaults to -1.

    Returns:
        float: The average loss over the iterator.
        float: The accuracy score.
        float: The precision score.
        float: The recall score.
        float: The F1 score.
    """
    model.eval()
    epoch_loss = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for features, labels, mask in iterator:
            features = features.to(device)
            labels = labels.to(device)
            mask = mask.to(device)

            output = model(features)

            output = output.view(-1, num_classes)
            labels = labels.view(-1).long()

            # Get the predictions
            preds = torch.argmax(output, dim=1)

            loss = criterion(output, labels)
            loss = (loss * mask.view(-1)).sum() / mask.sum()  # Apply masking for valid tokens

            epoch_loss += loss.item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return epoch_loss / len(iterator), accuracy, precision, recall, f1


def test_model(model, iterator, num_classes=2, pad_value=-1):
    """
    Test the given model on the provided iterator.
    Args:
        model (torch.nn.Module): The model to be tested.
        iterator (torch.utils.data.DataLoader): The data iterator.
        num_classes (int, optional): The number of classes. Defaults to 2.
        pad_value (int, optional): The padding value. Defaults to -1.
    Returns:
        tuple: A tuple containing the list of all true labels and the list of all predicted labels.
    """
    model.eval()
    all_labels = []
    all_preds = []
    with open('./outputs/prosody_bilstm_features_results.txt', 'w') as file:
        file.write("")
    with torch.no_grad():
        for features, labels, mask in iterator:
            features = features.to(device)
            labels = labels.to(device)
            mask = mask.to(device)
            output = model(features)

            # Get predictions
            preds = torch.argmax(output, dim=2)

            for i in range(features.shape[0]):
                gold_labels = labels[i].cpu().numpy().flatten()
                pred_labels = preds[i].cpu().numpy().flatten()
                mask_i = mask[i].cpu().numpy().flatten()

                # Filter out padding tokens using the mask
                gold_labels = gold_labels[mask_i > 0]
                pred_labels = pred_labels[mask_i > 0]

                # Write to file
                data = {
                    'Gold Label': gold_labels.tolist(),
                    'Predicted Label': pred_labels.tolist()
                }
                df = pd.DataFrame(data)
                with open('./outputs/prosody_bilstm_features_results.txt', 'a') as file:
                    file.write(df.to_string(index=False))
                    file.write("\n" + "-" * 50 + "\n")

                all_labels.extend(gold_labels.tolist())
                all_preds.extend(pred_labels.tolist())

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    print(f'Test Accuracy: {accuracy*100:.2f}%')
    print(f'Test Precision: {precision:.2f}')
    print(f'Test Recall: {recall:.2f}')
    print(f'Test F1 Score: {f1:.2f}')
    
    return all_labels, all_preds


def plot_metrics(train_losses, val_losses, val_accuracies, val_precisions, val_recalls, val_f1s):
    """
    Plots the metrics for training and validation.

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
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 3, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.legend()
    plt.subplot(2, 3, 2)
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.legend()
    plt.subplot(2, 3, 3)
    plt.plot(epochs, val_precisions, label='Validation Precision')
    plt.legend()
    plt.subplot(2, 3, 4)
    plt.plot(epochs, val_recalls, label='Validation Recall')
    plt.legend()
    plt.subplot(2, 3, 5)
    plt.plot(epochs, val_f1s, label='Validation F1 Score')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./outputs/bilstm_features_version.png')

if __name__ == "__main__":
    
    seed = 42 
    set_seed(seed)
    json_path = '../prosody/multi_label_features.json'  # Change data path here
    data = load_data(json_path)

    train_data, val_data, test_data = split_data(data)

    train_dataset = ProsodyDataset(dict(train_data))
    val_dataset = ProsodyDataset(dict(val_data))
    test_dataset = ProsodyDataset(dict(test_data))

    print(f'Train size: {len(train_dataset)}')
    print(f'Validation size: {len(val_dataset)}')
    print(f'Test size: {len(test_dataset)}')

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=lambda batch: collate_fn(batch, pad_value=-1))
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=lambda batch: collate_fn(batch, pad_value=-1))
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=lambda batch: collate_fn(batch, pad_value=-1))

    HIDDEN_DIM = 128
    OUTPUT_DIM = 1
    NUM_LAYERS = 2
    DROPOUT = 0.5
    NUM_ATTENTION_LAYERS = 2
    NUM_CLASSES = 4  # Change this depending on the number of classes (set to 2 for binary classification)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    feature_dim = next(iter(train_loader))[0].shape[2]

    encoder = Encoder(feature_dim, HIDDEN_DIM, NUM_LAYERS, DROPOUT).to(device)
    decoder = Decoder(HIDDEN_DIM, OUTPUT_DIM, NUM_LAYERS, DROPOUT, NUM_ATTENTION_LAYERS, num_classes=NUM_CLASSES).to(device)
    model = Seq2Seq(encoder, decoder).to(device)

    # CrossEntropyLoss for both binary and multi-class classification
    # Concatenate all labels into a single tensor
    # labels = torch.cat([torch.tensor(item['labels']) for item in data.values()])

    # # Remove any invalid labels, such as padding tokens (-1)
    # valid_labels = labels[labels != -1].numpy()  # Convert to numpy for sklearn

    # # Compute unique valid classes
    # valid_classes = np.unique(valid_labels)

    # # Compute class weights for the valid labels
    # class_weights = compute_class_weight('balanced', classes=valid_classes, y=valid_labels)

    # # Convert class weights back to a tensor and move to the device (GPU/CPU)
    # class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

    train_losses = []
    val_losses = []
    val_accuracies = []
    val_precisions = []
    val_recalls = []
    val_f1s = []

    N_EPOCHS = 500
    CLIP = 1

    early_stopping = EarlyStopping(patience=30, min_delta=0.001)
    best_valid_loss = float('inf')

    for epoch in range(N_EPOCHS):
        train_loss = train(model, train_loader, optimizer, criterion, num_classes=NUM_CLASSES)
        valid_loss, valid_acc, valid_precision, valid_recall, valid_f1 = evaluate(model, val_loader, criterion, num_classes=NUM_CLASSES)

        train_losses.append(train_loss)
        val_losses.append(valid_loss)
        val_accuracies.append(valid_acc)
        val_precisions.append(valid_precision)
        val_recalls.append(valid_recall)
        val_f1s.append(valid_f1)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'models/best-model-features-version.pt')

        print(f'Epoch: {epoch+1:02}')
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}% |  Precision: {valid_precision:.2f} |  Recall: {valid_recall:.2f} |  F1 Score: {valid_f1:.2f}')

        early_stopping(valid_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    model.load_state_dict(torch.load('models/best-model-features-version.pt'))
    test_model(model, test_loader, num_classes=NUM_CLASSES)
    plot_metrics(train_losses, val_losses, val_accuracies, val_precisions, val_recalls, val_f1s)
