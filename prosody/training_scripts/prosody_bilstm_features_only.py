import json
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

class EarlyStopping:
    """
    Class for early stopping during training based on validation loss.

    Args:
        patience (int): Number of epochs to wait for improvement in validation loss before stopping.
        min_delta (float): Minimum change in validation loss to be considered as improvement.

    Attributes:
        patience (int): Number of epochs to wait for improvement in validation loss before stopping.
        min_delta (float): Minimum change in validation loss to be considered as improvement.
        best_loss (float): Best validation loss achieved so far.
        counter (int): Number of epochs without improvement in validation loss.
        early_stop (bool): Flag indicating whether to stop training early.

    Methods:
        __call__(val_loss): Update the best validation loss and check if early stopping criteria are met.

    """

    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        """
        Update the best validation loss and check if early stopping criteria are met.

        Args:
            val_loss (float): Current validation loss.

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

def load_data(json_path):
    """
    Load data from a JSON file.

    Args:
        json_path (str): The path to the JSON file.

    Returns:
        dict: The loaded data from the JSON file.
    """
    with open(json_path, 'r') as file:
        data = json.load(file)
    return data

def split_data(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Splits the given data into train, validation, and test sets based on the provided ratios.

    Args:
        data (dict): The data to be split, represented as a dictionary.
        train_ratio (float): The ratio of data to be used for training. Default is 0.8.
        val_ratio (float): The ratio of data to be used for validation. Default is 0.1.
        test_ratio (float): The ratio of data to be used for testing. Default is 0.1.

    Returns:
        tuple: A tuple containing three lists, representing the train, validation, and test sets respectively.
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

    """
    processed_words = []
    for word in words:
        processed_words.extend(re.findall(r"[\w']+|[.,!?;]", word))
    return processed_words

class ProsodyDataset(Dataset):
    """
    A PyTorch dataset class for handling prosody data.

    Args:
        data (dict): A dictionary containing the data with features and labels.

    Attributes:
        data (dict): The input data dictionary.
        entries (list): A list of key-value pairs from the data dictionary.

    Methods:
        __len__(): Returns the length of the dataset.
        __getitem__(idx): Returns the features and labels for a given index.

    """

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

def collate_fn(batch):
    """
    Collate function for batching sequences of features and labels.

    Args:
        batch (list): A list of tuples, where each tuple contains a feature sequence and its corresponding label sequence.

    Returns:
        tuple: A tuple containing the padded feature sequence and the padded label sequence.
    """
    features = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    features_padded = torch.nn.utils.rnn.pad_sequence(features, batch_first=True, padding_value=0.0)
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0.0)

    max_length = features_padded.size(1)
    labels_padded = torch.nn.functional.pad(labels_padded, (0, max_length - labels_padded.size(1)), "constant", 0)

    return features_padded, labels_padded

class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        """
        Initializes the AttentionLayer module.

        Args:
            hidden_dim (int): The dimensionality of the hidden state.

        """
        super(AttentionLayer, self).__init__()
        self.attn = nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        """
        Performs the forward pass of the AttentionLayer module.

        Args:
            hidden (torch.Tensor): The hidden state.
            encoder_outputs (torch.Tensor): The encoder outputs.

        Returns:
            torch.Tensor: The attention weights.

        """
        src_len = encoder_outputs.size(1)
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        return torch.softmax(attention, dim=1)

class MultiAttention(nn.Module):
    """
    Multi-head attention module.

    Args:
        hidden_dim (int): The dimension of the hidden state.
        num_layers (int): The number of attention layers.

    Attributes:
        attention_layers (nn.ModuleList): List of attention layers.

    """

    def __init__(self, hidden_dim, num_layers):
        super(MultiAttention, self).__init__()
        self.attention_layers = nn.ModuleList([AttentionLayer(hidden_dim) for _ in range(num_layers)])

    def forward(self, hidden, encoder_outputs):
        """
        Forward pass of the multi-head attention module.

        Args:
            hidden (torch.Tensor): The hidden state.
            encoder_outputs (torch.Tensor): The encoder outputs.

        Returns:
            torch.Tensor: The attention weights.

        """
        attn_weights = []
        for layer in self.attention_layers:
            attn_weight = layer(hidden, encoder_outputs)
            attn_weights.append(attn_weight)
        attn_weights = torch.stack(attn_weights, dim=1).mean(dim=1)
        return attn_weights

class Encoder(nn.Module):
    """
    Encoder module that performs forward pass through a bidirectional LSTM.

    Args:
        feature_dim (int): The dimensionality of the input features.
        hidden_dim (int): The dimensionality of the hidden state of the LSTM.
        num_layers (int): The number of LSTM layers.
        dropout (float): The dropout probability.

    Attributes:
        lstm (nn.LSTM): The bidirectional LSTM layer.

    """

    def __init__(self, feature_dim, hidden_dim, num_layers, dropout):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(feature_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True, bidirectional=True)

    def forward(self, features):
        """
        Forward pass through the encoder.

        Args:
            features (torch.Tensor): The input features.

        Returns:
            outputs (torch.Tensor): The output sequence from the LSTM.
            hidden (torch.Tensor): The hidden state of the LSTM.
            cell (torch.Tensor): The cell state of the LSTM.

        """
        outputs, (hidden, cell) = self.lstm(features)
        return outputs, hidden, cell

class Decoder(nn.Module):
    """
    Decoder module for the sequence-to-sequence model.

    Args:
        hidden_dim (int): The dimensionality of the hidden state of the LSTM.
        output_dim (int): The dimensionality of the output.
        num_layers (int): The number of LSTM layers.
        dropout (float): The dropout probability.
        num_attention_layers (int): The number of attention layers.

    Attributes:
        hidden_dim (int): The dimensionality of the hidden state of the LSTM.
        lstm (nn.LSTM): The LSTM layer.
        fc (nn.Linear): The fully connected layer.
        sigmoid (nn.Sigmoid): The sigmoid activation function.
        attention (MultiAttention): The multi-head attention module.

    """

    def __init__(self, hidden_dim, output_dim, num_layers, dropout, num_attention_layers):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(hidden_dim * 4, hidden_dim, num_layers, dropout=dropout, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.attention = MultiAttention(hidden_dim, num_attention_layers)

    def forward(self, encoder_outputs, hidden, cell):
        """
        Forward pass of the decoder.

        Args:
            encoder_outputs (torch.Tensor): The output of the encoder.
            hidden (torch.Tensor): The hidden state of the LSTM.
            cell (torch.Tensor): The cell state of the LSTM.

        Returns:
            predictions (torch.Tensor): The output predictions.
            (hidden, cell) (tuple): The updated hidden and cell states.

        """
        attn_weights = self.attention(hidden[-1], encoder_outputs)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        lstm_input = torch.cat((context.unsqueeze(1).repeat(1, encoder_outputs.size(1), 1), encoder_outputs), dim=2)
        outputs, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        predictions = self.sigmoid(self.fc(outputs))
        return predictions, (hidden, cell)

class Seq2Seq(nn.Module):
    """
    A sequence-to-sequence model that consists of an encoder and a decoder.

    Args:
        encoder (nn.Module): The encoder module.
        decoder (nn.Module): The decoder module.
    """

    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, features):
        """
        Forward pass of the Seq2Seq model.

        Args:
            features (torch.Tensor): Input features.

        Returns:
            torch.Tensor: Output predictions.
        """
        encoder_outputs, hidden, cell = self.encoder(features)
        outputs, (hidden, cell) = self.decoder(encoder_outputs, hidden, cell)
        return outputs

def train(model, iterator, optimizer, criterion):
    """
    Trains the model on the given iterator using the specified optimizer and criterion.

    Args:
        model (torch.nn.Module): The model to be trained.
        iterator (torch.utils.data.DataLoader): The data iterator.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        criterion (torch.nn.Module): The loss criterion used for training.

    Returns:
        float: The average loss per batch during training.
    """
    model.train()
    epoch_loss = 0
    for features, labels in iterator:
        features = features.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        output = model(features)
        output = output.view(-1)
        labels = labels.view(-1).float()
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    """
    Evaluate the performance of a model on a given dataset.

    Args:
        model (torch.nn.Module): The model to evaluate.
        iterator (torch.utils.data.DataLoader): The data iterator.
        criterion (torch.nn.Module): The loss criterion.

    Returns:
        tuple: A tuple containing the average loss, accuracy, precision, recall, and F1 score.
    """
    model.eval()
    epoch_loss = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for features, labels in iterator:
            features = features.to(device)
            labels = labels.to(device)
            output = model(features)
            output = output.view(-1)
            labels = labels.view(-1).float()
            loss = criterion(output, labels)
            epoch_loss += loss.item()
            preds = (output > 0.4).float()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    return epoch_loss / len(iterator), accuracy, precision, recall, f1

def test_model(model, iterator):
    """
    Test the given model on the provided iterator and calculate evaluation metrics.

    Args:
        model (torch.nn.Module): The trained model to be tested.
        iterator (torch.utils.data.DataLoader): The data iterator for testing.

    Returns:
        tuple: A tuple containing two lists - `all_labels` and `all_preds`.
            `all_labels` (list): A list of all true labels.
            `all_preds` (list): A list of all predicted labels.
    """
    model.eval()
    all_labels = []
    all_preds = []
    with open('./outputs/prosody_bilstm_features_results.txt', 'w') as file:
        file.write("")
    with torch.no_grad():
        for features, labels in iterator:
            features = features.to(device)
            labels = labels.to(device)
            output = model(features)
            preds = (output > 0.4).float()

            for i in range(features.shape[0]):
                gold_labels = labels[i].cpu().numpy().flatten()
                pred_labels = preds[i].cpu().numpy().flatten()

                data = {
                    'Gold Label': gold_labels.tolist(),
                    'Predicted Label': pred_labels.tolist()
                }

                df = pd.DataFrame(data)
                with open('./outputs/prosody_bilstm_features_results.txt', 'a') as file:
                    file.write(df.to_string(index=False))
                    file.write("\n" + "-" * 50 + "\n")

            all_labels.extend(labels.cpu().numpy().flatten().tolist())
            all_preds.extend(preds.cpu().numpy().flatten().tolist())

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
    Plots the training and validation metrics over epochs.

    Args:
        train_losses (list): List of training losses.
        val_losses (list): List of validation losses.
        val_accuracies (list): List of validation accuracies.
        val_precisions (list): List of validation precisions.
        val_recalls (list): List of validation recalls.
        val_f1s (list): List of validation F1 scores.
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

def clean_up_sentence(words, gold_labels, pred_labels):
    """
    Cleans up the sentence by removing trailing occurrences of the word 'the' from the given lists of words, gold labels, and predicted labels.

    Args:
        words (list): The list of words in the sentence.
        gold_labels (list): The list of gold labels corresponding to each word in the sentence.
        pred_labels (list): The list of predicted labels corresponding to each word in the sentence.

    Returns:
        tuple: A tuple containing the filtered gold labels and filtered predicted labels after removing trailing occurrences of 'the'.
    """
    end_index = len(words) - 1
    while end_index >= 0 and words[end_index] == 'the':
        end_index -= 1

    filtered_gold_labels = gold_labels[:end_index+1]
    filtered_pred_labels = pred_labels[:end_index+1]

    for i in range(end_index+1, len(words)):
        filtered_gold_labels.append(gold_labels[i])
        filtered_pred_labels.append(pred_labels[i])

    return filtered_gold_labels, filtered_pred_labels

if __name__ == "__main__":
    json_path = '../prosody/reconstructed_extracted_features.json'
    data = load_data(json_path)

    train_data, val_data, test_data = split_data(data)

    train_dataset = ProsodyDataset(dict(train_data))
    val_dataset = ProsodyDataset(dict(val_data))
    test_dataset = ProsodyDataset(dict(test_data))

    print(f'Train size: {len(train_dataset)}')
    print(f'Validation size: {len(val_dataset)}')
    print(f'Test size: {len(test_dataset)}')

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    HIDDEN_DIM = 128
    OUTPUT_DIM = 1
    NUM_LAYERS = 2
    DROPOUT = 0.5
    NUM_ATTENTION_LAYERS = 2

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    feature_dim = next(iter(train_loader))[0].shape[2]

    encoder = Encoder(feature_dim, HIDDEN_DIM, NUM_LAYERS, DROPOUT).to(device)
    decoder = Decoder(HIDDEN_DIM, OUTPUT_DIM, NUM_LAYERS, DROPOUT, NUM_ATTENTION_LAYERS).to(device)
    model = Seq2Seq(encoder, decoder).to(device)

    sample_features, _ = next(iter(train_loader))
    summary(model, input_data=sample_features, device=device, depth=6)

    optimizer = optim.Adam(model.parameters(), weight_decay=1e-5)
    criterion = nn.BCELoss()

    train_losses = []
    val_losses = []
    val_accuracies = []
    val_precisions = []
    val_recalls = []
    val_f1s = []

    N_EPOCHS = 500
    CLIP = 1

    early_stopping = EarlyStopping(patience=15, min_delta=0.001)
    best_valid_loss = float('inf')

    for epoch in range(N_EPOCHS):
        train_loss = train(model, train_loader, optimizer, criterion)
        valid_loss, valid_acc, valid_precision, valid_recall, valid_f1 = evaluate(model, val_loader, criterion)

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
    test_model(model, test_loader)
    plot_metrics(train_losses, val_losses, val_accuracies, val_precisions, val_recalls, val_f1s)
