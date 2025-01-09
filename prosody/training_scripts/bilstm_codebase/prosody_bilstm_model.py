"""
Prosody Classification using BiLSTM with Attention

This script implements a sequence-to-sequence model using Bidirectional LSTM (BiLSTM)
with attention mechanisms for prosody classification. It includes data loading, preprocessing,
model definition, training, evaluation, testing, and visualization of training metrics.

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
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt

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

    Each sample consists of words, associated features, and labels.
    """
    def __init__(self, data):
        """
        Initializes the ProsodyDataset.

        Args:
            data (dict): Dictionary containing prosody data with sentence keys.
        """
        self.data = data
        self.sentences = list(data.keys())

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.sentences)

    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (words, features, labels)
                - words (list): List of words in the sentence.
                - features (torch.Tensor): Tensor of features associated with the words.
                - labels (torch.Tensor): Tensor of labels for each word.
        """
        sentence_key = self.sentences[idx]
        sentence_data = self.data[sentence_key]
        words = sentence_data['words']
        labels = torch.tensor(sentence_data['labels'], dtype=torch.long)
        # Uncomment the following line if positions are used
        # positions = torch.tensor(sentence_data['positions'], dtype=torch.long)
        features = torch.tensor(sentence_data['features'], dtype=torch.float)
        return words, features, labels  # , positions

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
        tuple: (words, features_padded, labels_padded, lengths)
            - words (list): List of word lists.
            - features_padded (torch.Tensor): Padded feature tensors (batch_size, max_seq_len, feature_dim).
            - labels_padded (torch.Tensor): Padded label tensors (batch_size, max_seq_len).
            - lengths (torch.Tensor): Original lengths of each sequence (batch_size).
    """
    words, features, labels = zip(*batch)  # , positions
    # Pad feature sequences to the same length
    features_padded = torch.nn.utils.rnn.pad_sequence(features, batch_first=True)
    # Pad label sequences to the same length
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True)
    # Uncomment the following line if positions are used
    # positions_padded = torch.nn.utils.rnn.pad_sequence(positions, batch_first=True)
    # Record the lengths of each sequence before padding
    lengths = torch.tensor([len(w) for w in words])
    return words, features_padded, labels_padded  # , positions_padded

# ==============================
# Data Loaders Creation Function
# ==============================

def create_data_loaders(data, batch_size=32, val_split=0.2, test_split=0.1):
    """
    Creates DataLoader instances for training, validation, and testing.

    Args:
        data (dict): Dictionary containing prosody data.
        batch_size (int, optional): Number of samples per batch. Defaults to 32.
        val_split (float, optional): Proportion of data for validation. Defaults to 0.2.
        test_split (float, optional): Proportion of data for testing. Defaults to 0.1.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
            - train_loader (DataLoader): DataLoader for training data.
            - val_loader (DataLoader): DataLoader for validation data.
            - test_loader (DataLoader): DataLoader for test data.
    """
    dataset = ProsodyDataset(data)
    val_size = int(len(dataset) * val_split)
    test_size = int(len(dataset) * test_split)
    train_size = len(dataset) - val_size - test_size
    # Split the dataset into train, validation, and test sets
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    # Create DataLoader instances for each split
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return train_loader, val_loader, test_loader

# ==============================
# Encoder Model
# ==============================

class Encoder(nn.Module):
    """
    Encoder module using a Bidirectional LSTM.

    Processes input features and encodes them into hidden representations.
    """
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        """
        Initializes the Encoder.

        Args:
            input_dim (int): Dimension of input features.
            hidden_dim (int): Dimension of LSTM hidden states.
            num_layers (int): Number of LSTM layers.
            dropout (float): Dropout rate between LSTM layers.
        """
        super(Encoder, self).__init__()
        # Bidirectional LSTM with specified input and hidden dimensions
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True, bidirectional=True)
    
    def forward(self, x):
        """
        Forward pass of the Encoder.

        Args:
            x (torch.Tensor): Input features (batch_size, seq_len, input_dim).

        Returns:
            tuple: (outputs, hidden, cell)
                - outputs (torch.Tensor): LSTM outputs for each time step (batch_size, seq_len, hidden_dim * 2).
                - hidden (torch.Tensor): Final hidden states (num_layers * 2, batch_size, hidden_dim).
                - cell (torch.Tensor): Final cell states (num_layers * 2, batch_size, hidden_dim).
        """
        # Pass input through LSTM
        outputs, (hidden, cell) = self.lstm(x)
        return outputs, hidden, cell

# ==============================
# Decoder Model
# ==============================

class Decoder(nn.Module):
    """
    Decoder module with attention mechanism.

    Generates predictions based on encoder outputs and current hidden state.
    """
    def __init__(self, hidden_dim, output_dim, num_layers, dropout):
        """
        Initializes the Decoder.

        Args:
            hidden_dim (int): Dimension of LSTM hidden states.
            output_dim (int): Dimension of output predictions.
            num_layers (int): Number of LSTM layers.
            dropout (float): Dropout rate between LSTM layers.
        """
        super(Decoder, self).__init__()
        # Bidirectional LSTM that takes concatenated context and encoder outputs as input
        self.lstm = nn.LSTM(hidden_dim * 2, hidden_dim, num_layers, dropout=dropout, batch_first=True, bidirectional=True)
        # Fully connected layer to generate output predictions
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # * 2 for bidirectional
    
    def forward(self, x, hidden, cell):
        """
        Forward pass of the Decoder.

        Args:
            x (torch.Tensor): Input to the decoder (batch_size, seq_len, hidden_dim * 2).
            hidden (torch.Tensor): Hidden state from the encoder (num_layers * 2, batch_size, hidden_dim).
            cell (torch.Tensor): Cell state from the encoder (num_layers * 2, batch_size, hidden_dim).

        Returns:
            tuple: (outputs, (hidden, cell))
                - outputs (torch.Tensor): Predicted outputs (batch_size, seq_len, output_dim).
                - hidden (torch.Tensor): Updated hidden states.
                - cell (torch.Tensor): Updated cell states.
        """
        # Pass through LSTM
        outputs, (hidden, cell) = self.lstm(x, (hidden, cell))
        # Apply sigmoid activation to generate binary predictions
        outputs = torch.sigmoid(self.fc(outputs))
        return outputs, (hidden, cell)

# ==============================
# Sequence-to-Sequence Model
# ==============================

class Seq2Seq(nn.Module):
    """
    Sequence-to-Sequence model combining Encoder and Decoder.

    Handles the flow of data from input features through the encoder and decoder to generate predictions.
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

    def forward(self, src, trg):
        """
        Forward pass of the Seq2Seq model.

        Args:
            src (torch.Tensor): Source input features (batch_size, seq_len, input_dim).
            trg (torch.Tensor): Target input features (batch_size, seq_len, input_dim).

        Returns:
            torch.Tensor: Output predictions (batch_size, seq_len, output_dim).
        """
        # Encode the source input
        encoder_outputs, hidden, cell = self.encoder(src)
        # Decode the target input using encoder outputs and hidden states
        outputs, (hidden, cell) = self.decoder(encoder_outputs, hidden, cell)
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
    for _, features, labels in iterator:
        optimizer.zero_grad()  # Reset gradients
        output = model(features, features)  # Forward pass
        output = output.view(-1)  # Flatten output tensor
        labels = labels.view(-1).float()  # Flatten labels tensor
        loss = criterion(output, labels)  # Compute loss
        loss.backward()  # Backpropagation
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
        for _, features, labels in iterator:
            output = model(features, features)  # Forward pass
            output = output.view(-1)  # Flatten output tensor
            labels = labels.view(-1).float()  # Flatten labels tensor
            loss = criterion(output, labels)  # Compute loss
            epoch_loss += loss.item()  # Accumulate loss
            preds = (output > 0.4).float()  # Generate binary predictions with threshold 0.4
            all_labels.extend(labels.cpu().numpy().tolist())  # Collect true labels
            all_preds.extend(preds.cpu().numpy().tolist())    # Collect predicted labels
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
    Tests the model on the test dataset and prints detailed results.

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
    with torch.no_grad():  # Disable gradient computation
        for words, features, labels in iterator:
            output = model(features, features)  # Forward pass
            preds = (output > 0.4).float()  # Generate binary predictions with threshold 0.4
            all_labels.extend(labels.cpu().numpy().tolist())  # Collect true labels
            all_preds.extend(preds.cpu().numpy().tolist())    # Collect predicted labels
            # Print detailed results for each sentence in the batch
            print(f'Sentence: {" ".join(words[0])}')
            print(f'Gold Labels: {labels[0].cpu().numpy()}')
            print(f'Predicted Labels: {preds[0].cpu().numpy().flatten()}')
    # Flatten the lists for metric calculations
    all_labels = [item for sublist in all_labels for item in sublist]
    all_preds = [item for sublist in all_preds for item in sublist]
    # Calculate evaluation metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    # Print metrics
    print(f'Test Accuracy: {accuracy*100:.2f}%')
    print(f'Test Precision: {precision:.2f}')
    print(f'Test Recall: {recall:.2f}')
    print(f'Test F1 Score: {f1:.2f}')
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
    plt.savefig('../prosody/bilstm_metrics.png')
    plt.close()
    print(f'Metrics plot saved to ../prosody/bilstm_metrics.png')

# ==============================
# Main Execution Block
# ==============================

if __name__ == '__main__':
    # ==============================
    # Data Loading and Preparation
    # ==============================

    # Load the data from a JSON file
    with open('../prosody/reconstructed_extracted_features.json', 'r') as f:
        data = json.load(f)

    # Create DataLoader instances for training, validation, and testing
    train_loader, val_loader, test_loader = create_data_loaders(data, batch_size=32)

    # ==============================
    # Model Instantiation
    # ==============================

    # Define model hyperparameters
    INPUT_DIM = len(data['pm04_in_027']['features'][0])  # Dimension of input features
    HIDDEN_DIM = 256      # Dimension of LSTM hidden states
    OUTPUT_DIM = 1        # Output dimension for binary classification
    NUM_LAYERS = 4        # Number of LSTM layers
    DROPOUT = 0.5         # Dropout rate between LSTM layers

    # Set the device to GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the Encoder and Decoder
    encoder = Encoder(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT).to(device)
    decoder = Decoder(HIDDEN_DIM, OUTPUT_DIM, NUM_LAYERS, DROPOUT).to(device)
    # Combine Encoder and Decoder into a Seq2Seq model
    model = Seq2Seq(encoder, decoder).to(device)

    # Uncomment the following line to see the model architecture summary
    # summary(model, input_data=(sample_words, sample_features), device=device)

    # ==============================
    # Optimizer, Scheduler, and Loss Function
    # ==============================

    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)  # Adam optimizer with weight decay
    # CrossEntropyLoss is commented out; using Binary Cross-Entropy Loss instead
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for binary classification

    # ==============================
    # Training Loop
    # ==============================

    N_EPOCHS = 50  # Number of training epochs
    CLIP = 1        # Gradient clipping value

    best_valid_loss = float('inf')  # Initialize the best validation loss
    # Lists to store training and validation metrics for each epoch
    train_losses = []
    val_losses = []
    val_accuracies = []
    val_precisions = []
    val_recalls = []
    val_f1s = []

    # Initialize EarlyStopping with patience of 10 epochs and minimum delta of 0.001
    early_stopping = EarlyStopping(patience=10, min_delta=0.001)

    # Iterate over epochs
    for epoch in range(N_EPOCHS):
        # Train the model for one epoch and get the training loss
        train_loss = train(model, train_loader, optimizer, criterion)
        # Evaluate the model on the validation set and get validation metrics
        valid_loss, valid_acc, valid_precision, valid_recall, valid_f1 = evaluate(model, val_loader, criterion)
        
        # Append current epoch's metrics to the lists
        train_losses.append(train_loss)
        val_losses.append(valid_loss)
        val_accuracies.append(valid_acc)
        val_precisions.append(valid_precision)
        val_recalls.append(valid_recall)
        val_f1s.append(valid_f1)
        
        # Check if the current validation loss is the best so far
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            # Save the model state if validation loss has improved
            torch.save(model.state_dict(), 'best-model.pt')
        
        # Print epoch statistics
        print(f'Epoch: {epoch+1:02}')
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}% |  Precision: {valid_precision:.2f} |  Recall: {valid_recall:.2f} |  F1 Score: {valid_f1:.2f}')

        # Check for early stopping
        early_stopping(valid_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # ==============================
    # Load Best Model and Test
    # ==============================

    # Load the best model saved during training
    model.load_state_dict(torch.load('best-model.pt'))
    # Test the model on the test set and print detailed results
    test_model(model, test_loader)

    # ==============================
    # Plotting Metrics
    # ==============================

    # Plot and save the training and validation metrics
    plot_metrics(train_losses, val_losses, val_accuracies, val_precisions, val_recalls, val_f1s)
