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
    Early stopping to stop the training when the validation loss does not improve after
    a certain number of epochs (patience).
    """
    def __init__(self, patience=5, min_delta=0):
        """
        Args:
            patience (int): How many epochs to wait after last time validation loss improved.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        """
        Call method to check if early stopping condition is met.

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

def load_data(json_path):
    """
    Load data from a JSON file.

    Args:
        json_path (str): Path to the JSON file.

    Returns:
        dict: Data loaded from the JSON file.
    """
    with open(json_path, 'r') as file:
        data = json.load(file)
    return data

def split_data(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Split the data into training, validation, and test sets.

    Args:
        data (dict): The dataset to split.
        train_ratio (float): Proportion of the data to include in the train set.
        val_ratio (float): Proportion of the data to include in the validation set.
        test_ratio (float): Proportion of the data to include in the test set.

    Returns:
        tuple: Randomly split data into train, validation, and test sets.
    """
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1"
    total = len(data)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    test_size = total - train_size - val_size
    return random_split(list(data.items()), [train_size, val_size, test_size])

def get_corpus(data):
    """
    Create a corpus for preprocessing.

    Args:
        data (dict): The dataset from which to create the corpus.

    Returns:
        list: List of tokenized words from the dataset.
    """
    corpus = []
    for entry in data.values():
        words = []
        for word in entry['words']:
            words.extend(re.findall(r"[\w']+|[.,!?;]", word))
        corpus.append(words)
    return corpus

def preprocess_text(words):
    """
    Preprocess text data by tokenizing the words.

    Args:
        words (list): List of words to preprocess.

    Returns:
        list: List of tokenized words.
    """
    processed_words = []
    for word in words:
        processed_words.extend(re.findall(r"[\w']+|[.,!?;]", word))
    return processed_words

def load_glove_embeddings(file_path, embedding_dim):
    """
    Load GloVe embeddings from a file.

    Args:
        file_path (str): Path to the GloVe embeddings file.
        embedding_dim (int): Dimension of the embeddings.

    Returns:
        dict: Dictionary mapping words to their GloVe embeddings.
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
    Create an embedding matrix using the GloVe embeddings.

    Args:
        word2idx (dict): Dictionary mapping words to their indices.
        embeddings_index (dict): Dictionary mapping words to their GloVe embeddings.
        embedding_dim (int): Dimension of the embeddings.

    Returns:
        np.array: Embedding matrix.
    """
    vocab_size = len(word2idx) + 1  # +1 for the unknown token
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, idx in word2idx.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[idx] = embedding_vector
        else:
            embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embedding_dim,))
    return embedding_matrix

class ProsodyDataset(Dataset):
    """
    Custom Dataset class for prosody data.
    """
    def __init__(self, data, word2idx, unk_idx):
        """
        Args:
            data (dict): The dataset.
            word2idx (dict): Dictionary mapping words to indices.
            unk_idx (int): Index for unknown words.
        """
        self.data = data
        self.word2idx = word2idx
        self.unk_idx = unk_idx
        self.entries = list(data.items())

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        """
        Get an item from the dataset.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            tuple: Processed words, features, and labels.
        """
        key, item = self.entries[idx]
        processed_words = preprocess_text(item['words'])
        words = [self.word2idx.get(word, self.unk_idx) for word in processed_words]
        features = torch.tensor(item['features'], dtype=torch.float32)
        labels = torch.tensor(item['labels'], dtype=torch.float32)
        return torch.tensor(words, dtype=torch.long), features, labels

def collate_fn(batch):
    """
    Collate function to pad sequences and ensure consistent lengths.

    Args:
        batch (list): List of tuples (words, features, labels).

    Returns:
        tuple: Padded words, features, and labels.
    """
    words = [item[0] for item in batch]
    features = [item[1] for item in batch]
    labels = [item[2] for item in batch]

    # Pad sequences
    words_padded = torch.nn.utils.rnn.pad_sequence(words, batch_first=True, padding_value=0)
    features_padded = torch.nn.utils.rnn.pad_sequence(features, batch_first=True, padding_value=0.0)
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0.0)

    # Ensure features and labels are padded to the same length as words
    max_length = words_padded.size(1)
    features_padded = torch.nn.functional.pad(features_padded, (0, 0, 0, max_length - features_padded.size(1)), "constant", 0)
    labels_padded = torch.nn.functional.pad(labels_padded, (0, max_length - labels_padded.size(1)), "constant", 0)

    return words_padded, features_padded, labels_padded

class Attention(nn.Module):
    """
    Attention mechanism with stopword masking.
    """
    def __init__(self, hidden_dim):
        """
        Args:
            hidden_dim (int): Dimension of the hidden state.
        """
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        """
        Forward pass for the attention mechanism.

        Args:
            hidden (torch.Tensor): Hidden state from the decoder.
            encoder_outputs (torch.Tensor): Outputs from the encoder.
            words (torch.Tensor): Input word indices.

        Returns:
            torch.Tensor: Attention weights.
        """
        batch_size = encoder_outputs.size(0)
        src_len = encoder_outputs.size(1)
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)

        return torch.softmax(attention, dim=1)

class Encoder(nn.Module):
    """
    Encoder for the Seq2Seq model.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout, embeddings, feature_dim):
        """
        Args:
            vocab_size (int): Size of the vocabulary.
            embedding_dim (int): Dimension of the embeddings.
            hidden_dim (int): Dimension of the hidden state.
            num_layers (int): Number of LSTM layers.
            dropout (float): Dropout rate.
            embeddings (torch.Tensor): Pre-trained embeddings.
            feature_dim (int): Dimension of the features.
        """
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = nn.Parameter(embeddings)
        self.embedding.weight.requires_grad = False  # Freeze embeddings if necessary
        self.lstm = nn.LSTM(embedding_dim + feature_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True, bidirectional=True)

    def forward(self, words, features):
        """
        Forward pass for the encoder.

        Args:
            words (torch.Tensor): Input word indices.
            features (torch.Tensor): Input features.

        Returns:
            tuple: Encoder outputs, hidden state, and cell state.
        """
        embedded = self.embedding(words)
        combined = torch.cat((embedded, features), dim=2)
        outputs, (hidden, cell) = self.lstm(combined)
        return outputs, hidden, cell

class Decoder(nn.Module):
    """
    Decoder with attention for the Seq2Seq model.
    """
    def __init__(self, hidden_dim, output_dim, num_layers, dropout):
        """
        Args:
            hidden_dim (int): Dimension of the hidden state.
            output_dim (int): Dimension of the output.
            num_layers (int): Number of LSTM layers.
            dropout (float): Dropout rate.
        """
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(hidden_dim * 4, hidden_dim, num_layers, dropout=dropout, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.attention = Attention(hidden_dim)

    def forward(self, encoder_outputs, hidden, cell):
        """
        Forward pass for the decoder.

        Args:
            encoder_outputs (torch.Tensor): Outputs from the encoder.
            hidden (torch.Tensor): Hidden state from the encoder.
            cell (torch.Tensor): Cell state from the encoder.
            words (torch.Tensor): Input word indices.

        Returns:
            tuple: Decoder predictions, hidden state, and cell state.
        """
        attn_weights = self.attention(hidden[-1], encoder_outputs)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        lstm_input = torch.cat((context.unsqueeze(1).repeat(1, encoder_outputs.size(1), 1), encoder_outputs), dim=2)
        outputs, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        predictions = self.sigmoid(self.fc(outputs))
        return predictions, (hidden, cell)

class Seq2Seq(nn.Module):
    """
    Sequence-to-Sequence model with attention.
    """
    def __init__(self, encoder, decoder):
        """
        Args:
            encoder (nn.Module): Encoder module.
            decoder (nn.Module): Decoder module.
        """
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, features):
        """
        Forward pass for the Seq2Seq model.

        Args:
            src (torch.Tensor): Input source word indices.
            features (torch.Tensor): Input features.
            words (torch.Tensor): Input word indices.

        Returns:
            torch.Tensor: Model outputs.
        """
        encoder_outputs, hidden, cell = self.encoder(src, features)
        outputs, (hidden, cell) = self.decoder(encoder_outputs, hidden, cell)
        return outputs

def train(model, iterator, optimizer, criterion):
    """
    Training function for the model.

    Args:
        model (nn.Module): Model to train.
        iterator (DataLoader): DataLoader for the training data.
        optimizer (torch.optim.Optimizer): Optimizer.
        criterion (nn.Module): Loss function.

    Returns:
        float: Average training loss.
    """
    model.train()
    epoch_loss = 0
    for words, features, labels in iterator:
        words = words.to(device)
        features = features.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        output = model(words, features)
        output = output.view(-1)
        labels = labels.view(-1).float()
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    """
    Evaluation function for the model.

    Args:
        model (nn.Module): Model to evaluate.
        iterator (DataLoader): DataLoader for the evaluation data.
        criterion (nn.Module): Loss function.

    Returns:
        tuple: Average evaluation loss, accuracy, precision, recall, and F1 score.
    """
    model.eval()
    epoch_loss = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for words, features, labels in iterator:
            words = words.to(device)
            features = features.to(device)
            labels = labels.to(device)
            output = model(words, features)
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

def test_model(model, iterator, word2idx):
    """
    Testing function for the model.

    Args:
        model (nn.Module): Model to test.
        iterator (DataLoader): DataLoader for the test data.
        word2idx (dict): Dictionary mapping words to indices.

    Returns:
        tuple: Lists of true labels and predicted labels.
    """
    model.eval()
    all_labels = []
    all_preds = []
    word_list = list(word2idx.keys())
    with open('prosody_bilstm_embeddings_results.txt', 'w') as file:
        file.write("")
    with torch.no_grad():
        for words, features, labels in iterator:
            words = words.to(device)
            features = features.to(device)
            labels = labels.to(device)
            output = model(words, features)
            preds = (output > 0.4).float()

            for i in range(words.shape[0]):
                word_indices = words[i].cpu().numpy()
                word_sentence = [word_list[idx] if idx < len(word_list) else '<UNK>' for idx in word_indices]
                gold_labels = labels[i].cpu().numpy().flatten()
                pred_labels = preds[i].cpu().numpy().flatten()

                # Clean up the sentence by removing unnecessary padding
                cleaned_words, cleaned_gold_labels, cleaned_pred_labels = clean_up_sentence(word_sentence, gold_labels, pred_labels)

                data = {
                    'Word': cleaned_words,
                    'Gold Label': cleaned_gold_labels.tolist(),
                    'Predicted Label': cleaned_pred_labels.tolist()
                }

                df = pd.DataFrame(data)
                # print(df.to_string(index=False))
                # ''''print("\n" + "-" * 50 + "\n")
                with open('prosody_bilstm_embeddings_results.txt', 'a') as file:
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
    Plot training and validation metrics.

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
    plt.savefig('../prosody/bilstm_embeddings_metrics.png')

def clean_up_sentence(words, gold_labels, pred_labels):
    """
    Clean up the sentence by removing unnecessary padding.

    Args:
        words (list): List of words.
        gold_labels (list): List of gold labels.
        pred_labels (list): List of predicted labels.

    Returns:
        tuple: Cleaned words, gold labels, and predicted labels.
    """
    end_index = len(words) - 1
    while end_index >= 0 and words[end_index] == 'the':
        end_index -= 1

    filtered_words = words[:end_index+1]
    filtered_gold_labels = gold_labels[:end_index+1]
    filtered_pred_labels = pred_labels[:end_index+1]

    for i in range(end_index+1, len(words)):
        if words[i] != 'the':
            filtered_words.append(words[i])
            filtered_gold_labels.append(gold_labels[i])
            filtered_pred_labels.append(pred_labels[i])

    return filtered_words, filtered_gold_labels, filtered_pred_labels

# Main script
if __name__ == "__main__":
    # Load and prepare data
    json_path = '../prosody/reconstructed_extracted_features.json'
    data = load_data(json_path)

    # Split data
    train_data, val_data, test_data = split_data(data)

    # Combine all data for Corpus
    combined_corpus = get_corpus(dict(train_data)) + get_corpus(dict(val_data)) + get_corpus(dict(test_data))

    # Load GloVe embeddings
    embedding_dim = 100  # or any other dimension you prefer
    glove_path = '../prosody/glove_embeddings/glove.6B.100d.txt'  # update with the path to your GloVe file
    glove_embeddings = load_glove_embeddings(glove_path, embedding_dim)

    # Create word2idx and embedding matrix using GloVe
    vocab = list(glove_embeddings.keys())
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    embedding_matrix = create_embedding_matrix(word2idx, glove_embeddings, embedding_dim)

    # Add a zero vector for unknown words
    embedding_matrix = np.vstack((embedding_matrix, np.zeros((1, embedding_dim))))
    unk_idx = len(vocab)  # Index for unknown words

    # Create datasets and dataloaders
    train_dataset = ProsodyDataset(dict(train_data), word2idx, unk_idx)
    val_dataset = ProsodyDataset(dict(val_data), word2idx, unk_idx)
    test_dataset = ProsodyDataset(dict(test_data), word2idx, unk_idx)

    print(f'Train size: {len(train_dataset)}')
    print(f'Validation size: {len(val_dataset)}')
    print(f'Test size: {len(test_dataset)}')

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # Model parameters
    VOCAB_SIZE = len(word2idx) + 1  # +1 for the unknown token
    EMBEDDING_DIM = 100  # dimension of the GloVe embeddings
    HIDDEN_DIM = 128
    OUTPUT_DIM = 1
    NUM_LAYERS = 2
    DROPOUT = 0.5

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Retrieve feature dimension from the first item in the train dataset
    feature_dim = next(iter(train_loader))[1].shape[2]

    embedding_tensor = torch.tensor(embedding_matrix, dtype=torch.float32)

    encoder = Encoder(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT, embedding_tensor, feature_dim).to(device)
    decoder = Decoder(HIDDEN_DIM, OUTPUT_DIM, NUM_LAYERS, DROPOUT).to(device)
    model = Seq2Seq(encoder, decoder).to(device)

    # Print the model summary
    # Prepare a sample input for the summary
    sample_words, sample_features, _ = next(iter(train_loader))
    sample_words = sample_words.to(device)
    sample_features = sample_features.to(device)

    summary(model, input_data=[sample_words, sample_features], device=device)
    
    optimizer = optim.Adam(model.parameters(), weight_decay=1e-5)
    criterion = nn.BCELoss()

    # Store metrics
    train_losses = []
    val_losses = []
    val_accuracies = []
    val_precisions = []
    val_recalls = []
    val_f1s = []

    # Training loop
    N_EPOCHS = 100
    CLIP = 1

    early_stopping = EarlyStopping(patience=10, min_delta=0.001)
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
            torch.save(model.state_dict(), 'models/best-model-embeddings-version.pt')

        print(f'Epoch: {epoch+1:02}')
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}% |  Precision: {valid_precision:.2f} |  Recall: {valid_recall:.2f} |  F1 Score: {valid_f1:.2f}')

        early_stopping(valid_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # Load the best model and test
    model.load_state_dict(torch.load('models/best-model-embeddings-version.pt'))
    test_model(model, test_loader, word2idx)

    # Plotting the metrics
    plot_metrics(train_losses, val_losses, val_accuracies, val_precisions, val_recalls, val_f1s)
