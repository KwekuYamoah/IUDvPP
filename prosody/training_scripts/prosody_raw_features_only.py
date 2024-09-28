import json
import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import pandas as pd
import re
import torch.nn.utils.rnn as rnn_utils

# Set the random seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_corpus(data):
    corpus = []
    for entry in data.values():
        words = []
        for word in entry['words']:
            words.extend(re.findall(r"[\w']+|[.,!?;]", word))
        corpus.append(words)
    return corpus

def load_data(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
    return data

def split_data(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1"
    total = len(data)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    test_size = total - train_size - val_size
    return random_split(list(data.items()), [train_size, val_size, test_size])

def preprocess_text(words):
    processed_words = []
    for word in words:
        processed_words.extend(re.findall(r"[\w']+|[.,!?;]", word))
    return processed_words

# Custom padding value (since your labels are 0 or 1, we'll use -1 for padding)
PADDING_VALUE = -1

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
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

class ProsodyDataset(Dataset):
    def __init__(self, data):
        self.entries = list(data.items())

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        key, item = self.entries[idx]
        processed_words = preprocess_text(item['words'])
        # Extract both prosodic and raw acoustic features
        prosodic_features = torch.tensor(item['prosodic_features'], dtype=torch.float32)
        raw_acoustic_features = torch.tensor(item['raw_acoustic_features'], dtype=torch.float32)
        labels = torch.tensor(item['labels'], dtype=torch.long)
        return processed_words, prosodic_features, raw_acoustic_features, labels

# Collate function with custom padding value (-1)
def collate_fn(batch):
    words = [item[0] for item in batch]  # List of lists of words
    prosodic_features = [item[1] for item in batch]
    raw_acoustic_features = [item[2] for item in batch]
    labels = [item[3] for item in batch]

    prosodic_features_padded = torch.nn.utils.rnn.pad_sequence(
        prosodic_features, batch_first=True, padding_value=0.0
    )
    raw_acoustic_features_padded = torch.nn.utils.rnn.pad_sequence(
        raw_acoustic_features, batch_first=True, padding_value=0.0
    )
    labels_padded = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=PADDING_VALUE
    )

    lengths = torch.tensor([len(f) for f in prosodic_features])  # Assuming both have same lengths

    return words, prosodic_features_padded, raw_acoustic_features_padded, labels_padded, lengths

# Feature projection layer to reduce dimensions
class FeatureProjection(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FeatureProjection, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.linear(x))

# Attention layer with masking
class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        # Updated input dimension from hidden_dim * 4 to hidden_dim * 4
        self.attn = nn.Linear(hidden_dim * 4, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs, mask):
        src_len = encoder_outputs.size(1)

        # Concatenate the last forward and backward hidden states
        hidden_forward = hidden[-2]  # Last layer forward hidden state
        hidden_backward = hidden[-1]  # Last layer backward hidden state
        hidden = torch.cat((hidden_forward, hidden_backward), dim=1)  # Shape: [batch_size, hidden_dim * 2]

        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)  # Shape: [batch_size, src_len, hidden_dim * 2]

        # Concatenate hidden and encoder outputs
        concatenated = torch.cat((hidden, encoder_outputs), dim=2)  # Shape: [batch_size, src_len, hidden_dim * 4]

        # Apply attention
        energy = torch.tanh(self.attn(concatenated))
        attention = self.v(energy).squeeze(2)

        # Apply the mask to ignore padding positions
        attention.masked_fill_(~mask, -1e10)

        return torch.softmax(attention, dim=1)

class MultiAttention(nn.Module):
    def __init__(self, hidden_dim, num_layers):
        super(MultiAttention, self).__init__()
        self.attention_layers = nn.ModuleList([AttentionLayer(hidden_dim) for _ in range(num_layers)])

    def forward(self, hidden, encoder_outputs, mask):
        attn_weights = []
        for layer in self.attention_layers:
            attn_weight = layer(hidden, encoder_outputs, mask)
            attn_weights.append(attn_weight)
        attn_weights = torch.stack(attn_weights, dim=1).mean(dim=1)
        return attn_weights

# Encoder with packing
class Encoder(nn.Module):
    def __init__(self, feature_dim, hidden_dim, num_layers, dropout):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(feature_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True, bidirectional=True)

    def forward(self, features, lengths):
        lengths_sorted, sorted_indices = lengths.sort(descending=True)
        features = features[sorted_indices]

        packed_input = rnn_utils.pack_padded_sequence(features, lengths_sorted.cpu(), batch_first=True, enforce_sorted=True)

        packed_output, (hidden, cell) = self.lstm(packed_input)

        outputs, _ = rnn_utils.pad_packed_sequence(packed_output, batch_first=True, padding_value=0.0)

        _, original_indices = sorted_indices.sort()
        outputs = outputs[original_indices]
        hidden = hidden[:, original_indices]
        cell = cell[:, original_indices]

        return outputs, hidden, cell

# Decoder with attention and masking
class Decoder(nn.Module):
    def __init__(self, hidden_dim, output_dim, num_layers, dropout, num_attention_layers):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(hidden_dim * 4, hidden_dim, num_layers, dropout=dropout, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.attention = MultiAttention(hidden_dim, num_attention_layers)

    def forward(self, encoder_outputs, hidden, cell, lengths):
        max_len = encoder_outputs.size(1)
        batch_size = encoder_outputs.size(0)
        # Create mask based on lengths
        mask = torch.arange(max_len).unsqueeze(0).expand(batch_size, max_len).to(encoder_outputs.device)
        mask = mask < lengths.unsqueeze(1)

        attn_weights = self.attention(hidden, encoder_outputs, mask)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)

        # lstm_input size: [batch_size, seq_len, hidden_dim * 4]
        lstm_input = torch.cat(
            (context.unsqueeze(1).repeat(1, encoder_outputs.size(1), 1), encoder_outputs), dim=2
        )
        outputs, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        predictions = self.fc(outputs)

        return predictions, (hidden, cell)

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, prosodic_projection, acoustic_projection):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.prosodic_projection = prosodic_projection
        self.acoustic_projection = acoustic_projection

    def forward(self, prosodic_features, raw_acoustic_features, lengths):
        # Project features
        projected_prosodic = self.prosodic_projection(prosodic_features)
        projected_acoustic = self.acoustic_projection(raw_acoustic_features)
        # Concatenate along the feature dimension
        features = torch.cat((projected_prosodic, projected_acoustic), dim=2)
        # Pass through the encoder
        encoder_outputs, hidden, cell = self.encoder(features, lengths)
        outputs, (hidden, cell) = self.decoder(encoder_outputs, hidden, cell, lengths)
        return outputs

def train(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    for words, prosodic_features, raw_acoustic_features, labels, lengths in iterator:
        prosodic_features = prosodic_features.to(device)
        raw_acoustic_features = raw_acoustic_features.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)

        optimizer.zero_grad()
        output = model(prosodic_features, raw_acoustic_features, lengths)

        # Flatten outputs and labels
        output = output.view(-1, num_classes)
        labels = labels.view(-1)

        mask = labels != PADDING_VALUE
        masked_output = output[mask]
        masked_labels = labels[mask]

        loss = criterion(masked_output, masked_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for words, prosodic_features, raw_acoustic_features, labels, lengths in iterator:
            prosodic_features = prosodic_features.to(device)
            raw_acoustic_features = raw_acoustic_features.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)

            output = model(prosodic_features, raw_acoustic_features, lengths)
            output = output.view(-1, num_classes)
            labels = labels.view(-1)

            mask = labels != PADDING_VALUE
            masked_output = output[mask]
            masked_labels = labels[mask]

            loss = criterion(masked_output, masked_labels)
            epoch_loss += loss.item()

            preds = torch.argmax(masked_output, dim=1)
            all_labels.extend(masked_labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    return epoch_loss / len(iterator), accuracy, precision, recall, f1

def test_model(model, iterator):
    model.eval()
    all_labels = []
    all_preds = []

    with open('./outputs/prosody_raw_acoustic_results.txt', 'w') as file:
        file.write("")

    with torch.no_grad():
        for words, prosodic_features, raw_acoustic_features, labels, lengths in iterator:
            prosodic_features = prosodic_features.to(device)
            raw_acoustic_features = raw_acoustic_features.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)

            output = model(prosodic_features, raw_acoustic_features, lengths)
            preds = torch.argmax(output, dim=2)

            for i in range(prosodic_features.shape[0]):
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
                with open('./outputs/prosody_raw_acoustic_results.txt', 'a') as file:
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
    print(f'Test Precision: {precision*100:.2f}')
    print(f'Test Recall: {recall*100:.2f}')
    print(f'Test F1 Score: {f1*100:.2f}')

    return all_labels, all_preds

def plot_metrics(train_losses, val_losses, val_accuracies, val_precisions, val_recalls, val_f1s):
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
    plt.savefig('./outputs/prosody_raw_acoustic_metrics.png')

def clean_up_sentence(words, gold_labels, pred_labels):
    filtered_words = []
    filtered_gold_labels = []
    filtered_pred_labels = []

    for i in range(len(words)):
        if i < len(gold_labels) and gold_labels[i] != PADDING_VALUE:
            filtered_words.append(words[i])
            filtered_gold_labels.append(int(gold_labels[i]))
            filtered_pred_labels.append(int(pred_labels[i]))

    return filtered_words, filtered_gold_labels, filtered_pred_labels

if __name__ == "__main__":
    seed = 42
    set_seed(seed)

    json_path = '../prosody/data/prosodic_raw_acoustic_features.json' 
    data = load_data(json_path)

    train_data, val_data, test_data = split_data(data)

    train_dataset = ProsodyDataset(dict(train_data))
    val_dataset = ProsodyDataset(dict(val_data))
    test_dataset = ProsodyDataset(dict(test_data))

    print(f'Train size: {len(train_dataset)}')
    print(f'Validation size: {len(val_dataset)}')
    print(f'Test size: {len(test_dataset)}')

    # Create the DataLoader
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # Get feature dimensions from the dataset (after padding)
    sample_batch = next(iter(train_loader))
    sample_words, sample_prosodic_features, sample_raw_acoustic_features, sample_labels, sample_lengths = sample_batch
    prosodic_features_dim = sample_prosodic_features.shape[2]
    raw_acoustic_features_dim = sample_raw_acoustic_features.shape[2]
    print(f"Prosodic features dimension: {prosodic_features_dim}")
    print(f"Raw acoustic features dimension: {raw_acoustic_features_dim}")

    # Number of classes (excluding padding)
    all_labels = []
    for _, _, _, labels in train_dataset:
        all_labels.extend(labels.numpy().flatten())

    num_classes = len(np.unique(all_labels))
    print(f'Model Training with {num_classes} classes')

    # Define model parameters
    PROJECTED_DIM = 128  # Dimension after projection
    INPUT_DIM = PROJECTED_DIM * 2  # Because we're concatenating two projected feature sets
    HIDDEN_DIM = 256
    OUTPUT_DIM = num_classes
    NUM_LAYERS = 16
    DROPOUT = 0.2
    NUM_ATTENTION_LAYERS = 4

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Instantiate the projection layers
    prosodic_projection = FeatureProjection(prosodic_features_dim, PROJECTED_DIM).to(device)
    acoustic_projection = FeatureProjection(raw_acoustic_features_dim, PROJECTED_DIM).to(device)

    # Instantiate the encoder and decoder
    encoder = Encoder(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT).to(device)
    decoder = Decoder(HIDDEN_DIM, OUTPUT_DIM, NUM_LAYERS, DROPOUT, NUM_ATTENTION_LAYERS).to(device)
    model = Seq2Seq(encoder, decoder, prosodic_projection, acoustic_projection).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, verbose=True)

    criterion = nn.CrossEntropyLoss(ignore_index=PADDING_VALUE)

    # Initialize metrics lists
    train_losses = []
    val_losses = []
    val_accuracies = []
    val_precisions = []
    val_recalls = []
    val_f1s = []

    N_EPOCHS = 500
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

        # Update the learning rate based on the validation loss
        scheduler.step(valid_loss)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'models/best-model-prosody-raw-acoustic.pt')

        print(f'Epoch: {epoch+1:02}')
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\tVal. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:.2f}%')
        print(f'\tPrecision: {valid_precision:.2f} | Recall: {valid_recall:.2f} | F1 Score: {valid_f1:.2f}')

        early_stopping(valid_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # Load the best model and test
    model.load_state_dict(torch.load('models/best-model-prosody-raw-acoustic.pt'))
    test_model(model, test_loader)
    plot_metrics(train_losses, val_losses, val_accuracies, val_precisions, val_recalls, val_f1s)
