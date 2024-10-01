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
import optuna  # Import Optuna for hyperparameter optimization

# Set the random seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
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

# Custom padding value for labels
PADDING_VALUE = -100  # Using -100 as it's the default ignore_index in CrossEntropyLoss

class EarlyStopping:
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

class ProsodyDataset(Dataset):
    def __init__(self, data):
        self.entries = list(data.items())

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        key, item = self.entries[idx]
        processed_words = preprocess_text(item['words'])
        features = torch.tensor(item['features'], dtype=torch.float32)
        labels = torch.tensor(item['labels'], dtype=torch.long)  # Changed to torch.long for class indices
        
        return processed_words, features, labels

# Collate function with custom padding value
def collate_fn(batch):
    words = [item[0] for item in batch]  # List of lists of words
    features = [item[1] for item in batch]
    labels = [item[2] for item in batch]

    features_padded = torch.nn.utils.rnn.pad_sequence(features, batch_first=True, padding_value=0.0)
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=PADDING_VALUE)

    lengths = torch.tensor([len(f) for f in features])

    return words, features_padded, labels_padded, lengths

# Attention layer with masking
class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.attn = nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs, mask):
        src_len = encoder_outputs.size(1)
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
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

    def forward(self, encoder_outputs, hidden, cell, mask):
        attn_weights = self.attention(hidden[-1], encoder_outputs, mask)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)

        lstm_input = torch.cat(
            (context.unsqueeze(1).repeat(1, encoder_outputs.size(1), 1), encoder_outputs), dim=2
        )
        outputs, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        predictions = self.fc(outputs)

        return predictions, (hidden, cell)

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, features, lengths):
        encoder_outputs, hidden, cell = self.encoder(features, lengths)
        max_len = encoder_outputs.size(1)
        batch_size = encoder_outputs.size(0)
        mask = torch.arange(max_len).unsqueeze(0).expand(batch_size, max_len).to(features.device)
        mask = mask < lengths.unsqueeze(1)
        outputs, (hidden, cell) = self.decoder(encoder_outputs, hidden, cell, mask)
        return outputs

def train(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    for words, features, labels, lengths in iterator:
        features = features.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)

        optimizer.zero_grad()
        output = model(features, lengths)
        # output: [batch_size, seq_len, num_classes]
        # labels: [batch_size, seq_len]

        # Flatten outputs and labels
        output = output.view(-1, num_classes)
        labels = labels.view(-1)

        # Mask padding positions
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
        for words, features, labels, lengths in iterator:
            features = features.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)

            output = model(features, lengths)
            # output: [batch_size, seq_len, num_classes]
            # labels: [batch_size, seq_len]

            # Flatten outputs and labels
            output = output.view(-1, num_classes)
            labels = labels.view(-1)

            # Mask padding positions
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

    with open('./outputs/prosody_bilstm_features_multiclass_results.txt', 'w') as file:
        file.write("")

    with torch.no_grad():
        for words, features, labels, lengths in iterator:
            features = features.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)

            output = model(features, lengths)
            preds = torch.argmax(output, dim=2)

            for i in range(features.shape[0]):
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
                with open('./outputs/prosody_bilstm_features_multiclass_results.txt', 'a') as file:
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
    print(f'Test Precision: {precision*100:.2f}%')
    print(f'Test Recall: {recall*100:.2f}%')
    print(f'Test F1 Score: {f1*100:.2f}%')

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
    plt.savefig('./outputs/bilstm_features_multiclass_metrics.png')

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

# Define the objective function for Optuna
def objective(trial):
    # Suggest hyperparameters
    HIDDEN_DIM = trial.suggest_categorical('HIDDEN_DIM', [128, 256, 512])
    NUM_LAYERS = trial.suggest_int('NUM_LAYERS', 2, 8, step=2)
    DROPOUT = trial.suggest_uniform('DROPOUT', 0.1, 0.5)
    NUM_ATTENTION_LAYERS = trial.suggest_int('NUM_ATTENTION_LAYERS', 2, 8, step=2)
    LR = trial.suggest_loguniform('LR', 1e-5, 1e-2)
    WEIGHT_DECAY = trial.suggest_loguniform('WEIGHT_DECAY', 1e-6, 1e-3)
    BATCH_SIZE = trial.suggest_categorical('BATCH_SIZE', [32, 64, 128])

    # Create data loaders with the suggested batch size
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # Get feature dimension from the dataset
    sample_words, sample_features, sample_labels, sample_lengths = next(iter(train_loader))
    feature_dim = sample_features.shape[2]

    # Determine number of classes dynamically
    all_labels = []

    for _, _, labels in train_dataset:
        all_labels.extend(labels.numpy().flatten())
    global num_classes
    num_classes = len(np.unique(all_labels))

    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # make device global
    

    encoder = Encoder(feature_dim, HIDDEN_DIM, NUM_LAYERS, DROPOUT).to(device)
    decoder = Decoder(HIDDEN_DIM, num_classes, NUM_LAYERS, DROPOUT, NUM_ATTENTION_LAYERS).to(device)
    model = Seq2Seq(encoder, decoder).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=False)

    criterion = nn.CrossEntropyLoss(ignore_index=PADDING_VALUE)

    N_EPOCHS = 50
    global CLIP
    CLIP = 1


    early_stopping = EarlyStopping(patience=10, min_delta=0.001)
    best_valid_f1 = 0.0

    for epoch in range(N_EPOCHS):
        train_loss = train(model, train_loader, optimizer, criterion)
        valid_loss, valid_acc, valid_precision, valid_recall, valid_f1 = evaluate(model, val_loader, criterion)

        # Update the learning rate scheduler
        scheduler.step(valid_loss)

        if valid_f1 > best_valid_f1:
            best_valid_f1 = valid_f1

        early_stopping(valid_loss)
        if early_stopping.early_stop:
            break

    return best_valid_f1  # Return validation F1 score to maximize

if __name__ == "__main__":

    seed = 42
    set_seed(seed)

    json_path = '../prosody/data/multi_label_features.json'
    data = load_data(json_path)

    # Split data with validation set
    train_data, val_data, test_data = split_data(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)

    train_dataset = ProsodyDataset(dict(train_data))
    val_dataset = ProsodyDataset(dict(val_data))
    test_dataset = ProsodyDataset(dict(test_data))

    print(f'Train size: {len(train_dataset)}')
    print(f'Validation size: {len(val_dataset)}')
    print(f'Test size: {len(test_dataset)}')

    # Hyperparameter optimization with Optuna
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)  # Increase n_trials for more extensive search

    print('Best trial:')
    trial = study.best_trial
    print(f'  F1 Score: {trial.value}')
    print('  Params: ')
    for key, value in trial.params.items():
        print(f'    {key}: {value}')

    # Use the best hyperparameters to retrain the model on the combined train and validation sets
    best_params = trial.params

    # Combine train and validation datasets
    combined_data = train_dataset.entries + val_dataset.entries
    combined_dataset = ProsodyDataset(dict(combined_data))

    BATCH_SIZE = best_params['BATCH_SIZE']

    train_loader = DataLoader(combined_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # Get feature dimension from the dataset
    sample_words, sample_features, sample_labels, sample_lengths = next(iter(train_loader))
    feature_dim = sample_features.shape[2]

    # Determine number of classes dynamically
    all_labels = []
    for _, _, labels in train_dataset:
        all_labels.extend(labels.numpy().flatten())
    num_classes = len(np.unique(all_labels))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Unpack best hyperparameters
    HIDDEN_DIM = best_params['HIDDEN_DIM']
    NUM_LAYERS = best_params['NUM_LAYERS']
    DROPOUT = best_params['DROPOUT']
    NUM_ATTENTION_LAYERS = best_params['NUM_ATTENTION_LAYERS']
    LR = best_params['LR']
    WEIGHT_DECAY = best_params['WEIGHT_DECAY']

    encoder = Encoder(feature_dim, HIDDEN_DIM, NUM_LAYERS, DROPOUT).to(device)
    decoder = Decoder(HIDDEN_DIM, num_classes, NUM_LAYERS, DROPOUT, NUM_ATTENTION_LAYERS).to(device)
    model = Seq2Seq(encoder, decoder).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    criterion = nn.CrossEntropyLoss(ignore_index=PADDING_VALUE)

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

    for epoch in range(N_EPOCHS):
        train_loss = train(model, train_loader, optimizer, criterion)
        valid_loss, valid_acc, valid_precision, valid_recall, valid_f1 = evaluate(model, test_loader, criterion)

        train_losses.append(train_loss)
        val_losses.append(valid_loss)
        val_accuracies.append(valid_acc)
        val_precisions.append(valid_precision)
        val_recalls.append(valid_recall)
        val_f1s.append(valid_f1)

        # Update the learning rate scheduler
        scheduler.step(valid_loss)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'models/best-model-features-multiclass-version.pt')

        print(f'Epoch: {epoch+1:02}')
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}% |  Precision: {valid_precision:.2f} |  Recall: {valid_recall:.2f} |  F1 Score: {valid_f1:.2f}')

        early_stopping(valid_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    model.load_state_dict(torch.load('models/best-model-features-multiclass-version.pt'))
    test_model(model, test_loader)
    plot_metrics(train_losses, val_losses, val_accuracies, val_precisions, val_recalls, val_f1s)
