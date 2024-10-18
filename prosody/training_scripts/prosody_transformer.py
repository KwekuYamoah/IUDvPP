import json
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

# Set the random seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

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

# Load data from a JSON file
def load_data(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
    return data

# Split the data into train, validation, and test sets
def split_data(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1"
    total = len(data)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    test_size = total - train_size - val_size
    return random_split(list(data.items()), [train_size, val_size, test_size])

# Custom dataset class for prosody features
class ProsodyDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.entries = list(data.items())

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        key, item = self.entries[idx]
        features = torch.tensor(item['features'], dtype=torch.float32)
        labels = torch.tensor(item['labels'], dtype=torch.long)  # Ensure labels are LongTensor
        return features, labels

# Custom collate function to handle padding
def collate_fn(batch):
    features = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    features_padded = torch.nn.utils.rnn.pad_sequence(features, batch_first=True, padding_value=0.0)
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0)  # Padding index is 0

    return features_padded, labels_padded

# Transformer Encoder
class TransformerEncoder(nn.Module):
    def __init__(self, feature_dim, hidden_dim, num_layers, dropout, num_heads=8):
        super(TransformerEncoder, self).__init__()
        self.input_fc = nn.Linear(feature_dim, hidden_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features):
        src = self.input_fc(features)
        src = self.dropout(src)
        memory = self.transformer_encoder(src)
        return memory

# Transformer Decoder
class TransformerDecoder(nn.Module):
    def __init__(self, hidden_dim, output_dim, num_layers, dropout, num_classes=2, num_heads=8):
        super(TransformerDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.embedding = nn.Embedding(num_classes, hidden_dim)  # Embedding layer for labels
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(hidden_dim, num_classes)

    def forward(self, memory, tgt):
        tgt = self.embedding(tgt)  # Embed the labels
        output = self.transformer_decoder(tgt, memory)
        output = self.fc_out(output)
        return output

# Seq2Seq model combining encoder and decoder
class TransformerSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(TransformerSeq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, features, labels):
        memory = self.encoder(features)
        outputs = self.decoder(memory, labels)
        return outputs

# Training function
def train(model, iterator, optimizer, criterion, num_classes=2):
    model.train()
    epoch_loss = 0
    for features, labels in iterator:
        features = features.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        output = model(features, labels)
        output = output.view(-1, num_classes)
        labels = labels.view(-1)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

# Evaluation function
def evaluate(model, iterator, criterion, num_classes=2):
    model.eval()
    epoch_loss = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for features, labels in iterator:
            features = features.to(device)
            labels = labels.to(device)
            output = model(features, labels)
            output = output.view(-1, num_classes)
            labels = labels.view(-1)
            preds = torch.argmax(output, dim=1)
            loss = criterion(output, labels)
            epoch_loss += loss.item()
            # Exclude padding indices from metrics
            non_pad_indices = labels != 0  # Assuming 0 is the PAD_IDX
            all_labels.extend(labels[non_pad_indices].cpu().numpy())
            all_preds.extend(preds[non_pad_indices].cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    return epoch_loss / len(iterator), accuracy, precision, recall, f1

# Main script
if __name__ == "__main__":
    set_seed(42)
    json_path = '../prosody/data/multi_label_features.json'
    data = load_data(json_path)

    train_data, val_data, test_data = split_data(data)

    train_dataset = ProsodyDataset(dict(train_data))
    val_dataset = ProsodyDataset(dict(val_data))
    test_dataset = ProsodyDataset(dict(test_data))

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    HIDDEN_DIM = 128
    OUTPUT_DIM = 1
    NUM_LAYERS = 4
    DROPOUT = 0.3
    NUM_HEADS = 8

    # Get feature dimension from padded features
    sample_features, sample_labels = next(iter(train_loader))
    feature_dim = sample_features.shape[2]

    # Number of classes (including padding index if necessary)
    all_labels = []
    for _, labels in train_dataset:
        all_labels.extend(labels.numpy().flatten())
    NUM_CLASSES = len(np.unique(all_labels))
    PAD_IDX = 0  # Assuming padding index is 0

    print(f'Model Training with {NUM_CLASSES} classes')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    encoder = TransformerEncoder(feature_dim, HIDDEN_DIM, NUM_LAYERS, DROPOUT, num_heads=NUM_HEADS).to(device)
    decoder = TransformerDecoder(HIDDEN_DIM, OUTPUT_DIM, NUM_LAYERS, DROPOUT, num_classes=NUM_CLASSES, num_heads=NUM_HEADS).to(device)
    model = TransformerSeq2Seq(encoder, decoder).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    N_EPOCHS = 100
    early_stopping = EarlyStopping(patience=15, min_delta=0.001)
    best_valid_loss = float('inf')

    # Create directory for saving the model if it doesn't exist
    os.makedirs('models', exist_ok=True)

    for epoch in range(N_EPOCHS):
        train_loss = train(model, train_loader, optimizer, criterion, num_classes=NUM_CLASSES)
        valid_loss, valid_acc, valid_precision, valid_recall, valid_f1 = evaluate(model, val_loader, criterion, num_classes=NUM_CLASSES)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'models/best-transformer-model.pt')

        print(f'Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Val. Loss: {valid_loss:.3f} | '
              f'Val. Acc: {valid_acc*100:.2f}% | Precision: {valid_precision:.2f} | '
              f'Recall: {valid_recall:.2f} | F1 Score: {valid_f1:.2f}')

        scheduler.step()
        early_stopping(valid_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # Load the best model and evaluate on the test set
    model.load_state_dict(torch.load('models/best-transformer-model.pt'))
    test_loss, test_acc, test_precision, test_recall, test_f1 = evaluate(model, test_loader, criterion, num_classes=NUM_CLASSES)
    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}% | '
          f'Precision: {test_precision:.2f} | Recall: {test_recall:.2f} | F1 Score: {test_f1:.2f}')
