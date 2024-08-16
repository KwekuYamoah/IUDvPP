import json
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import re

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

def collate_fn(batch):
    features = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    features_padded = torch.nn.utils.rnn.pad_sequence(features, batch_first=True, padding_value=0.0)
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0.0)

    max_length = features_padded.size(1)
    labels_padded = torch.nn.functional.pad(labels_padded, (0, max_length - labels_padded.size(1)), "constant", 0)

    return features_padded, labels_padded

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

class TransformerDecoder(nn.Module):
    def __init__(self, hidden_dim, output_dim, num_layers, dropout, num_classes=2, num_heads=8):
        super(TransformerDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.input_fc = nn.Linear(hidden_dim, hidden_dim)  # Ensure the input has the correct dimensionality
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(hidden_dim, output_dim * num_classes if num_classes > 2 else output_dim)
        self.activation = nn.Sigmoid() if num_classes == 2 else None

    def forward(self, memory, tgt):
        tgt = self.input_fc(tgt)  # Ensure target is projected to the correct hidden dimension
        output = self.transformer_decoder(tgt, memory)
        output = self.fc_out(output)

        if self.num_classes == 2:
            output = self.activation(output)

        return output

class TransformerSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(TransformerSeq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, features):
        print("Input features shape:", features.shape)
        memory = self.encoder(features)
        print("Memory shape after encoder:", memory.shape)
        outputs = self.decoder(memory, features)
        print("Outputs shape after decoder:", outputs.shape)
        return outputs

def train(model, iterator, optimizer, criterion, num_classes=2):
    model.train()
    epoch_loss = 0
    for features, labels in iterator:
        features = features.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        output = model(features)
        
        if num_classes == 2:
            output = output.view(-1)
            labels = labels.view(-1).float()
        else:
            output = output.view(-1, num_classes)
            labels = labels.view(-1).long()
        
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion, num_classes=2):
    model.eval()
    epoch_loss = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for features, labels in iterator:
            features = features.to(device)
            labels = labels.to(device)
            output = model(features)
            
            if num_classes == 2:
                output = output.view(-1)
                labels = labels.view(-1).float()
                preds = (output > 0.5).float()
            else:
                output = output.view(-1, num_classes)
                labels = labels.view(-1).long()
                preds = torch.argmax(output, dim=1)
            
            loss = criterion(output, labels)
            epoch_loss += loss.item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    return epoch_loss / len(iterator), accuracy, precision, recall, f1

def test_model(model, iterator, num_classes=2):
    model.eval()
    all_labels = []
    all_preds = []
    with open('./outputs/prosody_transformer_features_results.txt', 'w') as file:
        file.write("")
    with torch.no_grad():
        for features, labels in iterator:
            features = features.to(device)
            labels = labels.to(device)
            output = model(features)
            if num_classes == 2:
                preds = (output > 0.4).float()
            else:
                preds = torch.argmax(output, dim=2)
            for i in range(features.shape[0]):
                gold_labels = labels[i].cpu().numpy().flatten()
                pred_labels = preds[i].cpu().numpy().flatten()

                data = {
                    'Gold Label': gold_labels.tolist(),
                    'Predicted Label': pred_labels.tolist()
                }

                df = pd.DataFrame(data)
                with open('./outputs/prosody_transformer_features_results.txt', 'a') as file:
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

if __name__ == "__main__":
    
    seed = 42 
    set_seed(seed)
    json_path = '../prosody/multi_label_features.json' # Change data path here
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
    NUM_LAYERS = 4
    DROPOUT = 0.3
    NUM_HEADS = 8
    NUM_CLASSES = 4  # Change this depending on the number of classes (set to 2 for binary classification)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    feature_dim = next(iter(train_loader))[0].shape[2]

    encoder = TransformerEncoder(feature_dim, HIDDEN_DIM, NUM_LAYERS, DROPOUT, num_heads=NUM_HEADS).to(device)
    decoder = TransformerDecoder(HIDDEN_DIM, OUTPUT_DIM, NUM_LAYERS, DROPOUT, num_classes=NUM_CLASSES, num_heads=NUM_HEADS).to(device)
    model = TransformerSeq2Seq(encoder, decoder).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss() if NUM_CLASSES > 2 else nn.BCELoss()

    train_losses = []
    val_losses = []
    val_accuracies = []
    val_precisions = []
    val_recalls = []
    val_f1s = []

    N_EPOCHS = 100
    CLIP = 1

    early_stopping = EarlyStopping(patience=15, min_delta=0.001)
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
            torch.save(model.state_dict(), 'models/best-transformer-model.pt')

        print(f'Epoch: {epoch+1:02}')
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}% |  Precision: {valid_precision:.2f} |  Recall: {valid_recall:.2f} |  F1 Score: {valid_f1:.2f}')

        early_stopping(valid_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    model.load_state_dict(torch.load('models/best-transformer-model.pt'))
    test_model(model, test_loader, num_classes=NUM_CLASSES)
