import json
import torch
import numpy as np
from gensim.models import Word2Vec
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import pandas as pd
from torchview import draw_graph

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

# 1. Load JSON Data
def load_data(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
    return data

# 2. Split Data
def split_data(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1"
    total = len(data)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    test_size = total - train_size - val_size
    return random_split(list(data.items()), [train_size, val_size, test_size])

# 3. Train Word2Vec Model
def train_word2vec(corpus, embedding_dim=100):
    word2vec_model = Word2Vec(sentences=corpus, vector_size=embedding_dim, window=5, min_count=1, workers=4)
    word2vec_model.save("word2vec.model")
    return word2vec_model

# 4. Create Corpus for Word2Vec Training
def get_corpus(data):
    corpus = []
    for entry in data.values():
        corpus.append(entry['words'])
    return corpus

# Define the dataset class
class ProsodyDataset(Dataset):
    def __init__(self, data, word2idx, unk_idx):
        self.data = data
        self.word2idx = word2idx
        self.unk_idx = unk_idx
        self.entries = list(data.items())

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        key, item = self.entries[idx]
        words = [self.word2idx.get(word, self.unk_idx) for word in item['words']]
        features = torch.tensor(item['features'], dtype=torch.float32)
        labels = torch.tensor(item['labels'], dtype=torch.float32)
        return torch.tensor(words, dtype=torch.long), features, labels

def collate_fn(batch):
    words = torch.nn.utils.rnn.pad_sequence([item[0] for item in batch], batch_first=True, padding_value=0)
    features = torch.nn.utils.rnn.pad_sequence([item[1] for item in batch], batch_first=True, padding_value=0.0)
    labels = torch.nn.utils.rnn.pad_sequence([item[2] for item in batch], batch_first=True, padding_value=0.0)
    return words, features, labels

# Define the Encoder
class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout, embeddings, feature_dim):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = nn.Parameter(embeddings)
        self.embedding.weight.requires_grad = False  # Freeze embeddings if necessary
        self.lstm = nn.LSTM(embedding_dim + feature_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True, bidirectional=True)

    def forward(self, words, features):
        embedded = self.embedding(words)
        combined = torch.cat((embedded, features), dim=2)
        outputs, (hidden, cell) = self.lstm(combined)
        return outputs, hidden, cell

# Define the Decoder
class Decoder(nn.Module):
    def __init__(self, hidden_dim, output_dim, num_layers, dropout):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(hidden_dim * 2, hidden_dim, num_layers, dropout=dropout, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden, cell):
        outputs, (hidden, cell) = self.lstm(x, (hidden, cell))
        predictions = self.sigmoid(self.fc(outputs))
        return predictions, (hidden, cell)

# Define the Seq2Seq model
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, features):
        encoder_outputs, hidden, cell = self.encoder(src, features)
        outputs, (hidden, cell) = self.decoder(encoder_outputs, hidden, cell)
        return outputs

# Training function
def train(model, iterator, optimizer, criterion):
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

# Evaluation function
def evaluate(model, iterator, criterion):
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
            preds = (output > 0.3).float()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    return epoch_loss / len(iterator), accuracy, precision, recall, f1

# Testing function
def test_model(model, iterator, word2idx):
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
            preds = (output > 0.3).float()

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
                    'Predicted Label':cleaned_pred_labels.tolist()
                }
                

                df = pd.DataFrame(data)
                print(df.to_string(index=False))
                print("\n" + "-" * 50 + "\n")
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

# Plotting function
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
    plt.savefig('../prosody/bilstm_embeddings_metrics.png')


def clean_up_sentence(words, gold_labels, pred_labels):
    punctuation_marks = {'.', '!', '?'}

    # Find the index of the last word that ends with a punctuation mark
    end_index = None
    for i in range(len(words)):
        if any(words[i].endswith(punc) for punc in punctuation_marks):
            end_index = i

    if end_index is not None:
        # Remove all "the" after the sentence-ending word
        filtered_words = words[:end_index+1]
        filtered_gold_labels = gold_labels[:end_index+1]
        filtered_pred_labels = pred_labels[:end_index+1]

        for i in range(end_index+1, len(words)):
            if words[i] != 'the':
                filtered_words.append(words[i])
                filtered_gold_labels.append(gold_labels[i])
                filtered_pred_labels.append(pred_labels[i])

        return filtered_words, filtered_gold_labels, filtered_pred_labels
    else:
        return words, gold_labels, pred_labels


# Main script
if __name__ == "__main__":
    # Load and prepare data
    json_path = '../prosody/reconstructed_extracted_features.json'
    data = load_data(json_path)

    # Split data
    train_data, val_data, test_data = split_data(data)

    # Combine all data for Word2Vec training
    combined_corpus = get_corpus(dict(train_data)) + get_corpus(dict(val_data)) + get_corpus(dict(test_data))
    word2vec_model = train_word2vec(combined_corpus)

    # Create word2idx and embedding matrix
    vocab = word2vec_model.wv.index_to_key
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    embeddings = torch.tensor(word2vec_model.wv.vectors, dtype=torch.float32)
    unk_idx = len(vocab)  # Index for unknown words

    # Add a zero vector for unknown words
    embeddings = torch.cat((embeddings, torch.zeros((1, embeddings.size(1)))), 0)

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
    VOCAB_SIZE = len(vocab) + 1  # +1 for the unknown token
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 128
    OUTPUT_DIM = 1
    NUM_LAYERS = 2
    DROPOUT = 0.5

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Retrieve feature dimension from the first item in the train dataset
    feature_dim = next(iter(train_loader))[1].shape[2]

    encoder = Encoder(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT, embeddings, feature_dim).to(device)
    decoder = Decoder(HIDDEN_DIM, OUTPUT_DIM, NUM_LAYERS, DROPOUT).to(device)
    model = Seq2Seq(encoder, decoder).to(device)

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
            torch.save(model.state_dict(), 'best-model-embeddings-version.pt')

        print(f'Epoch: {epoch+1:02}')
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}% |  Precision: {valid_precision:.2f} |  Recall: {valid_recall:.2f} |  F1 Score: {valid_f1:.2f}')

        early_stopping(valid_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break
   
    # Load the best model and test
    model.load_state_dict(torch.load('best-model-embeddings-version.pt'))
    test_model(model, test_loader, word2idx)

    # Plotting the metrics
    plot_metrics(train_losses, val_losses, val_accuracies, val_precisions, val_recalls, val_f1s)

    
    
