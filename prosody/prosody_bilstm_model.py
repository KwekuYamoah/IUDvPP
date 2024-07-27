import json
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt

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
        self.data = data
        self.sentences = list(data.keys())

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence_key = self.sentences[idx]
        sentence_data = self.data[sentence_key]
        words = sentence_data['words']
        labels = torch.tensor(sentence_data['labels'], dtype=torch.long)
        #positions = torch.tensor(sentence_data['positions'], dtype=torch.long)
        features = torch.tensor(sentence_data['features'], dtype=torch.float)
        return words, features, labels, #positions

def collate_fn(batch):
    words, features, labels  = zip(*batch) #positions
    return words, torch.nn.utils.rnn.pad_sequence(features, batch_first=True), \
           torch.nn.utils.rnn.pad_sequence(labels, batch_first=True), \
           #torch.nn.utils.rnn.pad_sequence(positions, batch_first=True)

def create_data_loaders(data, batch_size=32, val_split=0.2, test_split=0.1):
    dataset = ProsodyDataset(data)
    val_size = int(len(dataset) * val_split)
    test_size = int(len(dataset) * test_split)
    train_size = len(dataset) - val_size - test_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return train_loader, val_loader, test_loader

class Encoder(nn.Module):
    # def __init__(self, input_dim, hidden_dim, num_layers):
    #     super(Encoder, self).__init__()
    #     self.hidden_dim = hidden_dim
    #     self.num_layers = num_layers
    #     self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, bidirectional=True, batch_first=True)
        
    # def forward(self, x):
    #     batch_size = x.size(0)
    #     h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim).to(x.device)
    #     c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim).to(x.device)
    #     outputs, (hidden, cell) = self.lstm(x, (h0, c0))
    #     hidden = hidden.view(self.num_layers, 2, batch_size, self.hidden_dim)
    #     hidden = torch.cat((hidden[:, -2, :, :], hidden[:, -1, :, :]), dim=2)
    #     cell = cell.view(self.num_layers, 2, batch_size, self.hidden_dim)
    #     cell = torch.cat((cell[:, -2, :, :], cell[:, -1, :, :]), dim=2)
    #     return outputs, (hidden, cell)

    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True, bidirectional=True)
    
    def forward(self, x):
        outputs, (hidden, cell) = self.lstm(x)
        return outputs, hidden, cell

class Decoder(nn.Module):
    # def __init__(self, hidden_dim, output_dim, num_layers):
    #     super(Decoder, self).__init__()
    #     self.hidden_dim = hidden_dim
    #     self.num_layers = num_layers
    #     self.lstm = nn.LSTM(hidden_dim * 2, hidden_dim, num_layers, bidirectional=True, batch_first=True)
    #     self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
    # def forward(self, x, hidden, cell):
    #     hidden = hidden.view(self.num_layers * 2, -1, self.hidden_dim)
    #     cell = cell.view(self.num_layers * 2, -1, self.hidden_dim)
    #     outputs, (hidden, cell) = self.lstm(x, (hidden, cell))
    #     predictions = self.fc(outputs)
    #     return predictions, (hidden, cell)

    def __init__(self, hidden_dim, output_dim, num_layers, dropout):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(hidden_dim * 2, hidden_dim, num_layers, dropout=dropout, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # * 2 for bidirectional

    def forward(self, x, hidden, cell):
        outputs, (hidden, cell) = self.lstm(x, (hidden, cell))
        outputs = torch.sigmoid(self.fc(outputs))
        return outputs, (hidden, cell)


class Seq2Seq(nn.Module):
    # def __init__(self, encoder, decoder):
    #     super(Seq2Seq, self).__init__()
    #     self.encoder = encoder
    #     self.decoder = decoder
        
    # def forward(self, src, trg):
    #     encoder_outputs, (hidden, cell) = self.encoder(src)
    #     hidden = hidden.view(self.encoder.num_layers * 2, -1, self.encoder.hidden_dim)
    #     cell = cell.view(self.encoder.num_layers * 2, -1, self.encoder.hidden_dim)
    #     outputs, (hidden, cell) = self.decoder(encoder_outputs, hidden, cell)
    #     return outputs

    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg):
        encoder_outputs, hidden, cell = self.encoder(src)
        outputs, (hidden, cell) = self.decoder(encoder_outputs, hidden, cell)
        return outputs


# def train(model, iterator, optimizer, criterion, clip):
#     model.train()
#     epoch_loss = 0
#     for _, features, labels in iterator:
#         optimizer.zero_grad()
#         output = model(features, features)
#         output = output.view(-1, output.shape[-1])
#         labels = labels.view(-1)
#         loss = criterion(output, labels)
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
       
#         optimizer.step()
#         epoch_loss += loss.item()
#     return epoch_loss / len(iterator)

# def evaluate(model, iterator, criterion):
#     model.eval()
#     epoch_loss = 0
#     all_labels = []
#     all_preds = []
#     with torch.no_grad():
#         for _, features, labels in iterator:
#             output = model(features, features)
#             output = output.view(-1, output.shape[-1])
#             labels = labels.view(-1)
#             loss = criterion(output, labels)
#             epoch_loss += loss.item()
#             preds = torch.argmax(output, dim=1)
#             all_labels.extend(labels.cpu().numpy())
#             all_preds.extend(preds.cpu().numpy())
#     accuracy = accuracy_score(all_labels, all_preds)
#     precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
#     recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
#     f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
#     return epoch_loss / len(iterator), accuracy, precision, recall, f1

# def test_model(model, iterator):
#     model.eval()
#     all_labels = []
#     all_preds = []
#     with torch.no_grad():
#         for words, features, labels in iterator:
#             output = model(features, features)
#             preds = torch.argmax(output, dim=2)
#             all_labels.extend(labels.cpu().numpy().tolist())
#             all_preds.extend(preds.cpu().numpy().tolist())
#             print(f'Sentence: {" ".join(words[0])}')
#             print(f'Gold Labels: {labels[0].cpu().numpy()}')
#             print(f'Predicted Labels: {preds[0].cpu().numpy()}')
#     # Flatten the lists
#     all_labels = [item for sublist in all_labels for item in sublist]
#     all_preds = [item for sublist in all_preds for item in sublist]
#     accuracy = accuracy_score(all_labels, all_preds)
#     precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
#     recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
#     f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
#     print(f'Test Accuracy: {accuracy*100:.2f}%')
#     print(f'Test Precision: {precision:.2f}')
#     print(f'Test Recall: {recall:.2f}')
#     print(f'Test F1 Score: {f1:.2f}')
#     return all_labels, all_preds


def train(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    for _, features, labels in iterator:
        optimizer.zero_grad()
        output = model(features, features)
        output = output.view(-1)
        labels = labels.view(-1).float()
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for _, features, labels in iterator:
            output = model(features, features)
            output = output.view(-1)
            labels = labels.view(-1).float()
            loss = criterion(output, labels)
            epoch_loss += loss.item()
            preds = (output > 0.4).float()
            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    return epoch_loss / len(iterator), accuracy, precision, recall, f1

def test_model(model, iterator):
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for words, features, labels in iterator:
            output = model(features, features)
            preds = (output > 0.4).float()
            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            print(f'Sentence: {" ".join(words[0])}')
            print(f'Gold Labels: {labels[0].cpu().numpy()}')
            print(f'Predicted Labels: {preds[0].cpu().numpy().flatten()}')
    all_labels = [item for sublist in all_labels for item in sublist]
    all_preds = [item for sublist in all_preds for item in sublist]
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    print(f'Test Accuracy: {accuracy*100:.2f}%')
    print(f'Test Precision: {precision:.2f}')
    print(f'Test Recall: {recall:.2f}')
    print(f'Test F1 Score: {f1:.2f}')
    return all_labels, all_preds




# Plotting the metrics
# def plot_metrics(train_losses, val_losses, val_accuracies, val_precisions, val_recalls, val_f1s, N_EPOCHS):
#     epochs = range(1, N_EPOCHS + 1)
    
#     plt.figure(figsize=(14, 10))
    
#     plt.subplot(2, 3, 1)
#     plt.plot(epochs, train_losses, label='Train Loss')
#     plt.plot(epochs, val_losses, label='Validation Loss')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.title('Loss')
#     plt.legend()
    
#     plt.subplot(2, 3, 2)
#     plt.plot(epochs, val_accuracies, label='Validation Accuracy')
#     plt.xlabel('Epochs')
#     plt.ylabel('Accuracy')
#     plt.title('Accuracy')
#     plt.legend()
    
#     plt.subplot(2, 3, 3)
#     plt.plot(epochs, val_precisions, label='Validation Precision')
#     plt.xlabel('Epochs')
#     plt.ylabel('Precision')
#     plt.title('Precision')
#     plt.legend()
    
#     plt.subplot(2, 3, 4)
#     plt.plot(epochs, val_recalls, label='Validation Recall')
#     plt.xlabel('Epochs')
#     plt.ylabel('Recall')
#     plt.title('Recall')
#     plt.legend()
    
#     plt.subplot(2, 3, 5)
#     plt.plot(epochs, val_f1s, label='Validation F1 Score')
#     plt.xlabel('Epochs')
#     plt.ylabel('F1 Score')
#     plt.title('F1 Score')
#     plt.legend()
    
#     plt.tight_layout()
#     plt.show()

def plot_metrics(train_losses, val_losses, val_accuracies, val_precisions, val_recalls, val_f1s):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 3, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.legend()

    plt.subplot(2, 3, 2)
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(2, 3, 3)
    plt.plot(epochs, val_precisions, label='Validation Precision')
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.title('Precision')
    plt.legend()

    plt.subplot(2, 3, 4)
    plt.plot(epochs, val_recalls, label='Validation Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.title('Recall')
    plt.legend()

    plt.subplot(2, 3, 5)
    plt.plot(epochs, val_f1s, label='Validation F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.title('F1 Score')
    plt.legend()

    plt.tight_layout()
    #plt.show()
    plt.savefig('../prosody/bilstm_metrics.png')







if __name__ == '__main__':
    # Load the data
    with open('../prosody/reconstructed_extracted_features.json', 'r') as f:
        data = json.load(f)

    train_loader, val_loader, test_loader = create_data_loaders(data, batch_size=32)
    
    # Model instantiation
    INPUT_DIM = len(data['pm04_in_027']['features'][0])
    HIDDEN_DIM = 256
    OUTPUT_DIM = 1  # Assuming binary labels
    NUM_LAYERS = 4
    DROPOUT = 0.5

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # encoder = Encoder(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS).to(device)
    # decoder = Decoder(HIDDEN_DIM, OUTPUT_DIM, NUM_LAYERS, DROPOUT).to(device)
    # model = Seq2Seq(encoder, decoder).to(device)

    encoder = Encoder(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT).to(device)
    decoder = Decoder(HIDDEN_DIM, OUTPUT_DIM, NUM_LAYERS, DROPOUT).to(device)
    model = Seq2Seq(encoder, decoder).to(device)


    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    #criterion = nn.CrossEntropyLoss()
    criterion = nn.BCELoss()

    # Training loop
    N_EPOCHS = 50
    CLIP = 1

    best_valid_loss = float('inf')
    # Store metrics
    train_losses = []
    val_losses = []
    val_accuracies = []
    val_precisions = []
    val_recalls = []
    val_f1s = []

    # Early stopping parameters
    early_stopping = EarlyStopping(patience=10, min_delta=0.001)

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
            torch.save(model.state_dict(), 'best-model.pt')
        
        print(f'Epoch: {epoch+1:02}')
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}% |  Precision: {valid_precision:.2f} |  Recall: {valid_recall:.2f} |  F1 Score: {valid_f1:.2f}')

        early_stopping(valid_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    
    # Load the best model and test
    model.load_state_dict(torch.load('best-model.pt'))
    test_model(model, test_loader)

    plot_metrics(train_losses, val_losses, val_accuracies, val_precisions, val_recalls, val_f1s)