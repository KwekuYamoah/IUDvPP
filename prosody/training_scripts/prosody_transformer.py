import os
# Optional: Enable CUDA_LAUNCH_BLOCKING for debugging (uncomment if needed)
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import json
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import multilabel_confusion_matrix, precision_recall_fscore_support
import string
import pandas as pd
import re

# Set the random seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# Custom padding value for labels
PADDING_VALUE = 0  # Use 0 as the padding index

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

# Preprocess words (remove punctuation and tokenize)
def preprocess_text(words):
    processed_words = []
    for word in words:
        processed_words.extend(re.findall(r"[\w']+|[.,!?;]", word))
    return processed_words

# Clean up sentence by removing padding
def clean_up_sentence(words, gold_labels, pred_labels, padding_value):
    filtered_words = []
    filtered_gold_labels = []
    filtered_pred_labels = []
    
    # Note: We no longer remove punctuation here to keep all words
    for i in range(len(words)):
        if gold_labels[i] != padding_value:
            filtered_words.append(words[i])
            filtered_gold_labels.append(int(gold_labels[i]))
            filtered_pred_labels.append(int(pred_labels[i]))
    return filtered_words, filtered_gold_labels, filtered_pred_labels

# Custom dataset class for prosody features
class ProsodyDataset(Dataset):
    def __init__(self, data):
        self.entries = list(data.items())

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        key, item = self.entries[idx]
        words = preprocess_text(item['words'])
        features = torch.tensor(item['features'], dtype=torch.float32)
        labels = torch.tensor(item['labels'], dtype=torch.long) + 1  # Shift labels by +1
        return words, features, labels

# Custom collate function to handle padding
def collate_fn(batch):
    words = [item[0] for item in batch]  # List of lists of words
    features = [item[1] for item in batch]
    labels = [item[2] for item in batch]

    features_padded = torch.nn.utils.rnn.pad_sequence(features, batch_first=True, padding_value=0.0)
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=PADDING_VALUE)
    lengths = torch.tensor([len(f) for f in features])

    return words, features_padded, labels_padded, lengths

# Positional Encoding for Transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))  # [d_model/2]
        pe[:, 0::2] = torch.sin(position * div_term)  # [max_len, d_model/2]
        pe[:, 1::2] = torch.cos(position * div_term)  # [max_len, d_model/2]
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return x

# Transformer Encoder
class TransformerEncoder(nn.Module):
    def __init__(self, feature_dim, hidden_dim, num_layers, dropout, num_heads=8):
        super(TransformerEncoder, self).__init__()
        self.input_fc = nn.Linear(feature_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(d_model=hidden_dim, max_len=5000)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True  # Align with data format
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features):
        src = self.input_fc(features)  # [batch_size, seq_len, hidden_dim]
        src = self.positional_encoding(src)  # [batch_size, seq_len, hidden_dim]
        src = self.dropout(src)
        memory = self.transformer_encoder(src)  # [batch_size, seq_len, hidden_dim]
        return memory

# Transformer Decoder
class TransformerDecoder(nn.Module):
    def __init__(self, hidden_dim, output_dim, num_layers, dropout, num_classes=2, num_heads=8):
        super(TransformerDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.embedding = nn.Embedding(
            num_embeddings=num_classes,
            embedding_dim=hidden_dim,
            padding_idx=0  # Set padding_idx to 0
        )
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True  # Align with data format
        )
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(hidden_dim, num_classes)  # Ensure output_dim == num_classes

    def forward(self, memory, tgt):
        tgt_embed = self.embedding(tgt)  # [batch_size, seq_len, hidden_dim]
        output = self.transformer_decoder(tgt_embed, memory)  # [batch_size, seq_len, hidden_dim]
        output = self.fc_out(output)  # [batch_size, seq_len, num_classes]
        return output

# TransformerSeq2Seq combining Encoder and Decoder
class TransformerSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(TransformerSeq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, features, labels, lengths):
        memory = self.encoder(features)  # [batch_size, seq_len, hidden_dim]
        outputs = self.decoder(memory, labels)  # [batch_size, seq_len, num_classes]
        return outputs

# Training function
def train(model, iterator, optimizer, criterion, num_classes=2):
    model.train()
    epoch_loss = 0
    for batch_idx, (words, features, labels, lengths) in enumerate(iterator):
        features = features.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)

        optimizer.zero_grad()
        output = model(features, labels, lengths)  # [batch_size, seq_len, num_classes]
        output = output.view(-1, num_classes)  # [batch_size * seq_len, num_classes]
        labels = labels.view(-1)  # [batch_size * seq_len]

        # Exclude padding indices from loss
        loss = criterion(output, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
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
        for batch_idx, (words, features, labels, lengths) in enumerate(iterator):
            features = features.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)

            output = model(features, labels, lengths)  # [batch_size, seq_len, num_classes]
            output = output.view(-1, num_classes)  # [batch_size * seq_len, num_classes]
            labels = labels.view(-1)  # [batch_size * seq_len]

            preds = torch.argmax(output, dim=1)  # [batch_size * seq_len]

            loss = criterion(output, labels)
            epoch_loss += loss.item()

            # Exclude padding indices from metrics
            non_pad_indices = labels != PADDING_VALUE
            labels_np = labels[non_pad_indices].cpu().numpy() - 1  # Adjust labels back by -1
            preds_np = preds[non_pad_indices].cpu().numpy() - 1    # Adjust preds back by -1
            all_labels.extend(labels_np)
            all_preds.extend(preds_np)

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    return epoch_loss / len(iterator), accuracy, precision, recall, f1

# Test model function
def test_model(model, iterator):
    model.eval()
    all_labels = []
    all_preds = []

    os.makedirs('./outputs', exist_ok=True)
    with open('./outputs/prosody_transformer_multiclass_results.txt', 'w') as file:
        file.write("")

    with torch.no_grad():
        for batch_idx, (words, features, labels, lengths) in enumerate(iterator):
            features = features.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)

            logits = model(features, labels, lengths)  # [batch_size, seq_len, num_classes]
            preds = torch.argmax(logits, dim=2)  # [batch_size, seq_len]

            for i in range(features.size(0)):
                word_sentence = words[i]  # List of words
                gold_labels = labels[i].cpu().numpy().flatten()
                pred_labels = preds[i].cpu().numpy().flatten()

                # Adjust labels back by -1
                gold_labels = gold_labels - 1
                pred_labels = pred_labels - 1
                PADDING_VALUE_EVAL = -1  # Update padding value for evaluation

                # Clean up the sentence by excluding padding positions
                cleaned_words, cleaned_gold_labels, cleaned_pred_labels = clean_up_sentence(
                    word_sentence, gold_labels, pred_labels, padding_value=PADDING_VALUE_EVAL
                )

                # Create DataFrame
                data = {
                    'Word': cleaned_words,
                    'Gold Label': cleaned_gold_labels,
                    'Predicted Label': cleaned_pred_labels
                }

                df = pd.DataFrame(data)
                with open('./outputs/prosody_transformer_multiclass_results.txt', 'a') as file:
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

    print('*' * 45)
    print(f'Test Accuracy: {accuracy*100:.2f}%')
    print(f'Test Precision: {precision*100:.2f}')
    print(f'Test Recall: {recall*100:.2f}')
    print(f'Test F1 Score: {f1*100:.2f}')
    print('*' * 45)

    return all_labels, all_preds

# Plot metrics function
def plot_metrics(train_losses, val_losses, val_accuracies, val_precisions, val_recalls, val_f1s):
    import matplotlib.pyplot as plt
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 3, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.legend()
    plt.title('Loss')
    
    plt.subplot(2, 3, 2)
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy')
    
    plt.subplot(2, 3, 3)
    plt.plot(epochs, val_precisions, label='Validation Precision')
    plt.legend()
    plt.title('Precision')
    
    plt.subplot(2, 3, 4)
    plt.plot(epochs, val_recalls, label='Validation Recall')
    plt.legend()
    plt.title('Recall')
    
    plt.subplot(2, 3, 5)
    plt.plot(epochs, val_f1s, label='Validation F1 Score')
    plt.legend()
    plt.title('F1 Score')
    
    plt.tight_layout()
    plt.savefig('./outputs/transformer_multiclass_metrics.png')
    plt.close()

# Evaluate on a new dataset
def evaluate_new_set(model, new_dataset_path):
    # Load new data
    new_data = load_data(new_dataset_path)
    new_dataset = ProsodyDataset(new_data)
    new_loader = DataLoader(new_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
    
    # Test the model on the new dataset and get predictions
    print('\n\nEvaluation on Held Out Set Dataset:')
    all_labels, all_preds = test_model(model, new_loader)

    return all_labels, all_preds

# Validate label integrity
def validate_labels(datasets, num_classes):
    for dataset in datasets:
        for idx, (_, _, labels) in enumerate(dataset):
            invalid_mask = (labels >= num_classes) | (labels < 0)
            if torch.any(invalid_mask):
                invalid_labels = labels[invalid_mask].unique().tolist()
                raise ValueError(f"Found invalid labels {invalid_labels} in dataset at index {idx}. Labels should be in the range [0, {num_classes - 1}].")
    print("All labels are valid.")

# Get all unique labels across multiple datasets
def get_all_unique_labels(datasets):
    unique_labels = set()
    for dataset in datasets:
        for _, _, labels in dataset:
            unique_labels.update(labels.numpy().flatten())
    return unique_labels

# Main script
if __name__ == "__main__":
    set_seed(42)
    json_path = '../prosody/data/ambiguous_prosody_multi_label_features_train.json'
    data = load_data(json_path)

    # Create a descriptive filename for the model
    dataset_name = "ambiguous_instructions"
    task_name = "prosody_multiclass"
    best_model_filename = f"models/best-transformer-model-{dataset_name}-{task_name}.pt"

    # Split data
    train_data, val_data, test_data = split_data(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)

    # Create datasets
    train_dataset = ProsodyDataset(dict(train_data))
    val_dataset = ProsodyDataset(dict(val_data))
    test_dataset = ProsodyDataset(dict(test_data))

    print(f'Train size: {len(train_dataset)}')
    print(f'Validation size: {len(val_dataset)}')
    print(f'Test size: {len(test_dataset)}')

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # Retrieve a sample batch to determine feature dimensions
    sample_words, sample_features, sample_labels, sample_lengths = next(iter(train_loader))
    feature_dim = sample_features.shape[2]

    # Determine number of classes based on all datasets
    all_unique_labels = get_all_unique_labels([train_dataset, val_dataset, test_dataset])
    NUM_CLASSES = len(all_unique_labels) + 1  # Increase by 1 due to label shifting
    print(f"All unique labels across datasets: {sorted(all_unique_labels)}")
    print(f"Model Training with {NUM_CLASSES} classes")

    # Define model hyperparameters
    HIDDEN_DIM = 128
    OUTPUT_DIM = NUM_CLASSES  # Updated to num_classes
    NUM_LAYERS = 2
    DROPOUT = 0.46413941258903124
    NUM_HEADS = 8

    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Initialize Encoder and Decoder
    encoder = TransformerEncoder(feature_dim, HIDDEN_DIM, NUM_LAYERS, DROPOUT, num_heads=NUM_HEADS).to(device)
    decoder = TransformerDecoder(HIDDEN_DIM, OUTPUT_DIM, NUM_LAYERS, DROPOUT, num_classes=NUM_CLASSES, num_heads=NUM_HEADS).to(device)
    model = TransformerSeq2Seq(encoder, decoder).to(device)

    summary(model, input_data=(sample_features.to(device),sample_labels.to(device), sample_lengths.to(device)), device=device)

    # Validate labels before training
    validate_labels([train_dataset, val_dataset, test_dataset], NUM_CLASSES)

    # Define optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), 
                           lr=0.001175385480166815, 
                           weight_decay=1.3835287809131501e-05)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Define loss function
    criterion = nn.CrossEntropyLoss(ignore_index=PADDING_VALUE)

    # Initialize lists to store metrics
    train_losses = []
    val_losses = []
    val_accuracies = []
    val_precisions = []
    val_recalls = []
    val_f1s = []

    # Define training parameters
    N_EPOCHS = 5
    CLIP = 1

    # Initialize early stopping
    early_stopping = EarlyStopping(patience=10, min_delta=0.001)
    best_valid_loss = float('inf')

    # Ensure the models directory exists
    os.makedirs('models', exist_ok=True)

    # Training loop
    for epoch in range(N_EPOCHS):
        train_loss = train(model, train_loader, optimizer, criterion, num_classes=NUM_CLASSES)
        valid_loss, valid_acc, valid_precision, valid_recall, valid_f1 = evaluate(model, val_loader, criterion, num_classes=NUM_CLASSES)

        train_losses.append(train_loss)
        val_losses.append(valid_loss)
        val_accuracies.append(valid_acc)
        val_precisions.append(valid_precision)
        val_recalls.append(valid_recall)
        val_f1s.append(valid_f1)

        # Update the learning rate scheduler
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(valid_loss)
        else:
            scheduler.step()

        # Save the model if validation loss has decreased
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), best_model_filename)

        # Print training progress
        print(f'Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Val. Loss: {valid_loss:.3f} | '
              f'Val. Acc: {valid_acc*100:.2f}% | Precision: {valid_precision:.2f} | '
              f'Recall: {valid_recall:.2f} | F1 Score: {valid_f1:.2f}')

        # Early stopping check
        early_stopping(valid_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # Load the best model and evaluate on the test set
    model.load_state_dict(torch.load(best_model_filename))
    test_loss, test_acc, test_precision, test_recall, test_f1 = evaluate(model, test_loader, criterion, num_classes=NUM_CLASSES)
    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}% | '
          f'Precision: {test_precision:.2f} | Recall: {test_recall:.2f} | F1 Score: {test_f1:.2f}')

    # Generate detailed test results
    test_model(model, test_loader)

    # Plot training and validation metrics
    plot_metrics(train_losses, val_losses, val_accuracies, val_precisions, val_recalls, val_f1s)

    # Evaluate model on held out set
    eval_json = "../prosody/data/ambiguous_prosody_multi_label_features_eval.json"
    # Evaluate the model on the new dataset
    true_labels, predicted_labels  = evaluate_new_set(model, eval_json)

    #log dir
    log_dir = "../prosody/outputs"

    # class names 
    class_names = [0,1,2]

    # Compute precision, recall, f1-score, and support for each class
    class_precision, class_recall, class_f1, class_support = precision_recall_fscore_support(
        true_labels, predicted_labels, average=None, zero_division=0)
    
    print(class_support)
    # Compute the multilabel confusion matrix
    confusion_matrices = multilabel_confusion_matrix(true_labels, predicted_labels)

    with open(f"{log_dir}/classwise_metrics.txt", "w") as f:
            for i, class_name in enumerate(class_names):
                f.write(f"{class_name}:\n")
                f.write(f"  Precision: {class_precision[i]:.4f}\n")
                f.write(f"  Recall: {class_recall[i]:.4f}\n")
                f.write(f"  F1-Score: {class_f1[i]:.4f}\n")
                f.write(f"  Support (True instances in eval data): {class_support[i]}\n")
                f.write("-" * 40 + "\n")

    #write confusion matrix to a file
    with open(f"{log_dir}/confusion_matrix.txt", "w") as f:
        for i, class_name in enumerate(class_names):
            tn, fp, fn, tp = confusion_matrices[i].ravel()  # Extract the values from the confusion matrix

            f.write(f"\nConfusion Matrix for {class_name}:\n")
            f.write(f"True Negatives (TN): {tn}\n")
            f.write(f"False Positives (FP): {fp}\n")
            f.write(f"False Negatives (FN): {fn}\n")
            f.write(f"True Positives (TP): {tp}\n")

            f.write("\nInterpretation:\n")
            f.write(f"  - The model correctly predicted that '{class_name}' is NOT present {tn} times (TN).\n")
            f.write(f"  - The model incorrectly predicted that '{class_name}' is present {fp} times when it was actually NOT present (FP).\n")
            f.write(f"  - The model incorrectly predicted that '{class_name}' is NOT present {fn} times when it was actually present (FN).\n")
            f.write(f"  - The model correctly predicted that '{class_name}' is present {tp} times (TP).\n")
            f.write("-" * 40 + "\n")
