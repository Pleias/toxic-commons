import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import DebertaV2Tokenizer, DebertaV2PreTrainedModel, DebertaV2Model, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, balanced_accuracy_score
from tqdm import tqdm
import argparse
import pandas as pd
import numpy as np
import ast
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class MultiHeadTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        # logging.debug("Initializing MultiHeadTextDataset")
        self.texts = texts
        self.labels = labels  # List of lists, one label per head
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        # logging.debug(f"Dataset length: {len(self.texts)}")
        return len(self.texts)

    def __getitem__(self, idx):
        # logging.debug(f"Getting item at index {idx}")
        text = self.texts[idx]
        labels = self.labels[idx]  # List of labels for each head

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        # logging.debug(f"Encoding: {encoding}")
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            **{f'label_{i}': torch.tensor(label, dtype=torch.long) for i, label in enumerate(labels)}
        }

class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.num_classes = num_classes
        self.weights = torch.ones(num_classes)
        # logging.debug(f"Initialized WeightedCrossEntropyLoss with num_classes={4}")

    def forward(self, inputs, targets):
        # logging.debug(f"Calculating loss with inputs shape: {inputs.shape}, targets shape: {targets.shape}")
        return F.cross_entropy(inputs, targets, weight=self.weights.to(inputs.device))

    def update_weights(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred, labels=np.arange(self.num_classes))

        row_sums = cm.sum(axis=1)  # Total for each true class
        col_sums = cm.sum(axis=0)  # Total for each predicted class
        false_negatives = row_sums - cm.diagonal()  # FN: Samples missed for each class
        false_positives = col_sums - cm.diagonal()  # FP: Samples wrongly predicted as each class

        total_errors = false_negatives + false_positives
        weights = np.log1p(total_errors)  # Logarithmic scaling of errors
        weights = weights / weights.sum() * self.num_classes 
        self.weights = torch.tensor(weights, dtype=torch.float)
        return self.weights

class MultiHeadDebertaForSequenceClassification(DebertaV2PreTrainedModel):
    def __init__(self, config, num_heads):
        super().__init__(config)
        self.num_heads = num_heads
        self.deberta = DebertaV2Model(config)
        self.heads = nn.ModuleList([nn.Linear(config.hidden_size, 4) for _ in range(num_heads)])  # Updated to 4 classes
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.weighted_loss = WeightedCrossEntropyLoss(4)  # Updated to 4 classes
        self.post_init()
        # logging.debug(f"Initialized MultiHeadDebertaForSequenceClassification with num_heads={num_heads}")

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        #logging.debug(f"Forward pass with input_ids shape: {input_ids.shape}, attention_mask shape: {attention_mask.shape}")
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]  # (batch_size, sequence_length, hidden_size)
        #logging.debug(f"DeBERTa output shape: {sequence_output.shape}")

        logits_list = [head(self.dropout(sequence_output[:, 0, :])) for head in self.heads]
        #logging.debug(f"Logits list: {[logit.shape for logit in logits_list]}")

        logits = torch.stack(logits_list, dim=1)
        #logging.debug(f"Stacked logits shape: {logits.shape}")

        loss = None
        if labels is not None:
            loss = sum(self.weighted_loss(logits[:, i, :], labels[i]) for i in range(self.num_heads))
            #logging.debug(f"Calculating loss with labels: {loss}")
        return loss, logits

def weighted_accuracy(y_true, y_pred):
    # logging.debug("Calculating weighted accuracy")
    cm = confusion_matrix(y_true, y_pred)
    # logging.debug(f"Confusion Matrix: {cm}")
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    return np.average(cm.diagonal(), weights=np.sum(cm, axis=1))

def train_model(train_data, val_data, epochs):
    # logging.info("Starting training model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # logging.info(f"Using device: {device}")

    X_train, y_train = train_data
    X_val, y_val = val_data

    tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-small")
    model = MultiHeadDebertaForSequenceClassification.from_pretrained("microsoft/deberta-v3-small", num_heads=5)
    model = model.to(device)

    train_dataset = MultiHeadTextDataset(X_train, y_train, tokenizer, max_length=512)
    val_dataset = MultiHeadTextDataset(X_val, y_val, tokenizer, max_length=512)

    #train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    #val_dataloader = DataLoader(val_dataset, batch_size=32)
    class_counts = np.bincount(np.concatenate(y_train))  # Get the count of samples per class
    class_weights = 1. / class_counts  # Inverse frequency for each class

    # Assign weights to each sample in the dataset based on their class label
    sample_weights = np.array([np.mean([class_weights[label] for label in labels]) for labels in y_train])

    # Create the WeightedRandomSampler
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    # Use the sampler in the DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=64, sampler=sampler)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)


    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    best_weighted_acc = 0
    best_model_state = None
    early_stopping_threshold = 3
    epochs_without_improvement = 0

    for epoch in range(epochs):
        # logging.info(f"Epoch {epoch+1}/{epochs}")
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in progress_bar:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = [batch[f'label_{i}'].to(device) for i in range(5)]  # List of labels for each head

            for i, label in enumerate(labels):
                # logging.debug(f"Label {i} dtype: {label.dtype}, min: {label.min()}, max: {label.max()}")
                assert label.dtype == torch.long, f"Label {i} is not of type torch.long, but got {label.dtype}"
                assert label.min() >= 0 and label.max() < 4, \
                    f"Label {i} is out of range: min={label.min()}, max={label.max()}"

            loss, _ = model(input_ids, attention_mask=attention_mask, labels=labels)
            # logging.debug(f"Loss: {loss}")

            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()

            progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item())})

        avg_train_loss = total_loss / len(train_dataloader)
        # logging.info(f"Average training loss: {avg_train_loss}")

        # Validation phase: Accumulate predictions and true labels across the entire epoch
        val_preds, val_labels = [[] for _ in range(5)], [[] for _ in range(5)]
        model.eval()
        for batch in val_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = [batch[f'label_{i}'].to(device) for i in range(5)]

            with torch.no_grad():
                _, logits = model(input_ids, attention_mask=attention_mask)

            # For each head, accumulate predictions and labels for the entire epoch
            for i in range(5):
                val_preds[i].extend(torch.argmax(logits[:, i, :], dim=1).cpu().numpy())
                val_labels[i].extend(labels[i].cpu().numpy())

        # At the end of the epoch, calculate metrics and update class weights
        # Initialize early stopping counters for each head
        epochs_without_improvement_per_head = [0] * 5  # Assuming 5 heads

        for i in range(5):
            logging.debug(f"Updating weights for head {i}")

            # Ensure confusion matrix covers all classes
            y_true_epoch = val_labels[i]
            y_pred_epoch = val_preds[i]

            # Calculate confusion matrix at the epoch level
            cm = confusion_matrix(y_true_epoch, y_pred_epoch, labels=np.arange(4))  # Assuming 4 classes

            model.weighted_loss.update_weights(y_true_epoch, y_pred_epoch)

            # Calculate other metrics
            precision = precision_score(y_true_epoch, y_pred_epoch, average='weighted', zero_division=0)
            recall = recall_score(y_true_epoch, y_pred_epoch, average='weighted')
            f1 = f1_score(y_true_epoch, y_pred_epoch, average='weighted')
            accuracy = accuracy_score(y_true_epoch, y_pred_epoch)
            balanced_acc = balanced_accuracy_score(y_true_epoch, y_pred_epoch)
            weighted_acc = weighted_accuracy(y_true_epoch, y_pred_epoch)

            logging.info(f"Results for Category {i}, Epoch {epoch+1}")
            logging.info(f"Precision: {precision:.4f}")
            logging.info(f"Recall: {recall:.4f}")
            logging.info(f"F1 Score: {f1:.4f}")
            logging.info(f"Accuracy: {accuracy:.4f}")
            logging.info(f"Balanced Accuracy: {balanced_acc:.4f}")
            logging.info(f"Weighted Accuracy: {weighted_acc:.4f}")
            logging.info(f"Confusion Matrix: {cm}")
            logging.info(f"Updated class weights: {model.weighted_loss.weights}")

            # Check for model improvement for this head
            if weighted_acc > best_weighted_acc:
                best_weighted_acc = weighted_acc
                best_model_state = model.state_dict()
                logging.info(f"New best model saved with weighted accuracy: {best_weighted_acc:.4f}")
                epochs_without_improvement_per_head[i] = 0  # Reset counter for this head
            else:
                epochs_without_improvement_per_head[i] += 1
                logging.info(f"Epochs without improvement for head {i}: {epochs_without_improvement_per_head[i]}")

        # Check if all heads have reached the early stopping threshold
        if all(e >= early_stopping_threshold for e in epochs_without_improvement_per_head):
            logging.info(f"Early stopping triggered after {epoch+1} epochs")
            break

    if best_model_state is not None:
        torch.save(best_model_state, 'celadon.pt')
        logging.info(f"Best model saved with weighted accuracy: {best_weighted_acc:.4f}")

def main(epochs):
    # logging.info("Reading training and validation data")
    train_df = pd.read_csv("balanced_train.csv")
    val_df = pd.read_csv("balanced_val.csv")

    X_train = train_df['original_text'].tolist()
    y_train = [list(map(int, ast.literal_eval(label))) for label in train_df['scores']]
    X_val = val_df['original_text'].tolist()
    y_val = [list(map(int, ast.literal_eval(label))) for label in val_df['scores']]

    NUM_LABELS = 4  # Replace with the actual number of classes

    # Function to validate a label list
    def is_valid_label_list(label_list):
        # Check if all labels are integers and within the valid range
        return all(isinstance(label, int) and 0 <= label < NUM_LABELS for label in label_list)

    # Filter out rows with invalid labels in the training set
    X_train_filtered = []
    y_train_filtered = []

    for text, labels_str in zip(train_df['original_text'], train_df['scores']):
        try:
            # Convert string representation of labels to a list of integers
            label_list = list(map(int, ast.literal_eval(labels_str)))

            # Keep only entries with valid labels
            if is_valid_label_list(label_list):
                X_train_filtered.append(text)
                y_train_filtered.append(label_list)

        except (ValueError, SyntaxError):
            # Skip rows where ast.literal_eval fails (e.g., invalid label format)
            continue

    # Apply the same filtering process to the validation set
    X_val_filtered = []
    y_val_filtered = []

    for text, labels_str in zip(val_df['original_text'], val_df['scores']):
        try:
            # Convert string representation of labels to a list of integers
            label_list = list(map(int, ast.literal_eval(labels_str)))

            # Keep only entries with valid labels
            if is_valid_label_list(label_list):
                X_val_filtered.append(text)
                y_val_filtered.append(label_list)

        except (ValueError, SyntaxError):
            # Skip rows where ast.literal_eval fails (e.g., invalid label format)
            continue

    train_model((X_train_filtered, y_train_filtered), (X_val_filtered, y_val_filtered), epochs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a multi-head DeBERTa model")
    parser.add_argument("--epochs", type=int, required=True, help="Number of epochs to train")
    args = parser.parse_args()
    main(args.epochs)

