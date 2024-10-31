import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import DebertaV2Tokenizer, DebertaV2PreTrainedModel, DebertaV2Model
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, balanced_accuracy_score
import pandas as pd
import numpy as np
import ast
import logging
from tqdm import tqdm 

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Dataset class
class MultiHeadTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels  # List of lists, one label per head
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
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

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            **{f'label_{i}': torch.tensor(label, dtype=torch.long) for i, label in enumerate(labels)}
        }

# Custom Multi-Head Model class
class MultiHeadDebertaForSequenceClassification(DebertaV2PreTrainedModel):
    def __init__(self, config, num_heads):
        super().__init__(config)
        self.num_heads = num_heads
        self.deberta = DebertaV2Model(config)
        self.heads = nn.ModuleList([nn.Linear(config.hidden_size, 4) for _ in range(num_heads)])  # Assuming 4 classes
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]  # (batch_size, sequence_length, hidden_size)

        logits_list = [head(self.dropout(sequence_output[:, 0, :])) for head in self.heads]
        logits = torch.stack(logits_list, dim=1)  # Shape: (batch_size, num_heads, num_classes)

        return logits

# Function to calculate weighted accuracy
def weighted_accuracy(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    return np.average(cm.diagonal(), weights=np.sum(cm, axis=1))

# Validation loop
def validate_model(val_data):
    logging.info("Starting validation")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_val, y_val = val_data

    tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-small")
    model = MultiHeadDebertaForSequenceClassification.from_pretrained("microsoft/deberta-v3-small", num_heads=5)
    model.load_state_dict(torch.load('celadon.pt', map_location=device))
    model = model.to(device)

    val_dataset = MultiHeadTextDataset(X_val, y_val, tokenizer, max_length=512)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Prepare for validation
    val_preds, val_labels = [[] for _ in range(5)], [[] for _ in range(5)]
    model.eval()
    progress_bar = progress_bar = tqdm(val_dataloader, desc=f"Test")
    for batch in progress_bar: 
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = [batch[f'label_{i}'].to(device) for i in range(5)]

        with torch.no_grad():
            logits = model(input_ids, attention_mask=attention_mask)

        # Accumulate predictions and labels for the entire dataset
        for i in range(5):
            val_preds[i].extend(torch.argmax(logits[:, i, :], dim=1).cpu().numpy())
            val_labels[i].extend(labels[i].cpu().numpy())

    # Evaluate metrics for each head
    for i in range(5):
        logging.info(f"Evaluating metrics for head {i}")
        y_true_epoch = val_labels[i]
        y_pred_epoch = val_preds[i]

        cm = confusion_matrix(y_true_epoch, y_pred_epoch, labels=np.arange(4))  # Assuming 4 classes

        precision = precision_score(y_true_epoch, y_pred_epoch, average='weighted', zero_division=0)
        recall = recall_score(y_true_epoch, y_pred_epoch, average='weighted')
        f1 = f1_score(y_true_epoch, y_pred_epoch, average='weighted')
        accuracy = accuracy_score(y_true_epoch, y_pred_epoch)
        balanced_acc = balanced_accuracy_score(y_true_epoch, y_pred_epoch)
        weighted_acc = weighted_accuracy(y_true_epoch, y_pred_epoch)

        logging.info(f"Results for Head {i}")
        logging.info(f"Precision: {precision:.4f}")
        logging.info(f"Recall: {recall:.4f}")
        logging.info(f"F1 Score: {f1:.4f}")
        logging.info(f"Accuracy: {accuracy:.4f}")
        logging.info(f"Balanced Accuracy: {balanced_acc:.4f}")
        logging.info(f"Weighted Accuracy: {weighted_acc:.4f}")
        logging.info(f"Confusion Matrix: \n{cm}")

# Main function
def main():
    logging.info("Reading test data")
    val_df = pd.read_csv("balanced_test.csv")

    X_val = val_df['original_text'].tolist()
    y_val = [list(map(int, ast.literal_eval(label))) for label in val_df['scores']]

    # Filtering valid labels
    NUM_LABELS = 4

    def is_valid_label_list(label_list):
        return all(isinstance(label, int) and 0 <= label < NUM_LABELS for label in label_list)

    X_val_filtered = []
    y_val_filtered = []

    for text, labels_str in zip(val_df['original_text'], val_df['scores']):
        try:
            label_list = list(map(int, ast.literal_eval(labels_str)))
            if is_valid_label_list(label_list):
                X_val_filtered.append(text)
                y_val_filtered.append(label_list)
        except (ValueError, SyntaxError):
            continue

    validate_model((X_val_filtered, y_val_filtered))

if __name__ == "__main__":
    main()

