import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import DebertaV2Tokenizer, DebertaV2PreTrainedModel, DebertaV2Model
import pandas as pd
import numpy as np
import ast
import logging
from tqdm import tqdm
import time
from model import MultiHeadDebertaForSequenceClassification

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MultiHeadTextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

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
            'text': text
        }

def predict(input_data):
    logging.info("Starting prediction")
    start_time = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_input = input_data

    tokenizer = AutoTokenizer.from_pretrained("PleIAs/celadon")
    model = MultiHeadDebertaForSequenceClassification.from_pretrained("PleIAs/celadon")
    model = model.to(device)

    input_dataset = MultiHeadTextDataset(X_input, tokenizer, max_length=512)
    input_dataloader = DataLoader(input_dataset, batch_size=32, shuffle=False)

    model.eval()
    results = []
    progress_bar = tqdm(input_dataloader, desc=f"Predicting")
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        texts = batch['text']

        with torch.no_grad():
            logits = model(input_ids, attention_mask=attention_mask)

        for i, text in enumerate(texts):
            scores = [torch.argmax(logits[i, head, :]).item() for head in range(5)]
            results.append({'text': text, 'scores': scores})

    output_df = pd.DataFrame(results)
    output_df.to_csv('/path/to/your/output.csv', index=False)

    end_time = time.time()
    total_time = end_time - start_time
    logging.info(f"Total prediction time: {total_time:.2f} seconds")

def main():
    logging.info("Reading input data")
    input_df = pd.read_csv("/path/to/your/input.csv")

    X_input = input_df['text'].tolist() if 'text' in input_df.columns else input_df['processed_text'].tolist()
    predict(X_input)

if __name__ == "__main__":
    main()

