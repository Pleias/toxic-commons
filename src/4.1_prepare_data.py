import pandas as pd
import os
from transformers import DebertaV2Tokenizer
from collections import defaultdict

# Initialize tokenizer
tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-small")

# File paths for the datasets
file_paths = {
    'English': ['annotated_US-PD-Books.csv', 'annotated_US-PD-Newspapers.csv'],
    'French': ['annotated_French-PD-Books.csv', 'annotated_French-PD-Newspapers.csv'],
    'Spanish': ['annotated_Spanish-PD-Books.csv', 'annotated_Spanish-PD-Books.csv'],
    'Italian': ['annotated_Italian-PD.csv'],
    'German': ['annotated_German-PD.csv', 'annotated_German-PD-Newspapers.csv'],
    'Polish': ['annotated_Polish-PD.csv'],
    'Dutch': ['annotated_Dutch-PD.csv'],
    'Latin': ['annotated_Latin-PD.csv']
}

# Desired proportions for each language
language_proportions = {
    'English': 1/4,
    'French': 1/4,
    'German': 1/10,
    'Spanish': 1/10,
    'Italian': 1/10,
    'Polish': 1/20,
    'Dutch': 1/20,
    'Latin': 1/20
}

# Step 1: Read and tokenize the data, calculate token count
def read_and_tokenize(file_path):
    print(file_path)
    df = pd.read_csv(os.path.join('merged_annotations', file_path))
    df = df.dropna(subset=['original_text'])
    df['token_count'] = df['original_text'].apply(lambda x: len(tokenizer.encode(x)))
    return df

# Step 2: Load and concatenate data for each language
data = defaultdict(list)
for language, files in file_paths.items():
    for file in files:
        data[language].append(read_and_tokenize(file))

for language in data:
    data[language] = pd.concat(data[language], ignore_index=True)

# Step 3: Calculate the total number of tokens for each language
total_tokens = {lang: data[lang]['token_count'].sum() for lang in data}

# Step 4: Calculate the limiting factor and target tokens for each language
limiting_factor = min(
    total_tokens['English'] / language_proportions['English'],
    total_tokens['French'] / language_proportions['French'],
    total_tokens['Spanish'] / language_proportions['Spanish'],
    total_tokens['Italian'] / language_proportions['Italian'],
    total_tokens['German'] / language_proportions['German'],
    total_tokens['Polish'] / language_proportions['Polish'],
    total_tokens['Dutch'] / language_proportions['Dutch'],
    total_tokens['Latin'] / language_proportions['Latin']
)

target_tokens = {lang: int(limiting_factor * language_proportions[lang]) for lang in total_tokens}

print("Target tokens per language:")
for lang, tokens in target_tokens.items():
    print(f"{lang}: {tokens} ({tokens/sum(target_tokens.values()):.2%})")

# Step 6: Sample data according to the target tokens
train_ratio = 0.7
test_ratio = 0.3

def sample_data(df, target_token_count):
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    cumulative_tokens = 0
    samples = []
    for _, row in df.iterrows():
        if cumulative_tokens >= target_token_count:
            break
        samples.append(row)
        cumulative_tokens += row['token_count']
    return pd.DataFrame(samples)

train_data = []
test_data = []

for language, df in data.items():
    train_target = int(target_tokens[language] * train_ratio)
    test_target = int(target_tokens[language] * test_ratio)

    train_samples = sample_data(df, train_target)
    train_data.append(train_samples)

    remaining_df = df[~df.index.isin(train_samples.index)]
    test_samples = sample_data(remaining_df, test_target)
    test_data.append(test_samples)

train_dataset = pd.concat(train_data, ignore_index=True)
test_dataset = pd.concat(test_data, ignore_index=True)

train_dataset = train_dataset.sample(frac=1, random_state=42).reset_index(drop=True)
test_dataset = test_dataset.sample(frac=1, random_state=42).reset_index(drop=True)

def print_stats(name, dataset):
    print(f"\n{name} dataset statistics:")
    print(f"Total samples: {len(dataset)}")
    print(f"Total tokens: {dataset['token_count'].sum()}")
    print("\nLanguage distribution:")
    for language in language_proportions:
        lang_count = len(dataset[dataset['original_text'].isin(data[language]['original_text'])])
        lang_tokens = dataset[dataset['original_text'].isin(data[language]['original_text'])]['token_count'].sum()
        print(f"{language}: {lang_count} samples ({lang_count/len(dataset):.2%}), {lang_tokens} tokens ({lang_tokens/dataset['token_count'].sum():.2%})")

print_stats("Training", train_dataset)
print_stats("Test", test_dataset)

train_dataset.to_csv('train.csv', index=False)
test_dataset.to_csv('test.csv', index=False)
print("\nDatasets saved as 'train.csv' and 'test.csv'")
