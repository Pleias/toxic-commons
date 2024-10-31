import os
import pandas as pd
from datasets import load_dataset

hf_org = 'PleIAs'

major_languages = ['US-PD-Newspapers', 'US-PD-Books', 'French-PD-Newspapers', 'French-PD-Books']
for dataset_name in major_languages:
    print(f'Beginning data loading for dataset {dataset_name} from Hugging Face...')
    dataset = load_dataset(f'{hf_org}/{dataset_name}')
    all_samples = dataset['train'].to_pandas()
    if len(all_samples) >= 250000:
        all_samples = all_samples[:250000]

    print(all_samples.head())
    all_samples.to_csv(f'samples/{dataset_name}_samples.csv', index=False)

german_samples = ['German-PD', 'German-PD-Newspapers']
for dataset_name in german_samples:
    print(f'Beginning data loading for dataset {dataset_name} from Hugging Face...')

    dataset = load_dataset(f'{hf_org}/{dataset_name}')
    all_samples = dataset['train'].to_pandas()

    if len(all_samples) >= 100000:
        all_samples = all_samples[:100000]

    print(all_samples.head())
    all_samples.to_csv(f'samples/{dataset_name}_samples.csv', index=False)

spanish_samples = ['Spanish-PD-Books', 'Spanish-PD-Newspapers']
for dataset_name in spanish_samples:
    print(f'Beginning data loading for dataset {dataset_name} from Hugging Face...')

    dataset = load_dataset(f'{hf_org}/{dataset_name}')
    all_samples = dataset['train'].to_pandas()

    if len(all_samples) >= 100000:
        all_samples = all_samples[:100000]

    print(all_samples.head())
    all_samples.to_csv(f'samples/{dataset_name}_samples.csv', index=False)

italian_samples = ['Italian-PD']
for dataset_name in italian_samples:
    print(f'Beginning data loading for dataset {dataset_name} from Hugging Face...')

    dataset = load_dataset(f'{hf_org}/{dataset_name}')
    all_samples = dataset['train'].to_pandas()

    if len(all_samples) >= 200000:
        all_samples = all_samples[:200000]

    print(all_samples.head())
    all_samples.to_csv(f'samples/{dataset_name}_samples.csv', index=False)

# Smaller datasets
small_samples = ['Dutch-PD', 'Portuguese-PD', 'Latin-PD', 'Polish-PD']
for dataset_name in small_samples:
    print(f'Beginning data loading for dataset {dataset_name} from Hugging Face...')

    dataset = load_dataset(f'{hf_org}/{dataset_name}')
    all_samples = dataset['train'].to_pandas()

    if len(all_samples) >= 100000:
        all_samples = all_samples[:100000]

    print(all_samples.head())
    all_samples.to_csv(f'samples/{dataset_name}_samples.csv', index=False)

