import pandas as pd
import argparse 
import glob

parser = argparse.ArgumentParser(description="Process batch params")
parser.add_argument('--dataset', help="dataset to pull from")
parser.add_argument('--count', help="number of files")
parser.add_argument('--mode', choices=['new', 'merged'], help="Combination mode, can either 'new' or 'merged'.", required=True)
args = parser.parse_args()

all_annotations = pd.DataFrame()

if args.mode == 'new':
    for i in range(1, int(args.count) + 1):
        temp = pd.read_csv(f'annotations/new_llama_annotations_{args.dataset}_{i}.csv')
        all_annotations = pd.concat([all_annotations, temp])

    all_annotations.to_csv(f'merged_annotations/annotated_{args.dataset}.csv')
    print(len(all_annotations))

elif args.mode == 'merged':
    for file in list(glob.glob('merged_annotations/*')):
        temp = pd.read_csv(file)
        all_annotations = pd.concat([all_annotations, temp])
    all_annotations.to_csv('dataset.csv')
    print(len(all_annotations))

