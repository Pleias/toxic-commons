import pandas as pd
import ast
import re

def clean_and_resave_data():
    def is_valid_score(score_str):
        try:
            score = ast.literal_eval(score_str)
            return isinstance(score, list) and len(score) == 5 and all(isinstance(x, int) and 0 <= x <= 3 for x in score)
        except:
            return False

    def fix_score(score_str):
        parts = score_str.split("im_end")
        score_pattern = r'##\s*.*?\s*Score\s*##\s*:\s*(\d+)'
        
        def extract_scores(part):
            matches = re.findall(score_pattern, part)
            if len(matches) >= 5:
                return [min(3, max(0, int(x))) for x in matches[:5]]
            return None
        
        for part in parts:
            scores = extract_scores(part)
            if scores:
                return str(scores)
        return None

    def process_file(file_path):
        df = pd.read_csv(file_path)
        df['valid_score'] = df['scores'].apply(is_valid_score)
        
        valid_df = df[df['valid_score']].copy()
        invalid_df = df[~df['valid_score']].copy()
        
        fixed_scores = invalid_df['scores'].apply(fix_score)
        fixed_df = invalid_df[fixed_scores.notnull()].copy()
        fixed_df.loc[:, 'scores'] = fixed_scores[fixed_scores.notnull()]
        
        unfixable_df = invalid_df[fixed_scores.isnull()]
        
        print(f"\nProcessing {file_path}:")
        print(f"Total samples: {len(df)}")
        print(f"Valid samples: {len(valid_df)}")
        print(f"Fixed samples: {len(fixed_df)}")
        print(f"Unfixable samples: {len(unfixable_df)}")
        
        if len(unfixable_df) > 0:
            print("\nExamples of unfixable samples:")
            for _, row in unfixable_df.head().iterrows():
                print(f"Text: {row['original_text'][:100]}...")
                print(f"Score: {row['scores']}")
                print("---")
        
        # Combine valid and fixed samples
        combined_df = pd.concat([valid_df, fixed_df])
        
        # Shuffle the combined dataframe
        shuffled_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        return shuffled_df.drop('valid_score', axis=1)

    # Process both files
    train_df = process_file('train.csv')
    val_df = process_file('test.csv')

    # Save the new files
    train_df.to_csv('train_clean.csv', index=False)
    val_df.to_csv('test_clean.csv', index=False)

    print("\nCleaned and shuffled data saved as 'train_clean.csv' and 'test_clean.csv'")

if __name__ == "__main__":
    clean_and_resave_data()
