import pandas as pd
import ast

# Load train and test datasets
train_df = pd.read_csv('train_clean.csv')
test_df = pd.read_csv('test_clean.csv')

# Function to count rows based on the sum of scores
def count_sum_based(df):
    # Calculate the sum of scores for each row
    df['sum_scores'] = df['scores'].apply(lambda x: sum(ast.literal_eval(x)))

    # Count the number of rows where the sum is 0 and > 0
    count_sum_0 = df[df['sum_scores'] == 0].shape[0]
    count_sum_greater_0 = df[df['sum_scores'] > 0].shape[0]

    return count_sum_0, count_sum_greater_0

# Get the counts for the train and test datasets
train_sum_0, train_sum_greater_0 = count_sum_based(train_df)
test_sum_0, test_sum_greater_0 = count_sum_based(test_df)

print(f"Train Dataset - Rows with sum == 0: {train_sum_0}, Rows with sum > 0: {train_sum_greater_0}")
print(f"Test Dataset - Rows with sum == 0: {test_sum_0}, Rows with sum > 0: {test_sum_greater_0}")

# Step 1: Resample the dataset so the number of rows with sum == 0 equals the number of rows with sum > 0
def balance_dataset_by_sum(df, count_sum_greater_0):
    # Split the dataframe into rows where sum == 0 and sum > 0
    df_sum_0 = df[df['sum_scores'] == 0]
    df_sum_greater_0 = df[df['sum_scores'] > 0]

    # Undersample the rows where sum == 0 to match the number of rows where sum > 0
    df_sum_0_resampled = df_sum_0.sample(n=count_sum_greater_0, random_state=42)

    # Concatenate the resampled 'sum == 0' rows with the 'sum > 0' rows
    balanced_df = pd.concat([df_sum_0_resampled, df_sum_greater_0]).reset_index(drop=True)

    return balanced_df

# Step 2: Balance the train and test datasets based on the sum of scores
balanced_train_df = balance_dataset_by_sum(train_df, train_sum_greater_0)
balanced_test_df = balance_dataset_by_sum(test_df, test_sum_greater_0)

# Step 3: Split the balanced test dataset into two halves
test_first_half = balanced_test_df.sample(frac=0.5, random_state=42)
test_second_half = balanced_test_df.drop(test_first_half.index)

# Save the balanced and split datasets
balanced_train_df.to_csv('balanced_train.csv', index=False)
test_first_half.to_csv('balanced_val.csv', index=False)
test_second_half.to_csv('balanced_test.csv', index=False)

print("Balanced datasets saved successfully.")

