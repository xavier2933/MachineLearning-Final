import pandas as pd

def extract_word_count_features(word_counts_data: pd.DataFrame) -> pd.DataFrame:

    word_counts_data = word_counts_data.select_dtypes(include='int')

    word_counts_features = pd.DataFrame()
    word_counts_features['total_word_count'] = word_counts_data.sum(axis=1)

    for length in range(2, 8):
        matching_cols = [column for column in word_counts_data.columns if len(column) == length]

        if matching_cols:
            word_counts_features[f'word_length_{length}'] = word_counts_data[matching_cols].sum(axis=1) / word_counts_features['total_word_count']

            word_counts_data = word_counts_data.drop(columns=matching_cols)

        else:
            word_counts_features[f'word_length_{length}'] = 0

    word_counts_features['8+ words'] = word_counts_data.sum(axis=1) / word_counts_features['total_word_count']

    return word_counts_features