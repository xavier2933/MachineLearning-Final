import pandas as pd

from DataExtraction.countVectorizer import extract_word_count_data
from DataExtraction.sentimentAnalyzer import extract_sentiment_data
from DataExtraction.fillerWords import compute_filler_percent
from DataExtraction.speakerBalance import load_and_parse_transcripts, compute_speaker_balance
from DataExtraction.wordCounter import extract_word_count_features

if __name__ == '__main__':

    CSV_FILE_PATH = "../RawProvidedData/MITInterview/transcripts.csv"

    count_vectorizer_data: pd.DataFrame = extract_word_count_data(CSV_FILE_PATH)
    sentiment_data: pd.DataFrame = extract_sentiment_data(CSV_FILE_PATH)
    word_count_features: pd.DataFrame = extract_word_count_features(count_vectorizer_data)

    df_transcripts = load_and_parse_transcripts(CSV_FILE_PATH)
    filler_percent_df = compute_filler_percent(df_transcripts)
    speaker_balance_df = compute_speaker_balance(df_transcripts)

    features_df = pd.concat([filler_percent_df, speaker_balance_df, word_count_features, sentiment_data], axis=1)
    features_df.to_csv('features.csv', index=False)
