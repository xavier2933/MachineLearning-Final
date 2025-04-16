import pandas as pd

from DataExtraction.countVectorizer import extract_word_count_data
from DataExtraction.sentimentAnalyzer import extract_sentiment_data
from DataExtraction.fillerWords import load_and_parse_transcripts, compute_filler_percent

if __name__ == '__main__':

    CSV_FILE_PATH = "RawProvidedData/MITInterview/transcripts.csv"

    count_vectorizer_data: pd.DataFrame = extract_word_count_data(CSV_FILE_PATH)
    sentiment_data: pd.DataFrame = extract_sentiment_data(CSV_FILE_PATH)

    # Filler word data extraction:
    df_transcripts = load_and_parse_transcripts(CSV_FILE_PATH)
    filler_percent_df = compute_filler_percent(df_transcripts)

    count_vectorizer_data.to_csv('word_counts.csv')
    sentiment_data.to_csv('sentiment_data.csv')
    filler_percent_df.to_csv('filler_word_counts.csv', index=False)
