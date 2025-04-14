import pandas as pd

from DataExtraction.countVectorizer import extract_word_count_data
from DataExtraction.sentimentAnalyzer import extract_sentiment_data

if __name__ == '__main__':

    CSV_FILE_PATH = "../RawProvidedData/MITInterview/transcripts.csv"

    count_vectorizer_data: pd.DataFrame = extract_word_count_data(CSV_FILE_PATH)
    sentiment_data: pd.DataFrame = extract_sentiment_data(CSV_FILE_PATH)

    count_vectorizer_data.to_csv('word_counts.csv')
    sentiment_data.to_csv('sentiment_data.csv')