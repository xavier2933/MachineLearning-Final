import pandas as pd

from DataExtraction.countVectorizer import extract_word_count_data

if __name__ == '__main__':
    count_vectorizer_data: pd.DataFrame = extract_word_count_data()
