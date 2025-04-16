import pandas as pd
import re
import nltk
from nltk.tokenize import PunktSentenceTokenizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# We need to make sure we use the correct tokenizer to avoid the punkt_tab error
from nltk.tokenize import PunktSentenceTokenizer
tokenizer = PunktSentenceTokenizer()

def load_transcripts(csv_path):

    df = pd.read_csv(csv_path)

    if len(df.columns) >= 2:
        df = df.iloc[:, :2]  # Take just the first two columns
        df.columns = ['id', 'transcript']
    
    return df

def clean_speaker_labels(text):

    cleaned_text = re.sub(r'Interviewer:\s*', '', text)
    cleaned_text = re.sub(r'Interviewee:\s*', '', cleaned_text)
    return cleaned_text

def split_into_sentences(transcript):

    turns = transcript.split('|')
    cleaned_turns = [clean_speaker_labels(turn) for turn in turns]

    full_text = ' '.join(cleaned_turns)
    sentences = tokenizer.tokenize(full_text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences

def analyze_sentiment(sentences):

    analyzer = SentimentIntensityAnalyzer()
    results = []
    
    for sentence in sentences:
        sentiment = analyzer.polarity_scores(sentence)
        
        results.append({
            'sentence': sentence,
            'compound': sentiment['compound'],
            'pos': sentiment['pos'],
            'neu': sentiment['neu'],
            'neg': sentiment['neg']
        })
    
    return results

def process_transcript_sentiment(transcript_id, transcript_text):

    sentences = split_into_sentences(transcript_text)
    sentiment_results = analyze_sentiment(sentences)
    
    return transcript_id, sentences, sentiment_results

def analyze_single_transcript(transcript_text):

    _, sentences, sentiment_results = process_transcript_sentiment(
        "test", transcript_text
    )

    print(f"Analyzing {len(sentences)} sentences")
    analyzer = SentimentIntensityAnalyzer()
    for sentence in sentences:
        vs = analyzer.polarity_scores(sentence)
        print("{:-<65} {}".format(sentence, str(vs)))
    
    return sentences, sentiment_results

def extract_sentiment_data(csv_path) -> pd.DataFrame:
    transcripts_df = load_transcripts(csv_path)

    output_data = pd.DataFrame(
        columns=['average_sentence_sentiment', 'minimum_sentence_sentiment', 'maximum_sentence_sentiment'])

    for index, row in transcripts_df.iterrows():
        transcript_id = row['id']
        transcript_text = row['transcript']

        _, sentences, sentiment_results = process_transcript_sentiment(
            transcript_id, transcript_text
        )

        compounds = [sentence_sentiment['compound'] for sentence_sentiment in sentiment_results]

        average_compound = sum(compounds) / len(compounds)
        max_compound = max(compounds)
        min_compound = min(compounds)

        new_row = pd.DataFrame({
            'average_sentence_sentiment': [average_compound],
            'minimum_sentence_sentiment': [min_compound],
            'maximum_sentence_sentiment': [max_compound]
        })
        output_data = pd.concat([output_data, new_row], ignore_index=True)

    return output_data