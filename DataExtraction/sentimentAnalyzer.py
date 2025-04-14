import pandas as pd
import re
import nltk
from nltk.tokenize import PunktSentenceTokenizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Download all necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# We need to make sure we use the correct tokenizer to avoid the punkt_tab error
from nltk.tokenize import PunktSentenceTokenizer
tokenizer = PunktSentenceTokenizer()

def load_transcripts(csv_path):
    """
    Load interview transcripts from a CSV file.
    
    Args:
        csv_path (str): Path to the CSV file
        
    Returns:
        pandas.DataFrame: DataFrame containing the transcripts
    """
    df = pd.read_csv(csv_path)
    
    # Assuming the first column is ID and second column is transcript
    # Rename columns for clarity
    if len(df.columns) >= 2:
        df = df.iloc[:, :2]  # Take just the first two columns
        df.columns = ['id', 'transcript']
    
    return df

def clean_speaker_labels(text):
    """
    Remove the 'Interviewer:' and 'Interviewee:' labels from text.
    
    Args:
        text (str): Text containing speaker labels
        
    Returns:
        str: Text with speaker labels removed
    """
    # Replace speaker labels with empty string
    cleaned_text = re.sub(r'Interviewer:\s*', '', text)
    cleaned_text = re.sub(r'Interviewee:\s*', '', cleaned_text)
    return cleaned_text

def split_into_sentences(transcript):
    """
    Split a transcript into individual sentences, removing speaker labels.
    
    Args:
        transcript (str): Full transcript text
        
    Returns:
        list: List of cleaned sentences
    """
    # First split by the delimiter '|' which separates dialogue turns
    turns = transcript.split('|')
    
    # Clean each turn by removing speaker labels
    cleaned_turns = [clean_speaker_labels(turn) for turn in turns]
    
    # Join all turns into a single text (makes sentence tokenization more accurate)
    full_text = ' '.join(cleaned_turns)
    
    # Split the text into sentences using our initialized tokenizer
    # instead of the sent_tokenize function that requires punkt_tab
    sentences = tokenizer.tokenize(full_text)
    
    # Clean up each sentence (remove extra whitespace)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences

def analyze_sentiment(sentences):
    """
    Analyze sentiment for a list of sentences using VADER.
    
    Args:
        sentences (list): List of sentences to analyze
        
    Returns:
        list: List of dictionaries containing sentences and their sentiment scores
    """
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
    """
    Process a single transcript and analyze sentiment.
    
    Args:
        transcript_id: ID of the transcript
        transcript_text (str): Full transcript text
        
    Returns:
        tuple: (transcript_id, sentences, sentiment_results)
    """
    # Split transcript into sentences
    sentences = split_into_sentences(transcript_text)
    
    # Analyze sentiment for each sentence
    sentiment_results = analyze_sentiment(sentences)
    
    return transcript_id, sentences, sentiment_results

def main(csv_path):
    """
    Main function to process transcripts and analyze sentiment.
    
    Args:
        csv_path (str): Path to the CSV file
    """
    # Load the transcripts
    print(f"Loading transcripts from {csv_path}...")
    transcripts_df = load_transcripts(csv_path)
    print(f"Loaded {len(transcripts_df)} transcripts.")
    
    # Create a sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()
    
    # Process each transcript
    all_results = []
    
    for idx, row in transcripts_df.iterrows():
        transcript_id = row['id']
        transcript_text = row['transcript']
        
        print(f"\nProcessing transcript {transcript_id}...")
        
        # Process the transcript and get sentiment results
        _, sentences, sentiment_results = process_transcript_sentiment(
            transcript_id, transcript_text
        )
        
        # Display results similar to the demo code
        print(f"Analyzing {len(sentences)} sentences for transcript {transcript_id}")
        for i, (sentence, result) in enumerate(zip(sentences, sentiment_results)):
            vs = result
            print("{:-<65} {}".format(sentence, str({
                'compound': vs['compound'],
                'neg': vs['neg'],
                'neu': vs['neu'],
                'pos': vs['pos']
            })))
        
        # Add transcript ID to each result and collect
        for result in sentiment_results:
            result['transcript_id'] = transcript_id
            all_results.append(result)
    
    # Create a DataFrame with all results
    results_df = pd.DataFrame(all_results)
    
    # Save results to CSV
    output_path = '../transcript_sentiment_analysis.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved sentiment analysis results to {output_path}")
    
    return results_df

def analyze_single_transcript(transcript_text):
    """
    Analyze a single transcript and print sentiment scores.
    Useful for testing or when you have just one transcript.
    
    Args:
        transcript_text (str): The transcript text to analyze
    """
    # Process the transcript
    _, sentences, sentiment_results = process_transcript_sentiment(
        "test", transcript_text
    )
    
    # Display results similar to the demo code
    print(f"Analyzing {len(sentences)} sentences")
    analyzer = SentimentIntensityAnalyzer()
    for sentence in sentences:
        vs = analyzer.polarity_scores(sentence)
        print("{:-<65} {}".format(sentence, str(vs)))
    
    return sentences, sentiment_results

def extract_sentiment_data(csv_path) -> pd.DataFrame:
    return main(csv_path)