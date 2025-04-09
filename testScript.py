import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


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

def parse_transcript(transcript):
    """
    Parse a transcript to separate interviewer and interviewee text.
    
    Args:
        transcript (str): Full transcript text
        
    Returns:
        tuple: (interviewer_text, interviewee_text)
    """
    # Split by the delimiter '|' which separates dialogue
    parts = transcript.split('|')
    
    interviewer_parts = []
    interviewee_parts = []
    
    for part in parts:
        # Check if this part starts with "Interviewer:" or "Interviewee:"
        if part.strip().startswith("Interviewer:"):
            interviewer_parts.append(part.replace("Interviewer:", "").strip())
        elif part.strip().startswith("Interviewee:"):
            interviewee_parts.append(part.replace("Interviewee:", "").strip())
    
    # Join all parts for each speaker
    interviewer_text = " ".join(interviewer_parts)
    interviewee_text = " ".join(interviewee_parts)
    
    return interviewer_text, interviewee_text

def process_transcripts(df):
    """
    Process all transcripts in the DataFrame.
    
    Args:
        df (pandas.DataFrame): DataFrame with transcript data
        
    Returns:
        pandas.DataFrame: DataFrame with separated interviewer and interviewee text
    """
    # Create new columns for the separated text
    df['interviewer_text'] = ""
    df['interviewee_text'] = ""
    
    # Process each transcript
    for idx, row in df.iterrows():
        interviewer_text, interviewee_text = parse_transcript(row['transcript'])
        df.at[idx, 'interviewer_text'] = interviewer_text
        df.at[idx, 'interviewee_text'] = interviewee_text
    
    return df

def create_count_vectors(processed_df, text_column='interviewee_text'):
    """
    Create count vectors using sklearn's CountVectorizer.
    
    Args:
        processed_df (pandas.DataFrame): DataFrame with processed transcripts
        text_column (str): Column name containing text to analyze
        
    Returns:
        tuple: (CountVectorizer, feature matrix, feature names)
    """
    # Initialize the CountVectorizer
    vectorizer = CountVectorizer(
        stop_words='english',  # Remove common English stop words
        min_df=2,              # Ignore terms that appear in less than 2 documents
        max_df=0.95            # Ignore terms that appear in more than 95% of documents
    )
    
    # Create the feature matrix
    X = vectorizer.fit_transform(processed_df[text_column])
    
    # Get feature names
    feature_names = vectorizer.get_feature_names_out()
    
    return vectorizer, X, feature_names

def main(csv_path):
    """
    Main function to process interview transcripts.
    
    Args:
        csv_path (str): Path to the CSV file
    """
    # Load the transcripts
    print(f"Loading transcripts from {csv_path}...")
    transcripts_df = load_transcripts(csv_path)
    print(f"Loaded {len(transcripts_df)} transcripts.")
    
    # Process the transcripts
    print("Processing transcripts...")
    processed_df = process_transcripts(transcripts_df)
    
    # Show sample of processed data
    print("\nSample of processed data:")
    print(processed_df[['id', 'interviewer_text', 'interviewee_text']].head())
    
    # Create count vectors for interviewee text
    print("\nCreating count vectors for interviewee text...")
    vectorizer, X, feature_names = create_count_vectors(processed_df)
    
    print(f"Created matrix with shape: {X.shape}")
    print(f"Number of features: {len(feature_names)}")
    
    # Show top features (for example purposes)
    print("\nSample features:")
    print(feature_names[:10])
    
    # You can do the same for interviewer text if needed
    print("\nCreating count vectors for interviewer text...")
    interviewer_vectorizer, interviewer_X, interviewer_features = create_count_vectors(
        processed_df, text_column='interviewer_text'
    )
    
    print(f"Created matrix with shape: {interviewer_X.shape}")
    
    # Save the processed data if needed
    processed_df.to_csv('processed_transcripts.csv', index=False)
    print("\nSaved processed transcripts to 'processed_transcripts.csv'")
    
    return processed_df, vectorizer, X, feature_names



def display_count_vectors(X, feature_names, ids=None, n_samples=3, n_features=20):
    """
    Display and interpret count vectors for a sample of documents.
    
    Args:
        X (csr_matrix): The sparse matrix of count vectors
        feature_names (array): Array of feature names (words)
        ids (array): Optional array of document IDs
        n_samples (int): Number of samples to display
        n_features (int): Number of top features to display per sample
    """
    # Convert sparse matrix to array for easier manipulation
    if isinstance(X, csr_matrix):
        X_array = X.toarray()
    else:
        X_array = X
    
    # Determine how many samples to show (min of n_samples and available samples)
    n_samples = min(n_samples, X_array.shape[0])
    
    print(f"\n{'='*80}")
    print(f"INTERPRETING COUNT VECTORS")
    print(f"{'='*80}")
    print(f"Total documents: {X_array.shape[0]}")
    print(f"Vocabulary size: {X_array.shape[1]} words")
    
    # Get some basic statistics
    doc_lengths = X_array.sum(axis=1)
    print(f"\nDocument word counts:")
    print(f"  Min: {doc_lengths.min():.0f} words")
    print(f"  Max: {doc_lengths.max():.0f} words")
    print(f"  Avg: {doc_lengths.mean():.1f} words")
    
    # Get the most common words overall
    word_counts = X_array.sum(axis=0)
    top_indices = word_counts.argsort()[-20:][::-1]
    
    print(f"\nTop 20 words across all documents:")
    for idx, count in zip(top_indices, word_counts[top_indices]):
        print(f"  {feature_names[idx]}: {count:.0f} occurrences")
    
    # Display sample documents and their top words
    print(f"\n{'-'*80}")
    print(f"SAMPLE DOCUMENT ANALYSIS")
    print(f"{'-'*80}")
    
    # Sample indices to display
    sample_indices = np.random.choice(X_array.shape[0], n_samples, replace=False)
    
    for i, idx in enumerate(sample_indices):
        # Get document info
        doc_id = ids[idx] if ids is not None else f"Document #{idx}"
        
        print(f"\nSample {i+1}: {doc_id}")
        print(f"Total words: {doc_lengths[idx]:.0f}")
        
        # Get the top words for this document
        doc_vec = X_array[idx]
        top_word_indices = doc_vec.argsort()[-n_features:][::-1]
        
        print(f"Top {n_features} words (with counts):")
        for word_idx in top_word_indices:
            if doc_vec[word_idx] > 0:  # Only show words that appear
                print(f"  {feature_names[word_idx]}: {doc_vec[word_idx]:.0f}")
        
        # Create a basic fingerprint of the document
        print(f"Document fingerprint: ", end="")
        fingerprint_indices = doc_vec.argsort()[-5:][::-1]
        print(" + ".join([f"{doc_vec[idx]:.0f}×{feature_names[idx]}" 
                          for idx in fingerprint_indices 
                          if doc_vec[idx] > 0]))
    
    print(f"\n{'='*80}")
    
    # Return a DataFrame with documents and their word counts for further analysis
    word_count_df = pd.DataFrame(X_array, columns=feature_names)
    if ids is not None:
        word_count_df['document_id'] = ids
    
    return word_count_df

# Example usage (to add to your existing code):
def analyze_count_vectors(processed_df, vectorizer, X, feature_names):
    """
    Analyze and display count vectors.
    """
    # Display and interpret the count vectors
    word_count_df = display_count_vectors(
        X, 
        feature_names, 
        ids=processed_df['id'].values
    )
    
    # You can save this DataFrame for further analysis
    word_count_df.to_csv('word_counts.csv')
    
    return word_count_df

if __name__ == "__main__":
    # Path to your CSV file
    csv_file_path = "MITInterview/MITInterview/transcripts.csv"
    
    # Run the main function
    processed_df, vectorizer, X, feature_names = main(csv_file_path)
    # Add this to your main function:
    word_count_df = analyze_count_vectors(processed_df, vectorizer, X, feature_names)
    # Now you can use X and vectorizer for further analysis with sklearn
    # For example:
    # from sklearn.cluster import KMeans
    # kmeans = KMeans(n_clusters=5).fit(X)
    # processed_df['cluster'] = kmeans.labels_