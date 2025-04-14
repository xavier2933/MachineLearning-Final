import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix
import numpy as np


def load_transcripts(csv_path) -> pd.DataFrame:

    print(f"Loading transcripts from {csv_path}...")
    df = pd.read_csv(csv_path)

    def take_only_two_columns(df: pd.DataFrame) -> pd.DataFrame:

        if len(df.columns) >= 2:
            df = df.iloc[:, :2]
            df.columns = ['id', 'transcript']

        return df

    df = take_only_two_columns(df)

    print(f"Loaded {len(df)} transcripts.")
    return df

def parse_transcript(df: pd.DataFrame) -> pd.DataFrame:

    print("Parsing transcripts...")

    SPLIT_DELIMITER = '|'

    # Create new columns for the separated text
    df['interviewer_text'] = ""
    df['interviewee_text'] = ""


    for transcript_index, row in df.iterrows():

        transcript = row['transcript']
        parts = transcript.split(SPLIT_DELIMITER)

        interviewer_parts = []
        interviewee_parts = []

        for part in parts:
            if part.strip().startswith("Interviewer:"):
                interviewer_parts.append(part.replace("Interviewer:", "").strip())
            elif part.strip().startswith("Interviewee:"):
                interviewee_parts.append(part.replace("Interviewee:", "").strip())

        interviewer_text = " ".join(interviewer_parts)
        interviewee_text = " ".join(interviewee_parts)

        df.at[transcript_index, 'interviewer_text'] = interviewer_text
        df.at[transcript_index, 'interviewee_text'] = interviewee_text

    return df

def create_count_vectors(parsed_df, text_column):

    IGNORE_WORDS_IN_LESS_THAN_N_DOCUMENTS = 2
    IGNORE_WORDS_IN_MORE_THAN_P_PERCENT_OF_DOCUMENTS = .95

    vectorizer = CountVectorizer(
        stop_words='english',  # Remove common English stop words
        min_df=IGNORE_WORDS_IN_LESS_THAN_N_DOCUMENTS,
        max_df=IGNORE_WORDS_IN_MORE_THAN_P_PERCENT_OF_DOCUMENTS
    )

    feature_matrix = vectorizer.fit_transform(parsed_df[text_column])
    feature_names = vectorizer.get_feature_names_out()

    print(f"Created matrix with shape: {feature_matrix.shape}")
    print(f"Number of features: {len(feature_names)}")

    def display_top_10_features():
        print("\nSample features:")
        print(feature_names[:10])

    display_top_10_features()
    
    return feature_matrix, feature_names


def calculate_word_counts(feature_matrix, feature_names, parsed_df, n_samples=3, n_features=20):

    ids = parsed_df['id'].values

    # Convert sparse matrix to array for easier manipulation
    if isinstance(feature_matrix, csr_matrix):
        X_array = feature_matrix.toarray()
    else:
        X_array = feature_matrix
    
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

def extract_word_count_data() -> pd.DataFrame:
    CSV_FILE_PATH = "../MITInterview/MITInterview/transcripts.csv"

    transcripts_df = load_transcripts(CSV_FILE_PATH)
    parsed_df = parse_transcript(transcripts_df)



    print("\nCreating count vectors for interviewee text...")
    interviewee_features, interviewee_feature_names = create_count_vectors(
        parsed_df, text_column='interviewee_text'
    )

    print("\nCreating count vectors for interviewer text...")
    interviewer_features, interviewer_feature_names = create_count_vectors(
        parsed_df, text_column='interviewer_text'
    )

    word_count_df = calculate_word_counts(parsed_df=parsed_df, feature_matrix=interviewee_features, feature_names=interviewee_feature_names)
    word_count_df.to_csv('word_counts.csv')

    return word_count_df
