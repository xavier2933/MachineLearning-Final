import pandas as pd

def load_and_parse_transcripts(csv_path) -> pd.DataFrame:
    """
    Loads the CSV, keeps the first two columns (id, transcript),
    and parses out text for Interviewer vs Interviewee.
    """
    df = pd.read_csv(csv_path)

    # Keep only the first two columns (ID + Transcript)
    df = df.iloc[:, :2]
    df.columns = ['id', 'transcript']

    # Create two new columns for the separated text
    df['interviewer_text'] = ""
    df['interviewee_text'] = ""

    for i, row in df.iterrows():
        parts = row['transcript'].split('|')
        
        interviewer_parts = []
        interviewee_parts = []
        
        for part in parts:
            trimmed = part.strip()
            if trimmed.startswith("Interviewer:"):
                text_portion = trimmed.replace("Interviewer:", "").strip()
                interviewer_parts.append(text_portion)
            elif trimmed.startswith("Interviewee:"):
                text_portion = trimmed.replace("Interviewee:", "").strip()
                interviewee_parts.append(text_portion)

        # Join all lines from each speaker into one string
        df.at[i, 'interviewer_text'] = " ".join(interviewer_parts)
        df.at[i, 'interviewee_text'] = " ".join(interviewee_parts)

    return df

def compute_speaker_balance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame with one column, 'speaker_balance',
    which is (# interviewee words) / (# total words).
    """
    results = []

    for _, row in df.iterrows():
        interviewer_text = str(row['interviewer_text']).strip()
        interviewee_text = str(row['interviewee_text']).strip()

        # Count words
        interviewer_word_count = len(interviewer_text.split()) if interviewer_text else 0
        interviewee_word_count = len(interviewee_text.split()) if interviewee_text else 0

        total_words = interviewer_word_count + interviewee_word_count

        if total_words > 0:
            balance = interviewee_word_count / total_words
        else:
            balance = 0

        results.append({"speaker_balance": balance})

    return pd.DataFrame(results)