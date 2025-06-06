# DataExtraction/filler_word_counter.py

import pandas as pd
import re

FILLER_PATTERNS = {
    "uh":  r"\buh+\b",      
    "um":  r"\bum+\b",      
    "like":        None,
    "you know":    None,
    "i mean":      None,
    "sort of":     None,
    "kind of":     None,
    "might":       None,
    "maybe":       None,
    "just":        None,
    "really":      None,
    "so":          None,
    "actually":    None,
    "basically":   None
}

def load_and_parse_transcripts(csv_path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Keep only first two columns: ID + transcript
    df = df.iloc[:, :2]
    df.columns = ['id', 'transcript']

    # Add interviewee_text column
    df['interviewee_text'] = ""
    for i, row in df.iterrows():
        parts = row['transcript'].split('|')
        interviewee_parts = [
            part.replace("Interviewee:", "").strip()
            for part in parts if part.strip().startswith("Interviewee:")
        ]
        df.at[i, 'interviewee_text'] = " ".join(interviewee_parts)

    return df

def compute_filler_percent(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame with 'id' and 'filler%' columns.
    'filler%' = total_filler_words / total_words_in_interviewee_text.
    """
    results = []

    for _, row in df.iterrows():
        text = row['interviewee_text'].lower()
        total_words = len(text.split())
        total_fillers = 0

        # Count all filler words
        for filler, pattern in FILLER_PATTERNS.items():
            if pattern is None:
                pattern = rf'\b{re.escape(filler)}\b'
            count = len(re.findall(pattern, text))
            total_fillers += count

        # Compute filler percentage
        if total_words > 0:
            filler_percent = total_fillers / total_words
        else:
            filler_percent = 0

        # Keep the participant's 'id'
        results.append({
            "id": row["id"],
            "filler%": filler_percent
        })

    return pd.DataFrame(results)
