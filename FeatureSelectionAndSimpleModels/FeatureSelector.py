from typing import Tuple

import pandas as pd
import seaborn
from matplotlib import pyplot
from sklearn.ensemble import RandomForestRegressor


def load_feature_selected_data() -> Tuple[pd.DataFrame, pd.DataFrame]:


    FEATURE_DATA_FILEPATH = "../DataExtraction/features.csv"
    FEATURE_CLASSIFICATIONS_FILEPATH = "../RawProvidedData/MITInterview/scores.csv"

    feature_data = pd.read_csv(FEATURE_DATA_FILEPATH)
    feature_classifications = pd.read_csv(FEATURE_CLASSIFICATIONS_FILEPATH)

    feature_data, feature_classifications = do_feature_selection(feature_data=feature_data, feature_classifications=feature_classifications)
    return feature_data, feature_classifications

def do_feature_selection(feature_data: pd.DataFrame, feature_classifications: pd.DataFrame, do_visualization=False) -> Tuple[pd.DataFrame, pd.DataFrame]:

    feature_data, feature_classifications = standardize_data(feature_data, feature_classifications, do_visualization)

    if do_visualization:
        do_model_based_feature_selection(feature_data, feature_classifications)

    feature_data = feature_data.drop(columns=[
        "interviewer_length",
        "interviewee_length",
        "cluster",
        "total_word_count",
        "similarity",
        "minimum_sentence_sentiment",
        "maximum_sentence_sentiment",
        "word_length_2", "word_length_3",
        "word_length_4",
        "word_length_5",
        "speaker_balance"
    ])

    if do_visualization:
        visualize_correlation_matrix(feature_data)

    return feature_data, feature_classifications

def standardize_data(feature_data: pd.DataFrame, feature_classifications: pd.DataFrame, do_visualizations=False) -> Tuple[pd.DataFrame, pd.DataFrame]:

    feature_classifications = feature_classifications.rename(columns={'Participant': 'id'})

    feature_data = cleanse_id_column(feature_data)
    feature_classifications = cleanse_id_column(feature_classifications)

    merged_data = pd.merge(feature_data, feature_classifications, on='id')
    merged_data = merged_data.drop(columns=['Excited', 'id'])


    if do_visualizations:
        visualize_correlation_matrix(merged_data)

    feature_data = merged_data.drop(columns=['Overall'])
    feature_classifications = merged_data['Overall'].to_frame()

    return feature_data, feature_classifications


def cleanse_id_column(data_frame: pd.DataFrame) -> pd.DataFrame:
    data_frame['id'] = data_frame['id'].str.replace(r'\D', '', regex=True).astype(int)
    sorted_data_frame = data_frame.sort_values(by='id')

    return sorted_data_frame


def visualize_correlation_matrix(feature_data: pd.DataFrame) -> None:
    corr = feature_data.corr()

    pyplot.figure(figsize=(15, 15))
    seaborn.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    pyplot.title('Correlation Matrix')
    pyplot.tight_layout()
    pyplot.show()


def do_model_based_feature_selection(feature_data: pd.DataFrame, feature_classifications: pd.DataFrame) -> pd.DataFrame:

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(feature_data, feature_classifications)

    importances = model.feature_importances_

    importance_data = pd.DataFrame({
        'Feature': feature_data.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    show_importance_data_plot(importance_data)


def show_importance_data_plot(importance_data: pd.DataFrame) -> None:
    pyplot.figure(figsize=(15, 10))
    seaborn.barplot(data=importance_data, x='Importance', y='Feature')
    pyplot.title('Feature Importances')
    pyplot.tight_layout()
    pyplot.show()


if __name__ == "__main__":

    FEATURE_DATA_FILEPATH = "../DataExtraction/features.csv"
    FEATURE_CLASSIFICATIONS_FILEPATH = "../RawProvidedData/MITInterview/scores.csv"

    feature_data = pd.read_csv(FEATURE_DATA_FILEPATH)
    feature_classifications = pd.read_csv(FEATURE_CLASSIFICATIONS_FILEPATH)

    do_feature_selection(feature_data=feature_data, feature_classifications=feature_classifications, do_visualization=True)