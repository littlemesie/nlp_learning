import os
from kashgari.corpus import DataReader
from kashgari import utils

def load_data(subset_name='train', shuffle=True):
    """
    Load dataset as sequence labeling format, char level tokenized

    Args:
        subset_name: {train, test, valid}
        shuffle: should shuffle or not, default True.

    Returns:
        dataset_features and dataset labels
    """

    if subset_name == 'train':
        file_path = '../../data/ChineseDailyNerCorpus/example.train'
    elif subset_name == 'test':
        file_path = '../../data/ChineseDailyNerCorpus/example.test'
    else:
        file_path = '../../data/ChineseDailyNerCorpus/example.dev'

    x_data, y_data = DataReader.read_conll_format_file(file_path)
    if shuffle:
        x_data, y_data = utils.unison_shuffled_copies(x_data, y_data)

    return x_data, y_data