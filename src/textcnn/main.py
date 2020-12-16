
import numpy as np
import os
import time
import datetime
from textcnn import process_data
from textcnn.text_cnn import TextCNN
import math
from tensorflow.contrib import learn




def load_data(w2v_model, train_data_file, test_sample_percentage=0.1):
    """Loads starter word-vectors and train/dev/test data."""
    # Load the starter word vectors
    print("Loading data...")
    x_text, y = process_data.load_data_and_labels(train_data_file)
    # for x in x_text:
    #     l = len(x.split(" "))
    #     break

    max_document_length = max([len(x.split(" ")) for x in x_text])
    print('len(x) = ', len(x_text), ' ', len(y))
    print(' max_document_length = ', max_document_length)

    x = []
    vocab_size = 0
    if(w2v_model is None):
      vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
      x = np.array(list(vocab_processor.fit_transform(x_text)))
      vocab_size = len(vocab_processor.vocabulary_)

      # out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", str(int(time.time()))))
      vocab_processor.save("vocab.txt")
      print('save vocab.txt')
    else:
      x = process_data.get_text_idx(x_text, w2v_model.vocab_hash, max_document_length)
      vocab_size = len(w2v_model.vocab_hash)
      print('use w2v .bin')

    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    dev_sample_index = -1 * int(test_sample_percentage * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

    return x_train, x_dev, y_train, y_dev, vocab_size

def train():
    w2v_wr = process_data.w2v_wrapper("../../data/text_cnn/vectors.bin")
    train_data_file = '../../data/text_cnn/cutclean_label_corpus10000.txt'
    x_train, x_dev, y_train, y_dev, vocab_size = load_data(w2v_wr.model, train_data_file)
    config = {
        'w2v_model': w2v_wr.model,
        'sequence_length': x_train.shape[1],
        'num_classes': y_train.shape[1],
        'vocab_size': vocab_size,
        'embedding_size': 128,
        'filter_sizes': [2, 3, 4],
        'num_filters': 128,
        'batch_size': 64,
        'dropout_keep_prob': 0.5,
        'l2_reg_lambda': 0
    }
    model = TextCNN(**config)
    model.fit(x_train, y_train)

if __name__ == '__main__':
    train()
