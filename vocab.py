from utils import read_data
from collections import Counter
from itertools import chain
import sys

class Vocab:
    def __init__(self, pad_token='<pad>'):
        """
        Builds a word - integer mapping dictionary
        @param pad_token : (int) Token used for padding sentences [Default : 0]
        """
        self.word2idx = {}
        self.pad_token = pad_token

    def build(self, word_list, vocab_size=None):
        """
        Build vocab on train & test set (change this later to just the train set with a <unk> token for test set)
        @param word_list : (List[List[str]]) List of words for every sentence
        @param vocab_size : (int) Max vocab size [Default : Max Possible]
        """
        word_count = Counter(chain(*word_list))
        if vocab_size is None:
            vocab_size = len(word_count)
        else:
            vocab_size -= 1  #consider pad_token
        sorted_count = word_count.most_common(vocab_size)
        self.word2idx = {w:i+1 for i, (w, _) in enumerate(sorted_count)}
        self.word2idx[self.pad_token] = 0

    def load_prebuilt(self, path):
        """
        Build vocab from pre-built list
        @param path : (str) Path to the vocab file
        """
        with open(path, encoding='utf-8') as f:
            vocab_list = f.readlines()
            vocab_list = [item.strip() for item in vocab_list]
        self.word2idx = {w:i+1 for i, w in enumerate(vocab_list)}
        self.word2idx[self.pad_token] = 0

if __name__ == "__main__":
    """
    Mode 1: Build Custom Dictionary
    Mode 2: Load Pre-built Dictioanry
    """
    base_dir = "./data"
    vocab_file = "./data/imdb.vocab"
    vocab = Vocab()
    mode = sys.argv[1]
    if mode == "1":
        train_reviews, test_reviews, _, _ = read_data(base_dir)
        vocab.build(train_reviews + test_reviews)
    elif mode == "2":
        vocab.load_prebuilt(vocab_file)
    else:
        print("Invalid Mode!")
