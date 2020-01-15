import os
import re

def read_data(data_path):
    """
    Reads Data from directory and converts it to lists

    @param data_path : (str) Path to the base data directory
    @returns train_reviews : (List[List[str]]) List of all reviews in train dir - lower case
    @returns test_reviews : (List[List[str]]) List of all reviews in test dir - lower case
    @returns train_labels : (List[int]) List of all labels in train dir - encoded
    @returns test_labels : (List[int]) List of all labels in test dir - encoded
    """

    train_reviews, test_reviews = [], []
    train_labels, test_labels = [], []
    for category in ["train", "test"]:
        folder = os.path.join(data_path, category)
        for sentiment in ["pos", "neg"]:
            path = os.path.join(folder, sentiment)
            for file_name in os.listdir(path):
                file = open(os.path.join(path, file_name), encoding='utf-8')
                sent = file.read().lower().strip()
                sent = normalize_sentence(sent)
                sent = sent.split()
                if category == "train":
                    train_reviews.append(sent)
                    if sentiment == "pos":
                        train_labels.append(1)
                    else:
                        train_labels.append(0)
                else:
                    test_reviews.append(sent)
                    if sentiment == "pos":
                        test_labels.append(1)
                    else:
                        test_labels.append(0)
                file.close()
    return train_reviews, test_reviews, train_labels, test_labels

def pad_sentences(word_list, pad_token='<pad>', seq_len=500):
    """
    Pad or Truncate sentences to seq_len
    @param word_list : (List[List[str]]) List of words for each sentence
    @param pad_token : (str) Token to be used for padding
    @param seq_len : (int) Length of padded sequences
    @returns padded_sents : (List[List[str]]) List of padded sentences represented as word lists
    """
    padded_sents = []
    for sent in word_list:
        if(len(sent) > seq_len):
            sent = sent[:seq_len]
        sent = sent + [pad_token] * (seq_len - len(sent))
        padded_sents.append(sent)
    return padded_sents

def normalize_sentence(sent):
    """
    Normalize a sentence - removing all unnecessary characters
    @param sent : (str) Unnormalized sentence
    @returns norm_sent : (str)  Normalized sentence
    """
    sent = sent.replace('<br />', '')
    sent = re.sub(r'(\W)(?=\1)', '', sent)
    sent = re.sub(r"([.!?])", r" \1", sent)
    norm_sent = re.sub(r"[^a-zA-Z.!?]+", r" ", sent)
    return norm_sent