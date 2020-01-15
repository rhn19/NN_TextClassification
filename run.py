from utils import read_data, pad_sentences
from vocab import Vocab
from model import Classifier
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time

#CONSTANTS
path = "./data"
DEVICE = torch.device("cuda:0")
EMBED_DIM = 400
HIDDEN_DIM = 256
NUM_OUTPUTS = 1
NUM_LAYERS = 2
LEARN_RATE = 1e-3
BATCH_SIZE = 50
epochs = 4
print_every = 100
clip=5

def word_to_int(sents, word2idx):
    """
    Creates integer lists for the corresponding mapping in word2idx
    @param sents : (List[List[str]]) List of padded sentences returned by pad_sentences
    @param word2idx : (Dict) Word Integer mapping dictionary
    @returns int_sents : (List[List[int]]) List of integer mapped sentences
    """
    int_sents = []
    for sent in sents:
        sent = [word2idx[w] for w in sent]
        int_sents.append(sent)
    return int_sents

def train(model, criterion, optimizer, train_loader, val_loader, epochs, print_every=100, clip=5):
    counter = 0
    max_loss = float('inf')

    model.train()
    # train for some number of epochs
    for e in range(epochs):
        start = time.time()
        # initialize hidden state
        h = model.init_hidden(BATCH_SIZE, DEVICE)

        # batch loop
        for inputs, labels in train_loader:
            counter += 1

            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            h = tuple([each.data for each in h])

            # zero accumulated gradients
            model.zero_grad()

            # get the output from the model
            inputs = inputs.type(torch.LongTensor).to(DEVICE)
            output, h = model(inputs, h)

            # calculate the loss and perform backprop
            loss = criterion(output.squeeze(), labels.float())
            loss.backward()
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            # loss stats
            if counter % print_every == 0:
                # Get validation loss
                val_h = model.init_hidden(BATCH_SIZE, DEVICE)
                val_losses = []
                model.eval()
                for inputs, labels in val_loader:

                    # Creating new variables for the hidden state, otherwise
                    # we'd backprop through the entire training history
                    val_h = tuple([each.data for each in val_h])

                    inputs = inputs.type(torch.LongTensor).to(DEVICE)
                    output, val_h = model(inputs, val_h)
                    val_loss = criterion(output.squeeze(), labels.float())

                    val_losses.append(val_loss.item())

                model.train()

                if np.mean(val_losses) < max_loss:
                    print("New Best! Saving Model...")
                    max_loss = np.mean(val_losses)
                    torch.save(model.state_dict(), 'imdb_best.pth')

                print("Epoch: {}/{}...".format(e+1, epochs),
                    "Iters: {}...".format(counter),
                    "Loss: {:.6f}...".format(loss.item()),
                    "Val Loss: {:.6f}...".format(np.mean(val_losses)),
                    "Time: {}".format(time.time() - start))

def test(model, criterion, test_loader):
    test_losses = [] # track loss
    num_correct = 0

    model.load_state_dict(torch.load('imdb_best.pth'))

    # init hidden state
    h = model.init_hidden(BATCH_SIZE, DEVICE)

    model.eval()
    # iterate over test data
    for inputs, labels in test_loader:

        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        h = tuple([each.data for each in h])
        
        # get predicted outputs
        inputs = inputs.type(torch.LongTensor).to(DEVICE)
        output, h = model(inputs, h)
        
        # calculate loss
        test_loss = criterion(output.squeeze(), labels.float())
        test_losses.append(test_loss.item())
        
        # convert output probabilities to predicted class (0 or 1)
        pred = torch.round(output.squeeze())  # rounds to the nearest integer
        
        # compare predictions to true label
        correct_tensor = pred.eq(labels.float().view_as(pred))
        correct = np.squeeze(correct_tensor.cpu().numpy())
        num_correct += np.sum(correct)


    # -- stats! -- ##
    # avg test loss
    print("Test loss: {:.3f}".format(np.mean(test_losses)))

    # accuracy over all test data
    test_acc = num_correct/len(test_loader.dataset)
    print("Test accuracy: {:.3f}".format(test_acc))

if __name__ == "__main__":
    print("Reading data from Corpus...")
    train_reviews, test_reviews, train_labels, test_labels = read_data(path)

    vocab = Vocab()
    print("Building Vocab...")
    vocab.build(train_reviews + test_reviews)

    VOCAB_SIZE = len(vocab.word2idx)

    padded_train = pad_sentences(train_reviews)
    padded_test = pad_sentences(test_reviews)

    int_train = word_to_int(padded_train, vocab.word2idx)
    int_test = word_to_int(padded_test, vocab.word2idx)

    x_train = Variable(torch.LongTensor(int_train).to(DEVICE))
    x_test = Variable(torch.LongTensor(int_test).to(DEVICE))

    y_train = Variable(torch.LongTensor(train_labels).to(DEVICE))
    y_test = Variable(torch.LongTensor(test_labels).to(DEVICE))

    model = Classifier(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_OUTPUTS, NUM_LAYERS)
    print(model)
    model =  model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARN_RATE)
    criterion = nn.BCELoss()
    criterion = criterion.to(DEVICE)

    train_data = TensorDataset(x_train, y_train)
    val_data = TensorDataset(x_test, y_test)
    test_data = TensorDataset(x_test, y_test)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

    print("Starting Training")
    train(model, criterion, optimizer, train_loader, val_loader, epochs, print_every, clip)
    print("Starting Evaluation")
    test(model, criterion, test_loader)