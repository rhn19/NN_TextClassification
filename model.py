import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
	def __init__(self,vocab_size, embedding_dim, hidden_dim, num_outputs, num_layers, drop_prob=0.5, bidirectional=True):
		super(Classifier, self).__init__()
		self.hidden_dim = hidden_dim
		self.num_layers = num_layers
		if bidirectional:
			self.num_dir = 2
		else:
			self.num_dir = 1

		self.embed = nn.Embedding(vocab_size, embedding_dim)
		self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, dropout=drop_prob, batch_first=True, bidirectional=bidirectional)
		self.drop = nn.Dropout(p=0.3)
		self.fc = nn.Linear(hidden_dim*self.num_dir, num_outputs)
		self.sig = nn.Sigmoid()

	def forward(self,inputs, hidden):
		batch_size = inputs.size(0)		# (batch, seq_len)
		embed_inp = self.embed(inputs)	# (batch, seq_len, embed_dim)
		rnn , hidden = self.lstm(embed_inp, hidden)
		h = hidden[0]	# (layers * dirn, batch, hidden_size)
		drop = self.drop(torch.cat((h[-2,:,:], h[-1,:,:]), dim = 1))	# (batch, hidden_size * dirn) Hidden states from the last layer
		fc = self.fc(drop)	# (batch, num_outputs)
		out = self.sig(fc)
		
		return out, hidden

	def init_hidden(self, batch_size, device):
		weight = next(self.parameters()).data
		hidden = (weight.new(self.num_layers*self.num_dir, batch_size, self.hidden_dim).zero_().to(device),
					weight.new(self.num_layers*self.num_dir, batch_size, self.hidden_dim).zero_().to(device))
		return hidden