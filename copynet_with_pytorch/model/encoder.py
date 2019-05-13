import torch
from torch import nn
from torch.autograd import Variable


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(input_size, self.embedding_size)
        self.embedding.weight.data.normal_(0, 1 / self.embedding_size ** 0.5)
        self.gru = nn.GRU(embedding_size, hidden_size, bidirectional=True, batch_first=True)

    def forward(self, input, hidden, lengths):
        # input batch must be sorted by sequence length
        # replace OOV words with <UNK> before embedding
        input = input.masked_fill(input > self.embedding.num_embeddings, 3)
        embedded = self.embedding(input)
        packed_embedded = torch.nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True)
        self.gru.flatten_parameters()
        output, hidden = self.gru(packed_embedded, hidden)
        output, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return output, hidden

    def init_hidden(self, batch_size):
        # bidirectional rnn
        hidden = Variable(torch.zeros(2, batch_size, self.hidden_size))
        if next(self.parameters()).is_cuda:
            return hidden.cuda()
        else:
            return hidden
