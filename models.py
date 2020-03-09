
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np


class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()

        self.use_cuda = torch.cuda.is_available()
        self.method = method
        self.hidden_size = hidden_size

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, hidden, targets, mask=None):
        this_batch_size = targets.size(0)
        max_len = targets.size(1)

        # Create variable to store attention energies
        attn_energies = Variable(torch.zeros(this_batch_size, max_len))  # B x S

        if torch.cuda.is_available():
            attn_energies = attn_energies.cuda()

        # For each batch of encoder outputs
        for b in range(this_batch_size):
            # Calculate energy for each encoder output
            for i in range(max_len):
                attn_energies[b, i] = self.score(hidden[:, b], targets[b, i].unsqueeze(0))

        if mask is not None:
            attn_energies = attn_energies + mask

        # Normalize energies to weights in range 0 to 1, resize to 1 x B x S
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

    def score(self, hidden, target):
        if self.method == 'dot':
            energy = torch.dot(hidden.squeeze(0), target.squeeze(0))
            return energy

        elif self.method == 'general':
            energy = self.attn(target)
            return torch.dot(hidden.squeeze(0), energy.squeeze(0))

        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, target), 1))
            energy = self.v.dot(energy)
            return energy


class BasicRNN(nn.Module):
    def __init__(self, embedding_dim, hidden_size, lang, pretrained_embeddings,
                 num_layers, vocab_size, num_classes, dropout):
        super(BasicRNN, self).__init__()

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        if pretrained_embeddings is not None:
            for i in range(vocab_size):
                word = lang.index2word[i]
                if word in pretrained_embeddings:
                    self.word_embeds.weight[i] = nn.Parameter(torch.FloatTensor(pretrained_embeddings[word]))
            self.word_embeds = nn.Embedding.from_pretrained(self.word_embeds.weight)

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 1
        self.rnn = nn.RNN(input_size=embedding_dim,
                          hidden_size=hidden_size, num_layers=num_layers,
                          batch_first=True, bidirectional=False)
        self.fc1 = nn.Linear(hidden_size * self.num_directions, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(p=dropout)
        # figure this out
        self.use_cuda = torch.cuda.is_available()

    def freeze_layer(self, layer):
        fc = self.fc1
        if layer == "fc2":
            fc = self.fc2
        for param in fc.parameters():
            print(param)
            param.requires_grad = False


    def forward(self, inputs, seq_lengths):
        batch_size = inputs.size(0)

        inputs = self.word_embeds(inputs)

        # Set initial states
        h0 = Variable(torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size))
        if self.use_cuda:
            h0 = h0.cuda()

        # Forward propagate RNN
        outputs, _ = self.rnn(inputs, h0)

        # Decode hidden state of last time step
        outputs = F.relu(self.fc1(outputs[:, -1, :]))

        outputs = self.dropout(outputs)

        outputs = self.fc2(outputs)

        return outputs

    def to_cuda(self, tensor):
        if torch.cuda.is_available():
            return tensor.cuda()
        else:
            return tensor


class AttentionRNN(BasicRNN):
    def __init__(self, embedding_dim, hidden_size, lang, pretrained_embeddings,
                 num_layers, vocab_size, num_classes, dropout):
        super(AttentionRNN, self).__init__(
            embedding_dim, hidden_size, lang, pretrained_embeddings,
            num_layers, vocab_size, num_classes, dropout)
        self.attn = Attn('general', hidden_size)

    def forward(self, inputs, lang, seq_lengths):
        batch_size = inputs.size(0)

        embedded = self.word_embeds(inputs)

        total_length = embedded.size(1)  # get the max sequence length

        # Set initial states
        h0 = Variable(torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size))
        if torch.cuda.is_available():
            h0 = h0.cuda()

        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, seq_lengths, batch_first=True)

        # Forward propagate RNN
        # rnn_outputs, state = self.rnn(embedded, h0)
        rnn_outputs, state = self.rnn(packed, h0)

        rnn_outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(
            rnn_outputs, batch_first=True, total_length=total_length)  # unpack (back to padded)

        encoder_mask = torch.Tensor(np.array(inputs.cpu().data.numpy() == lang.PAD_token,
                                             dtype=float) * (-1e6))  # [b x seq]
        encoder_mask = Variable(self.to_cuda(encoder_mask))

        # use attention to compute soft alignment score corresponding
        # between each of the hidden_state and the last hidden_state of the RNN
        attn_weights = self.attn(state, rnn_outputs, mask=encoder_mask)
        new_state = attn_weights.bmm(rnn_outputs)  # B x 1 x N

        # Decode hidden state of last time step
        # outputs = F.relu(self.fc1(rnn_outputs[:, -1, :]))
        outputs = F.relu(self.fc1(new_state.squeeze(1)))

        outputs = self.dropout(outputs)

        outputs = self.fc2(outputs)

        return outputs


class LSTM(BasicRNN):
    def __init__(self, embedding_dim, hidden_size, lang, pretrained_embeddings,
                 num_layers, vocab_size, num_classes, dropout):
        super(LSTM, self).__init__(embedding_dim, hidden_size, lang, pretrained_embeddings,
                                     num_layers, vocab_size, num_classes, dropout)
        self.rnn = nn.LSTM(input_size=embedding_dim,
                           hidden_size=hidden_size, num_layers=num_layers,
                           batch_first=True, bidirectional=False)

    def forward(self, inputs, seq_lengths):
        batch_size = inputs.size(0)

        inputs = self.word_embeds(inputs)

        # Set initial states
        h0 = Variable(torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size))
        c0 = Variable(torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size))
        if torch.cuda.is_available():
            h0 = h0.cuda()
            c0 = c0.cuda()

        # Forward propagate RNN
        outputs, _ = self.rnn(inputs, (h0, c0))

        # Decode hidden state of last time step
        outputs = F.relu(self.fc1(outputs[:, -1, :]))

        outputs = self.dropout(outputs)

        outputs = self.fc2(outputs)

        return outputs


class GRURNN(BasicRNN):
    def __init__(self, embedding_dim, hidden_size, lang, pretrained_embeddings,
                 num_layers, vocab_size, num_classes, dropout):
        super(GRURNN, self).__init__(embedding_dim, hidden_size, lang, pretrained_embeddings,
                                     num_layers, vocab_size, num_classes, dropout)
        self.rnn = nn.GRU(input_size=embedding_dim,
                          hidden_size=hidden_size, num_layers=num_layers,
                          batch_first=True, bidirectional=False)


class AttentionGRURNN(AttentionRNN):
    def __init__(self, embedding_dim, hidden_size, lang, pretrained_embeddings,
                 num_layers, vocab_size, num_classes, dropout):
        super(AttentionGRURNN, self).__init__(
            embedding_dim, hidden_size, lang, pretrained_embeddings,
            num_layers, vocab_size, num_classes, dropout)

        self.rnn = nn.GRU(input_size=embedding_dim,
                          hidden_size=hidden_size, num_layers=num_layers,
                          batch_first=True, bidirectional=False)


class HighwayNetwork(nn.Module):
    def __init__(self, input_size):
        super(HighwayNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, input_size, bias=True)
        self.fc2 = nn.Linear(input_size, input_size, bias=True)

    def forward(self, x):
        t = F.sigmoid(self.fc1(x))
        return torch.mul(t, F.relu(self.fc2(x))) + torch.mul(1 - t, x)


class CNN(nn.Module):
    def __init__(self, vocab_size, output_size, embedding_dim, lang,
                 pretrained_embeddings, dropout=0.1):
        super(CNN, self).__init__()

        self.use_cuda = torch.cuda.is_available()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.output_size = output_size
        self.dropout = dropout

        print('vocab_size:', vocab_size)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if pretrained_embeddings is not None:
            for i in range(vocab_size):
                word = lang.index2word[i]
                if word in pretrained_embeddings:
                    self.embedding.weight[i] = nn.Parameter(torch.FloatTensor(pretrained_embeddings[word]))
            self.embedding = nn.Embedding.from_pretrained(self.embedding.weight)

        self.conv1 = None
        self.conv2 = None
        self.init_conv1_layer()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1))
        self.init_conv2_layer()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1))

        self.fc1 = None
        self.fc2 = None
        self.init_fc_layers()

        # Highway Networks
        self.batch_norm = nn.BatchNorm1d(num_features=128, affine=False)
        self.highway1 = HighwayNetwork(input_size=128)
        self.highway2 = HighwayNetwork(input_size=128)

    def init_conv1_layer(self):
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=(5, self.embedding_dim), stride=1, padding=2),
            nn.ReLU())

    def init_conv2_layer(self):
        self.conv2 = nn.Sequential(
            nn.Conv2d(5, 20, kernel_size=(5, 3), stride=1),
            nn.ReLU())

    def freeze_conv1_layer(self):
        for param in self.conv1.parameters():
            param.requires_grad = False

    def freeze_conv2_layer(self):
        for param in self.conv2.parameters():
            param.requires_grad = False

    def init_fc_layers(self):
        self.fc1 = nn.Sequential(
            nn.Linear(4160, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        self.fc2 = nn.Linear(256, self.output_size)

    def forward(self, input_seqs):
        x1 = self.embedding(input_seqs)
        x2 = x1.unsqueeze(1)
        x3 = self.conv1(x2)
        x4 = x3.transpose(1, 3)
        x5 = self.maxpool1(x4)
        x6 = self.conv2(x5)
        x7 = x6.transpose(1, 3)
        x8 = self.maxpool2(x7)
        x9 = x8.view(x8.size(0), -1)
        x10 = self.fc1(x9)
        x = self.fc2(x10)

        # print('x1:', x1.size())
        # print('x2:', x2.size())
        # print('x3:', x3.size())
        # print('x4:', x4.size())
        # print('x5:', x5.size())
        # print('x6:', x6.size())
        # print('x7:', x7.size())
        # print('x8:', x8.size())
        # print('x9:', x9.size())
        # print('x10:', x10.size())

        # x = self.batch_norm(x)
        # x = self.highway1(x)
        # x = self.highway2(x)
        return x




