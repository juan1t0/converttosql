import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, device):
        super(EncoderRNN, self).__init__()
        
        self.HiddenSize = hidden_size
        self.Device = device
        self.Embedding = nn.Embedding(input_size, hidden_size)
        self.GRU = nn.GRU(hidden_size, hidden_size)
    
    def forward(self, input, hidden):
        embedded = self.Embedding(input).view(1,1,-1)
        output = embedded
        output, hidden = self.GRU(output, hidden)
        return output, hidden
    
    def init_hidden(self):
        return torch.zeros(1, 1, self.HiddenSize, device=self.Device)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, device, dropout_p=0.1, max_length=20):
        super(DecoderRNN, self).__init__()
        self.HiddenSize = hidden_size
        self.OutputSize = output_size
        self.DropoutP = dropout_p
        self.MaxLength = max_length
        self.Device = device

        self.Embedding = nn.Embedding(self.OutputSize, self.HiddenSize)
        self.Attn = nn.Linear(self.HiddenSize*2, self.MaxLength)
        self.AttnCombine = nn.Linear(self.HiddenSize*2, self.HiddenSize)
        self.Dropout = nn.Dropout(self.DropoutP)
        self.GRU = nn.GRU(self.HiddenSize, self.HiddenSize)
        self.Out = nn.Linear(self.HiddenSize, self.OutputSize)
    
    def forward(self, input, hidden, encoder_outputs):
        embedded = self.Embedding(input).view(1,1,-1)
        embedded = self.Dropout(embedded)

        attn_weights = F.softmax(self.Attn(torch.cat((embedded[0], hidden[0]),1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.AttnCombine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.GRU(output, hidden)
        
        output = F.log_softmax(self.Out(output[0]), dim=1)
        return output, hidden, attn_weights

    def init_hidden(self):
        return torch.zeros(1, 1, self.HiddenSize, device=self.Device)


def indexes_from_sentence(corpus, sentence):
    return [corpus.Word2Index[word] for word in sentence.split()]

def tensor_from_sentence(corpus, sentence, device, EOS_token):
    indexes = indexes_from_sentence(corpus, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1,1)

def tensor_from_pair(pair, input_corpus, output_corpus, device, EOS_token):
    input_tensor = tensor_from_sentence(input_corpus, pair[0], device, EOS_token)
    output_tensor = tensor_from_sentence(output_corpus, pair[1], device, EOS_token)
    return (input_tensor, output_tensor)