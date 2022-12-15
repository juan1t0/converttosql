import time
import random

import torch
from torch import optim
import torch.nn as nn

from .seq2seq import tensor_from_pair, tensor_from_sentence
from .utils import time_since, SOS_token, EOS_token


class Processor():
    def __init__(self, encoder, decoder, pairs, device, maxlen=20, teacher_forcing_ratio=0.5):
        self.Encoder = encoder
        self.Decoder = decoder
        self.EncoderOptimizer = None
        self.DecoderOptimizer = None
        self.Criterion = None
        self.MaxLength = maxlen
        self.TeacherForcingRatio = teacher_forcing_ratio
        self.Pairs = pairs
        self.Device = device
    
    def train_step(self, input_tensor, output_tensor):
        encoder_hidden = self.Encoder.init_hidden()

        self.EncoderOptimizer.zero_grad()
        self.DecoderOptimizer.zero_grad()

        input_len = input_tensor.size(0)
        target_len = output_tensor.size(0)

        encoder_outputs = torch.zeros(self.MaxLength, self.Encoder.HiddenSize, device=self.Device)

        loss = 0

        for ei in range(input_len):
            encoder_output, encoder_hidden = self.Encoder(input_tensor[ei],encoder_hidden)
            encoder_outputs[ei] = encoder_output[0,0]
        
        decoder_input = torch.tensor([[SOS_token]], device=self.Device)
        decoder_hidden = encoder_hidden
        use_teacher_forcing = True if random.random() < self.TeacherForcingRatio else False

        for di in range(target_len):
            decoder_output, decoder_hidden, decoder_attention = self.Decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += self.Criterion(decoder_output, output_tensor[di])
            if use_teacher_forcing:
                decoder_input = output_tensor[di]
            else:
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()
                if decoder_input.item() == EOS_token:
                    break
        
        loss.backward()
        self.EncoderOptimizer.step()
        self.DecoderOptimizer.step()

        return loss.item() / target_len

    def train(self, n_iters,  input_corpus, output_corpus, print_every=1000, plot_every=1000, learnig_rate=0.01, opt=None, floss=None):
        start = time.time()
        plot_losses = []
        print_loss_total = 0
        plot_loss_total = 0
        training_pairs = [tensor_from_pair(random.choice(self.Pairs), input_corpus, output_corpus, self.Device, EOS_token) for i in range(n_iters)]
        self.Encoder.train()
        self.Decoder.train()
        if opt:
            self.EncoderOptimizer = opt
            self.DecoderOptimizer = opt
        else:
            self.EncoderOptimizer = optim.SGD
            self.DecoderOptimizer = optim.SGD
        if floss:
            self.Criterion = floss
        else:
            self.Criterion = nn.NLLLoss
        self.EncoderOptimizer = self.EncoderOptimizer(self.Encoder.parameters(), lr=learnig_rate)
        self.DecoderOptimizer = self.DecoderOptimizer(self.Decoder.parameters(), lr=learnig_rate)
        self.Criterion = self.Criterion()

        for iter in range(1, n_iters+1):
            training_pair = training_pairs[iter-1]
            inputTensor = training_pair[0]
            outputTensor = training_pair[1]

            loss = self.train_step(inputTensor, outputTensor)
            print_loss_total += loss
            plot_loss_total += loss

            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f'%(time_since(start, iter/n_iters),iter, iter/n_iters *100, print_loss_avg))
            if iter % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_loss_total = 0
                plot_losses.append(plot_loss_avg)
            
        return plot_losses
    
    def evaluate(self, sentence, input_corpus, output_corpus):
        #with torch.no_grad:
        self.Encoder.eval()
        self.Decoder.eval()
        input_tensor = tensor_from_sentence(input_corpus, sentence, self.Device, EOS_token)
        input_len = input_tensor.size()[0]
        encoder_hidden = self.Encoder.init_hidden()
        encoder_outputs = torch.zeros(self.MaxLength, self.Encoder.HiddenSize, device=self.Device)

        for ei in range(input_len):
            encoder_output, encoder_hidden = self.Encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] += encoder_output[0,0]
        
        decoder_input = torch.tensor([[SOS_token]], device=self.Device)
        decoder_hidden = encoder_hidden
        decoded_words = []
        decoder_attentions = torch.zeros(self.MaxLength, self.MaxLength)

        for di in range(self.MaxLength):
            decoder_output, decoder_hidden, decoder_attention = self.Decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_corpus.Index2Word[topi.item()])
            
            decoder_input = topi.squeeze().detach()
        
        return decoded_words, decoder_attentions[:di+1]
    
    def evaluateRandomly(self, input_corpus, output_corpus, n=5):
        for i in range(n):
            pair = random.choice(self.Pairs)
            print('IN:', pair[0])
            print('TG:', pair[1])
            output_words,_ = self.evaluate(pair[0], input_corpus, output_corpus)
            output_sentence = ' '.join(output_words)
            print('PD:', output_sentence)
            print('')