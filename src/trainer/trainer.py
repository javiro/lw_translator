import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math

from torch import optim


class MTModelTrainer(object):
    def __init__(self, dataloader, encoder, decoder, device):
        self.dataloader = dataloader
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    @staticmethod
    def asMinutes(s):
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    def timeSince(self, since, percent):
        now = time.time()
        s = now - since
        es = s / (percent)
        rs = es - s
        return '%s (- %s)' % (self.asMinutes(s), self.asMinutes(rs))

    def train_epoch(self, encoder_optimizer, decoder_optimizer, criterion):

        total_loss = 0
        for data in self.dataloader:
            input_tensor, target_tensor = data

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            encoder_outputs, encoder_hidden = self.encoder(input_tensor)
            decoder_outputs, _, _ = self.decoder(encoder_outputs, encoder_hidden, target_tensor)

            loss = criterion(
                decoder_outputs.view(-1, decoder_outputs.size(-1)),
                target_tensor.view(-1)
            )
            loss.backward()

            encoder_optimizer.step()
            decoder_optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.dataloader)


    def train(self, n_epochs, learning_rate=0.001, print_every=100, plot_every=100):
        start = time.time()
        plot_losses = []
        print_loss_total = 0  # Reset every print_every
        plot_loss_total = 0  # Reset every plot_every

        encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=learning_rate)
        decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=learning_rate)
        criterion = nn.NLLLoss()

        for epoch in range(1, n_epochs + 1):
            loss = self.train_epoch(encoder_optimizer, decoder_optimizer, criterion)
            print_loss_total += loss
            plot_loss_total += loss

            if epoch % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (self.timeSince(start, epoch / n_epochs),
                                            epoch, epoch / n_epochs * 100, print_loss_avg))

            if epoch % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

    def save_model(self, input_lang_code, output_lang_code):
        torch.save(self.encoder.state_dict(), f'trained_models/encoder_{input_lang_code}_{output_lang_code}.pth')
        torch.save(self.decoder.state_dict(), f'trained_models/decoder_{input_lang_code}_{output_lang_code}.pth')
