import torch

from src.data_loader.data_loader import MTDataloader
from src.models.encoder import EncoderRNN
from src.models.attn_decoder_rnn import AttnDecoderRNN
from src.trainer.trainer import MTModelTrainer
import itertools

hidden_size = 128
batch_size = 32
device = "cpu"
max_length = 10

language_combinations = list(itertools.combinations(["sv", "da", "nb", "en"], 2))

for input_lang_code, output_lang_code in [*language_combinations, *[(l2, l1) for l1, l2 in language_combinations]]:

    print("#"*100)    
    print(f"Training {input_lang_code} to {output_lang_code}")
    
    mt_dl = MTDataloader(batch_size, input_lang_code, output_lang_code, max_length, device, "train_small")

    input_lang, output_lang, train_dataloader = mt_dl.get_dataloader()

    encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    decoder = AttnDecoderRNN(hidden_size, output_lang.n_words).to(device)

    model_trainer = MTModelTrainer(train_dataloader, encoder, decoder, device)
    model_trainer.train(80, print_every=5, plot_every=5)

    model_trainer.save_model(input_lang_code, output_lang_code)
