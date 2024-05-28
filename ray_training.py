from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig

import torch
import ray

from src.data_loader.data_loader import MTDataloader
from src.models.encoder import EncoderRNN
from src.models.attn_decoder_rnn import AttnDecoderRNN
from src.trainer.trainer import MTModelTrainer
import itertools
import ray
from ray import train
from ray.train import Checkpoint, RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer


hidden_size = 128
batch_size = 32
device = "cpu"
max_length = 10


def train_func():
    # Your PyTorch training code here.
    input_lang_code, output_lang_code = "en", "da"
    print(f"Training {input_lang_code} to {output_lang_code}")
    
    mt_dl = MTDataloader(batch_size, input_lang_code, output_lang_code, max_length, device, "train_small")

    input_lang, output_lang, train_dataloader = mt_dl.get_dataloader()

    encoder = EncoderRNN(input_lang.n_words, hidden_size)
    decoder = AttnDecoderRNN(hidden_size, output_lang.n_words)

    encoder = ray.train.torch.prepare_model(encoder)
    decoder = ray.train.torch.prepare_model(decoder)

    model_trainer = MTModelTrainer(train_dataloader, encoder, decoder, device)
    model_trainer.train(80, print_every=5, plot_every=5)

    model_trainer.save_model(input_lang_code, output_lang_code)

scaling_config = ScalingConfig(num_workers=2, use_gpu=True)
trainer = TorchTrainer(train_func, scaling_config==ScalingConfig(
    num_workers=2, use_gpu=True))
run_config=RunConfig(storage_path="/mnt/cluster_storage")
result = trainer.fit()