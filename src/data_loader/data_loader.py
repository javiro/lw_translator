import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from src.etl.prepare_data import prepareData


class MTDataloader(object):
    def __init__(self, batch_size, input_lang_code, output_lang_code, max_length, device, dataset):
        self.batch_size = batch_size
        self.input_lang_code = input_lang_code
        self.output_lang_code = output_lang_code
        self.max_length = max_length
        self.device = device
        self.dataset = dataset
        self.SOS_token = 0
        self.EOS_token = 1

    @staticmethod
    def indexesFromSentence(lang, sentence):
        return [lang.word2index[word] for word in sentence.split(' ')]

    def tensorFromSentence(self, lang, sentence):
        indexes = self.indexesFromSentence(lang, sentence)
        indexes.append(self.EOS_token)
        return torch.tensor(indexes, dtype=torch.long, device=self.device).view(1, -1)

    def tensorsFromPair(self, input_lang, output_lang, pair):
        input_tensor = self.tensorFromSentence(input_lang, pair[0])
        target_tensor = self.tensorFromSentence(output_lang, pair[1])
        return (input_tensor, target_tensor)

    def get_dataloader(self):
        input_lang, output_lang, pairs = prepareData(
            self.input_lang_code, self.output_lang_code, self.max_length, self.dataset, True)

        n = len(pairs)
        input_ids = np.zeros((n, self.max_length), dtype=np.int32)
        target_ids = np.zeros((n, self.max_length), dtype=np.int32)

        for idx, (inp, tgt) in enumerate(pairs):
            inp_ids = self.indexesFromSentence(input_lang, inp)
            tgt_ids = self.indexesFromSentence(output_lang, tgt)
            inp_ids.append(self.EOS_token)
            tgt_ids.append(self.EOS_token)
            input_ids[idx, :len(inp_ids)] = inp_ids
            target_ids[idx, :len(tgt_ids)] = tgt_ids

        train_data = TensorDataset(torch.LongTensor(input_ids).to(self.device),
                                torch.LongTensor(target_ids).to(self.device))

        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.batch_size)
        return input_lang, output_lang, train_dataloader
