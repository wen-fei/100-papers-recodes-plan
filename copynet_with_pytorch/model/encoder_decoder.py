from torch import nn
from torch.autograd import Variable

from data.dataset import SequencePairDataset
from model.encoder import EncoderRNN
from model.decoder import CopyNetDecoder


class EncoderDecoder(nn.Module):
    def __init__(self, lang, max_length, hidden_size,
                 embedding_size, spDataset: SequencePairDataset, parse_func):
        super(EncoderDecoder, self).__init__()
        self.lang = lang
        self.encoder = EncoderRNN(len(self.lang.tok_to_idx),
                                  hidden_size,
                                  embedding_size)
        decoder_hidden_size = 2 * self.encoder.hidden_size
        self.decoder = CopyNetDecoder(decoder_hidden_size,
                                      embedding_size,
                                      lang,
                                      max_length)
        self.spDataset = spDataset
        self.parse_func = parse_func

    def forward(self, inputs, lengths,
                targets=None, keep_prob=1.0, teacher_forcing=0.0):
        """

        :param inputs:
        :param lengths:
        :param targets:
        :param keep_prob:
        :param teacher_forcing:
        :return:
        """
        batch_size = inputs.data.shape[0]
        hidden = self.encoder.init_hidden(batch_size)
        encoder_outputs, hidden = self.encoder(inputs, hidden, lengths)
        decoder_outputs, sampled_idxs = self.decoder(encoder_outputs,
                                                     inputs,
                                                     hidden,
                                                     targets=targets,
                                                     teacher_forcing=teacher_forcing)
        return decoder_outputs, sampled_idxs

    def get_response(self, input_string):
        """

        :param input_string: input string = sentence
        :param spDataset: SequencePairDataset
        :param parse_func: function : sentence -> word list
        :return: output string = sentence
        """
        input_tokens = ['<SOS>'] + self.parse_func(input_string) + ['<EOS>']
        input_seq = self.spDataset.tokens_to_seq(input_tokens)
        input_variable = Variable(input_seq).view(1, -1)
        if next(self.parameters()).is_cuda:
            input_variable = input_variable.cuda()
        outputs, idxs = self.forward(input_variable, [len(input_seq)])
        idxs = idxs.data.view(-1)
        eos_idx = list(idxs).index(2) if 2 in list(idxs) else len(idxs)
        output_string = self.spDataset.seq_to_string(idxs[:eos_idx + 1], input_tokens=input_tokens)
        return output_string

    def interactive(self, unsmear):
        input_string = ''
        while input_string != 'Quit':
            input_string = input('\ninput=>')
            output_string = self.get_response(input_string)
            if unsmear:
                output_string = output_string.replace('<SOS>', '')
                output_string = output_string.replace('<EOS>', '')
            print('\n=>{}\n'.format(output_string))
