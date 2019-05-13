import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from data.dataset import Language
from .utils import to_one_hot, DecoderBase


class CopyNetDecoder(DecoderBase):
    def __init__(self, hidden_size, embedding_size, lang: Language, max_length):
        """

        :param hidden_size:
        :param embedding_size:
        :param lang:
        :param max_length:
        """
        super(CopyNetDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.lang = lang
        self.max_length = max_length
        self.embedding = nn.Embedding(len(self.lang.tok_to_idx), self.embedding_size, padding_idx=0)
        # note:
        # embedding.weight.data.shape = torch.Size([20, 20])
        self.embedding.weight.data.normal_(0, 1 / self.embedding_size ** 0.5)
        self.embedding.weight.data[0, :] = 0.0

        self.attn_W = nn.Linear(self.hidden_size, self.hidden_size)
        self.copy_W = nn.Linear(self.hidden_size, self.hidden_size)

        # input = (context + selective read size + embedding)
        self.gru = nn.GRU(2 * self.hidden_size + self.embedding.embedding_dim,
                          self.hidden_size, batch_first=True)
        self.out = nn.Linear(self.hidden_size, len(self.lang.tok_to_idx))

    def forward(self, encoder_outputs, inputs, final_encoder_hidden,
                targets=None, keep_prob=1.0, teacher_forcing=0.0):
        """

        :param encoder_outputs:
        :param inputs:
        :param final_encoder_hidden:
        :param targets:
        :param keep_prob:
        :param teacher_forcing:
        :return:
        """
        batch_size = encoder_outputs.data.shape[0]
        seq_length = encoder_outputs.data.shape[1]
        hidden = Variable(torch.zeros(1, batch_size, self.hidden_size))
        if next(self.parameters()).is_cuda:
            hidden = hidden.cuda()

        # every decoder output seq starts with <SOS>
        sos_output = Variable(torch.zeros((batch_size,
                                           self.embedding.num_embeddings + seq_length)))
        sampled_idx = Variable(torch.ones((batch_size, 1)).long())
        if next(self.parameters()).is_cuda:
            sos_output = sos_output.cuda()
            sampled_idx = sampled_idx.cuda()
        # index 1 is the <SOS> token, one-hot encoding
        sos_output[:, 1] = 1.0
        decoder_outputs = [sos_output]
        sampled_idxs = [sampled_idx]
        if keep_prob < 1.0:
            dropout_mask = (Variable(torch.rand(batch_size, 1,
                                                2 * self.hidden_size + self.embedding.embedding_dim))
                            < keep_prob).float()
        else:
            dropout_mask = None
        selective_read = Variable(torch.zeros(batch_size, 1, self.hidden_size))
        one_hot_inpout_seq = to_one_hot(inputs, len(self.lang.tok_to_idx) + seq_length)
        if next(self.parameters()).is_cuda:
            selective_read = selective_read.cuda()
            one_hot_inpout_seq = one_hot_inpout_seq.cuda()
        for step_idx in range(1, self.max_length):
            if (targets is not None) and (teacher_forcing > 0.0) and (step_idx < targets.shape[1]):
                # replace some inputs with the targets (i.e. teacher forcing)
                teacher_forcing_mask = Variable((torch.rand((batch_size, 1)) < teacher_forcing), requires_grad=False)
                if next(self.parameters()).is_cuda:
                    teacher_forcing_mask = teacher_forcing_mask.cuda()
                sampled_idx = sampled_idx.masked_scatter(teacher_forcing_mask, targets[:, step_idx - 1: step_idx])
            sampled_idx, output, hidden, selective_read = self.step(sampled_idx, hidden, encoder_outputs,
                                                                    selective_read, one_hot_inpout_seq,
                                                                    dropout_mask=dropout_mask)
            decoder_outputs.append(output)
            sampled_idxs.append(sampled_idx)
        decoder_outputs = torch.stack(decoder_outputs, dim=1)
        sampled_idxs = torch.stack(sampled_idxs, dim=1)
        return decoder_outputs, sampled_idxs

    def step(self, prev_idx, prev_hidden, encoder_outputs, prev_selective_read,
             one_hot_input_seq, dropout_mask=None):
        """

        :param prev_idx:
        :param prev_hidden:
        :param encoder_outputs:
        :param prev_selective_read:
        :param one_hot_input_seq:
        :param dropout_mask:
        :return:
        """
        batch_size = encoder_outputs.shape[0]
        seq_length = encoder_outputs.shape[1]
        vocab_size = len(self.lang.tok_to_idx)
        # Attention mechanism
        transformed_hidden = self.attn_W(prev_hidden).view(batch_size, self.hidden_size, 1)
        # reduce encoder outputs and hidden to get scores.
        # remove singleton dimension from multiplication.
        attn_scores = torch.bmm(encoder_outputs, transformed_hidden)
        # apply softmax to scores to get normalize weights
        attn_weights = F.softmax(attn_scores, dim=1)
        # [b. 1, hidden] weighted sum of encoder_outputs (i.e. values)
        context = torch.bmm(torch.transpose(attn_weights, 1, 2), encoder_outputs)

        # Call the RNN
        # [b, 1] bools indicating which seqs copied on the previous step
        out_of_vocab_mask = prev_idx > vocab_size
        unks = torch.ones_like(prev_idx).long() * 3
        # replace copied tokens with <UNK> token before embedding
        prev_idx = prev_idx.masked_scatter(out_of_vocab_mask, unks)
        # embed input (i.e. previous output token)
        embedded = self.embedding(prev_idx)
        rnn_input = torch.cat((context, prev_selective_read, embedded), dim=2)
        if dropout_mask is not None:
            if next(self.parameters()).is_cuda:
                dropout_mask = dropout_mask.cuda()
            rnn_input *= dropout_mask
        self.gru.flatten_parameters()
        # state.shape = [b, 1, hidden]
        output, hidden = self.gru(rnn_input, prev_hidden)

        # copy mechanism
        transformed_hidden2 = self.copy_W(output).view(batch_size, self.hidden_size, 1)
        # this is linear. add activation function before multiplying.
        copy_score_seq = torch.bmm(encoder_outputs, transformed_hidden2)
        # [b, vocab_size + seq_length]
        copy_scores = torch.bmm(torch.transpose(copy_score_seq, 1, 2), one_hot_input_seq).squeeze(1)
        # tokens not present in the input sequence
        missing_token_mask = (one_hot_input_seq.sum(dim=1) == 0)
        # <MSK> tokens are not part of any sequence
        missing_token_mask[:, 0] = 1
        copy_scores = copy_scores.masked_fill(missing_token_mask, -1000000.0)

        # generate mechanism
        # [b, vocab_size]
        gen_scores = self.out(output.squeeze(1))
        # penalize <MSK> tokens in generate mode too
        gen_scores[:, 0] = -1000000.0

        # Combine results from copy and generate mechanisms
        combined_scores = torch.cat((gen_scores, copy_scores), dim=1)
        probs = F.softmax(combined_scores, dim=1)
        gen_probs = probs[:, :vocab_size]
        gen_padding = Variable(torch.zeros(batch_size, seq_length))
        if next(self.parameters()).is_cuda:
            gen_padding = gen_padding.cuda()
        # [b, vocab_size + seq_length]
        gen_probs = torch.cat((gen_probs, gen_padding), dim=1)
        copy_probs = probs[:, vocab_size:]
        final_probs = gen_probs + copy_probs
        log_probs = torch.log(final_probs + 10 ** -10)
        _, topi = log_probs.topk(1)
        sampled_idx = topi.view(batch_size, 1)

        # create selective read embedding for next time step
        reshaped_idxs = sampled_idx.view(-1, 1, 1).expand(one_hot_input_seq.size(0),
                                                          one_hot_input_seq.size(1), 1)
        # [b, seq_length, 1]
        pos_in_input_of_sampled_token = one_hot_input_seq.gather(2, reshaped_idxs)
        selected_scores = pos_in_input_of_sampled_token * copy_score_seq
        selected_scores_norm = F.normalize(selected_scores, p=1)
        selected_read = (selected_scores_norm * encoder_outputs).sum(dim=1).unsqueeze(1)
        return sampled_idx, log_probs, hidden, selected_read
