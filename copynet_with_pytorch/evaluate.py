import torch
from torch.autograd import Variable

from torch import nn
from tqdm import tqdm
from model.encoder_decoder import EncoderDecoder
from utils import trim_seqs, to_np, seq_to_string


def evaluate(encoder_decoder: EncoderDecoder, data_loader):
    # what does this return for ignored idxs? same length output?
    loss_function = nn.NLLLoss(ignore_index=0, reduce=False)
    losses = []
    all_output_seqs = []
    all_target_seqs = []
    for batch_idx, (input_idxs, target_idxs, _, _) in enumerate(tqdm(data_loader)):
        input_lengths = (input_idxs != 0).long().sum(dim=1)
        sorted_lengths, order = torch.sort(input_lengths, descending=True)
        with torch.no_grad():
            input_variable = Variable(input_idxs[order, :][:, :max(input_lengths)]) #, volatile=True)
            target_variable = Variable(target_idxs[order, :]) #, volatile=True)
        batch_size = input_variable.shape[0]
        output_log_probs, output_seqs = encoder_decoder(input_variable, list(sorted_lengths))
        all_output_seqs.extend(trim_seqs(output_seqs))
        all_target_seqs.extend([list(seq[seq > 0])] for seq in to_np(target_variable))
        flattened_log_probs = output_log_probs.view(batch_size * encoder_decoder.decoder.max_length, -1)
        batch_losses = loss_function(flattened_log_probs, target_variable.contiguous().view(-1))
        losses.extend(list(to_np(batch_losses)))
    mean_loss = len(losses) / sum(losses)
    return mean_loss


def print_output(input_seq, encoder_decoder: EncoderDecoder,
                 input_tokens=None, target_tokens=None, target_seq=None):
    idx_to_tok = encoder_decoder.lang.idx_to_tok
    if input_tokens is not None:
        input_string = ''.join(input_tokens)
    else:
        input_string = seq_to_string(input_tokens, idx_to_tok)
    lengths = list((input_seq != 0).long().sum(dim=0))
    input_variable = Variable(input_seq).view(1, -1)
    target_variable = Variable(target_seq).view(1, -1)
    if target_tokens is not None:
        target_string = ''.join(target_tokens)
    elif target_seq is not None:
        target_string = seq_to_string(target_seq, idx_to_tok, input_tokens=input_tokens)
    else:
        target_string = ''
    if target_seq is not None:
        target_eos_idx = list(target_seq).index(2) if 2 in list(target_seq) else len(target_seq)
        target_outputs, _ = encoder_decoder(input_variable, lengths,
                                            targets=target_variable, teacher_forcing=1.0)
        target_log_prob = sum([target_outputs[0, step_idx, target_idx]
                               for step_idx, target_idx in enumerate(target_seq[:target_eos_idx + 1])])
    outputs, idxs = encoder_decoder(input_variable, lengths)
    idxs = idxs.data.view(-1)
    eos_idx = list(idxs).index(2) if 2 in list(idxs) else len(idxs)
    string = seq_to_string(idxs[:eos_idx + 1], idx_to_tok, input_tokens=input_tokens)
    log_prob = sum([outputs[0, step_idx, idx] for step_idx, idx in enumerate(idxs[:eos_idx + 1])])
    print('>{}\n'.format(input_string), flush=True)
    if target_seq is not None:
        print('={}'.format(target_string), flush=True)
    print('<{}'.format(string), flush=True)
    if target_seq is not None:
        print('target_log_prob:{}'.format(target_log_prob), flush=True)
    print('output log prob:{}'.format(float(log_prob)))
    print('-' * 100, '\n')
    return idxs
