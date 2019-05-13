import argparse
import os
import time
import numpy as np

import torch
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from data.dataset import SequencePairDataset
from model.encoder_decoder import EncoderDecoder
from evaluate import evaluate
from utils import to_np, trim_seqs

from tensorboardX import SummaryWriter
from tqdm import tqdm
from data import parser
from data import reader


def train(encoder_decoder: EncoderDecoder,
          train_data_loader: DataLoader,
          model_name,
          val_data_loader: DataLoader,
          keep_prob,
          teacher_forcing_schedule,
          lr,
          max_length):
    global_step = 0
    loss_function = torch.nn.NLLLoss(ignore_index=0)
    optimizer = optim.Adam(encoder_decoder.parameters(), lr=lr)
    model_path = './model/' + model_name + '/'
    for epoch, teacher_forcing in enumerate(teacher_forcing_schedule):
        print('epoch {}'.format(epoch), flush=True)
        for batch_idx, (input_idxs, target_idxs, input_tokens, target_tokens) in enumerate(tqdm(train_data_loader)):
            # input_idxs and target_idxs have dim (batch_size x max_len)
            # they are NOT sorted by length
            lengths = (input_idxs != 0).long().sum(dim=1)
            sorted_lengths, order = torch.sort(lengths, descending=True)
            input_variable = Variable(input_idxs[order, :][:, :max(lengths)])
            target_variable = Variable(target_idxs[order, :])
            optimizer.zero_grad()
            output_log_probs, output_seqs = encoder_decoder(input_variable,
                                                            list(sorted_lengths),
                                                            targets=target_variable,
                                                            keep_prob=keep_prob,
                                                            teacher_forcing=teacher_forcing)
            batch_size = input_variable.shape[0]
            flattened_outputs = output_log_probs.view(batch_size * max_length, -1)
            batch_loss = loss_function(flattened_outputs, target_variable.contiguous().view(-1))
            batch_loss.backward()
            optimizer.step()
            batch_outputs = trim_seqs(output_seqs)
            batch_targets = [[list(seq[seq > 0])] for seq in list(to_np(target_variable))]
            if (global_step < 10) or \
                    (global_step % 10 == 0 and global_step < 100) or \
                    (global_step % 100 == 0 and epoch < 2):
                input_string = 'today is a good day'
                output_string = encoder_decoder.get_response(input_string)
                writer.add_text('example:', output_string, global_step=global_step)
            if global_step % 100 == 0:
                writer.add_scalar('train_batch_loss', batch_loss, global_step)
                for tag, value in encoder_decoder.named_parameters():
                    tag = tag.replace('.', '/')
                    writer.add_histogram('weights/' + tag, value, global_step, bins='doane')
                    writer.add_histogram('grads/' + tag, to_np(value.grad), global_step, bins='doane')
            global_step += 1
        val_loss = evaluate(encoder_decoder, val_data_loader)
        writer.add_scalar('val_loss', val_loss, global_step=global_step)
        encoder_embeddings = encoder_decoder.encoder.embedding.weight.data
        encoder_vocab = encoder_decoder.lang.tok_to_idx.keys()
        writer.add_embedding(encoder_embeddings, metadata=encoder_vocab, global_step=0, tag='encoder_embeddings')
        input_string = 'today is a good day'
        output_string = encoder_decoder.get_response(input_string)
        writer.add_text('example', output_string, global_step=global_step)
        print('=>', output_string)
        print('val loss: {}'.format(val_loss), flush=True)
        torch.save(encoder_decoder, '{}{}_{}.pt'.format(model_path, model_name, epoch))
        print('-' * 100, flush=True)


def main(model_name, data_list,
         use_cuda, batch_size, teacher_forcing_schedule,
         keep_prob, val_size, lr, vocab_limit, hidden_size,
         embedding_size, max_length, seed=14, interact=True):
    model_path = './model/' + model_name + '/'
    print('training model_name with use_cuda={}, batch_size={}'
          .format(model_name, use_cuda, batch_size), flush=True)
    print('teacher_forcing_schedule={}'.format(teacher_forcing_schedule), flush=True)
    print(('keep_prob={}, val_size={}, lr={}, vocab_limit={}, hidden_size={},' +
           ' embedding_size={}, max_length={}, seed={}').format(keep_prob, val_size, lr, vocab_limit, hidden_size,
                                                                embedding_size, max_length, seed))
    if os.path.isdir(model_path):
        print('loading encoder and decoder from model_path', flush=True)
        encoder_decoder = torch.load(model_path + model_name + '.pt')
        print('creating training and validation dataset with saved languages', flush=True)
        train_dataset = SequencePairDataset(data_list=data_list,
                                            lang=encoder_decoder.lang,
                                            use_cuda=use_cuda,
                                            is_val=False,
                                            val_size=val_size,
                                            use_extended_vocab=True,
                                            vocab_limit=vocab_limit)
        val_dataset = SequencePairDataset(data_list=data_list,
                                          use_cuda=use_cuda,
                                          is_val=True,
                                          val_size=val_size,
                                          use_extended_vocab=True,
                                          vocab_limit=vocab_limit)
    else:
        os.mkdir(model_path)
        print('creating trainign and validation datasets', flush=True)
        train_dataset = SequencePairDataset(data_list=data_list,
                                            vocab_limit=vocab_limit,
                                            use_cuda=use_cuda,
                                            is_val=False,
                                            val_size=val_size,
                                            seed=seed,
                                            use_extended_vocab=True)
        val_dataset = SequencePairDataset(data_list=data_list,
                                          lang=train_dataset.lang,
                                          use_cuda=use_cuda,
                                          is_val=True,
                                          val_size=val_size,
                                          seed=seed,
                                          use_extended_vocab=True,
                                          vocab_limit=vocab_limit)
        print('creating encoder-decoder model', flush=True)
        encoder_decoder = EncoderDecoder(train_dataset.lang,
                                         max_length,
                                         hidden_size=hidden_size,
                                         embedding_size=embedding_size,
                                         parse_func=parser.get_word_list,
                                         spDataset=train_dataset)
        torch.save(encoder_decoder, model_path + '/{}.pt'.format(model_name))
    if use_cuda:
        encoder_decoder = encoder_decoder.cuda()
    else:
        encoder_decoder = encoder_decoder.cpu()
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size)

    train(encoder_decoder,
          train_data_loader,
          model_name,
          val_data_loader,
          keep_prob,
          teacher_forcing_schedule,
          lr,
          encoder_decoder.decoder.max_length)
    if interact:
        encoder_decoder.interactive(False)


if __name__ == '__main__':

    data_list = reader.read_files('./data/test.txt', parser.get_word_list)

    parser_ = argparse.ArgumentParser(description='Parse training parameters')
    parser_.add_argument('--model_name', type=str, default='test-model',
                         help=('the name of subdirectory of ./model/ that' +
                               'contains encoder and decoder model files'))
    parser_.add_argument('--epochs', type=int, default=50,
                         help='the number of epochs to train')
    parser_.add_argument('--use_cuda', action='store_true', default='True',
                         help='flag indicating that cuda will be used')
    parser_.add_argument('--batch_size', type=int, default=150,
                         help='number of examples in a batch')
    parser_.add_argument('--teacher_forcing_fraction', type=float, default=0.4,
                         help='fraction of batches that will use teacher forcing during training')
    parser_.add_argument('--scheduled_teacher_forcing', action='store_true', default=1.0,
                         help=('Linearly decrease the teacher forcing fraction' +
                               ' from 1.0 to 0.0 over the specified number of epochs'))
    parser_.add_argument('--keep_prob', type=float, default=1.0,
                         help='Probability of keeping an element in the dropout step.')

    parser_.add_argument('--val_size', type=float, default=0.1,
                         help='fraction of data to use for validation')

    parser_.add_argument('--lr', type=float, default=0.0005,
                         help='Learning rate.')

    parser_.add_argument('--vocab_limit', type=int, default=150,
                         help='When creating a new Language object the vocab'
                              'will be truncated to the most frequently'
                              'occurring words in the training dataset.')

    parser_.add_argument('--hidden_size', type=int, default=16,
                         help='The number of RNN units in the encoder. 2x this '
                              'number of RNN units will be used in the decoder')

    parser_.add_argument('--embedding_size', type=int, default=64,
                         help='Embedding size used in both encoder and decoder')

    parser_.add_argument('--max_length', type=int, default=30,
                         help='Sequences will be padded or truncated to this size.')

    args = parser_.parse_args()

    writer = SummaryWriter('./logs/%s_%s' % (args.model_name, str(int(time.time()))))
    if args.scheduled_teacher_forcing:
        schedule = np.arange(1.0, 0.0, -1.0 / args.epochs)
    else:
        schedule = np.ones(args.epochs) * args.teacher_forcing_fraction

    main(args.model_name, data_list,
         args.use_cuda, args.batch_size,
         schedule, args.keep_prob, args.val_size,
         args.lr, args.vocab_limit,
         args.hidden_size,
         args.embedding_size, args.max_length)
