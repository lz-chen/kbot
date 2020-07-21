from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from pathlib import Path
from typing import Optional
import numpy as np

import torch
import torch.nn as nn
from torch import optim
import random
import os

from chatbot_model.data_to_tensor import batch2TrainData
from chatbot_model.format_data import loadPrepareData
from chatbot_model.vocabulary import SOS_token, Vocabulary
from chatbot_model.simple_seq2seq.decoder_rnn import LuongAttnDecoderRNN
from chatbot_model.simple_seq2seq.encoder_rnn import EncoderRNN
from chatbot_model.simple_seq2seq.evaluate import modelEvaluator
from chatbot_model.simple_seq2seq.utils import map_pretrained_embeddings

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
print(f"using cuda: {USE_CUDA}")


def maskNLLLoss(inp, target, mask):
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, nTotal.item()


class modelTrainer:
    def __init__(self,
                 vocab_fname: str = 'nc_data/formatted_texts/nc.pkl',
                 load_filename: Optional[str] = None
                 ):

        self.load_filename = load_filename
        self.voc = Vocabulary.from_file(Path(vocab_fname))

    def configure_model(self,
                        attn_model: str = 'dot',
                        pretrained_embed_path: Optional[Path] = None,
                        embed_size: int = 300,
                        hidden_size: int = 200,
                        encoder_n_layers: int = 2,
                        decoder_n_layers: int = 2,
                        dropout: float = 0.1,
                        checkpoint_iter=4000
                        ):
        # attn_model = 'general'
        # attn_model = 'concat'
        self.attn_model = attn_model
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.encoder_n_layers = encoder_n_layers
        self.decoder_n_layers = decoder_n_layers
        self.dropout = dropout
        # Set checkpoint to load from; set to None if starting from scratch

        # loadFilename = os.path.join(save_dir, model_name, corpus_name,
        #                            '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers,
        #                            hidden_size),
        #                            '{}_checkpoint.tar'.format(checkpoint_iter))

        # Load model if a loadFilename is provided
        if self.load_filename:
            # If loading on same machine the model was trained on
            self.checkpoint = torch.load(self.load_filename)
            # If loading a model trained on GPU to CPU
            # checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
            self.encoder_sd = self.checkpoint['en']
            self.decoder_sd = self.checkpoint['de']
            self.encoder_optimizer_sd = self.checkpoint['en_opt']
            self.decoder_optimizer_sd = self.checkpoint['de_opt']
            self.embedding_sd = self.checkpoint['embedding']
            self.voc.__dict__ = self.checkpoint['voc_dict']

        print('Building encoder and decoder ...')
        # Initialize word embeddings

        if pretrained_embed_path:
            # https://stackoverflow.com/questions/55042931/cudnn-error-cudnn-status-bad-param-can
            # -someone-explain-why-i-am-getting-this-er
            embed_weight = np.load(pretrained_embed_path.as_posix()).astype(np.float32)
            self.embedding = nn.Embedding.from_pretrained(
                embeddings=torch.from_numpy(embed_weight),
                freeze=True)
        else:
            self.embedding = nn.Embedding(self.voc.num_words, self.embed_size)

        if self.load_filename:
            self.embedding.load_state_dict(self.embedding_sd)
        # Initialize encoder & decoder models
        self.encoder = EncoderRNN(self.embed_size,
                                  self.hidden_size,
                                  self.embedding,
                                  self.encoder_n_layers,
                                  dropout)
        self.decoder = LuongAttnDecoderRNN(self.attn_model,
                                           self.embedding,
                                           self.hidden_size,
                                           self.voc.num_words,
                                           self.decoder_n_layers,
                                           self.dropout)
        if self.load_filename:
            self.encoder.load_state_dict(self.encoder_sd)
            self.decoder.load_state_dict(self.decoder_sd)
        # Use appropriate device
        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)
        print('Models built and ready to go!')
        print(self.encoder)
        print(self.decoder)
        # return encoder, decoder

    def configure_train(self,
                        model_name='nc_chatbot',
                        clip=50.0,
                        teacher_forcing_ratio=0.7,
                        learning_rate=0.01,
                        decoder_learning_ratio=5.0,
                        n_iteration=4000,
                        print_every=10,
                        eval_every=5,
                        save_every=500,
                        batch_size=64):
        # Ensure dropout layers are in train mode
        self.model_name = model_name
        self.clip = clip
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.learning_rate = learning_rate
        self.n_iteration = n_iteration
        self.print_every = print_every
        self.eval_every = eval_every
        self.save_every = save_every
        self.batch_size = batch_size

        self.encoder.train()
        self.decoder.train()

        # Initialize optimizers
        print('Building optimizers ...')
        self.encoder_optimizer = optim.Adam(self.
                                            encoder.parameters(), lr=learning_rate)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(),
                                            lr=learning_rate * decoder_learning_ratio)
        if self.load_filename:
            self.encoder_optimizer.load_state_dict(self.encoder_optimizer_sd)
            self.decoder_optimizer.load_state_dict(self.decoder_optimizer_sd)

        # If you have cuda, configure cuda to call
        for state in self.encoder_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

        for state in self.decoder_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

    def train(self,
              pairs,
              save_dir,
              n_iteration,
              corpus_name):
        # Load batches for each iteration
        # training_batches = [batch2TrainData(self.voc,
        #                                     [random.choice(pairs) for _ in range(
        #                                         self.batch_size)]) for _ in range(n_iteration)]

        # Initializations
        print('Initializing ...')
        start_iteration = 1
        print_loss = 0
        if self.load_filename:
            start_iteration = self.checkpoint['iteration'] + 1

        # Training loop
        print("Training...")
        for iteration in range(start_iteration, n_iteration + 1):
            # random.shuffle(pairs)
            pairs = np.random.permutation(pairs)
            # todo the end batch is not included
            training_batches = [
                batch2TrainData(self.voc,
                                [x for x in pairs[nbatch * self.batch_size:(nbatch + 1) *
                                                                           self.batch_size]]) for
                nbatch in range(len(pairs) // self.batch_size)]

            for i_batch, training_batch in enumerate(training_batches):
                # training_batch = training_batches[iteration - 1]
                # Extract fields from batch
                input_variable, lengths, target_variable, mask, max_target_len = training_batch

                # Run a training iteration with batch
                loss = self.train_one_batch(input_variable, lengths, target_variable, mask,
                                            max_target_len)
                print_loss += loss

            # Print progress
            if iteration % self.print_every == 0:
                print_loss_avg = print_loss / self.print_every
                print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(
                    iteration,
                    iteration / n_iteration * 100,
                    print_loss_avg))
                print_loss = 0

            if iteration % self.eval_every == 0:
                self.encoder.eval()
                self.decoder.eval()
                modelEvaluator.eval_during_train(self.encoder,
                                                 self.decoder,
                                                 pairs,
                                                 self.voc)
                self.encoder.train()
                self.decoder.train()

            # Save checkpoint
            if (iteration % self.save_every == 0):
                directory = os.path.join(save_dir, self.model_name, corpus_name,
                                         '{}-{}_{}'.format(self.encoder_n_layers,
                                                           self.decoder_n_layers,
                                                           self.hidden_size))
                if not os.path.exists(directory):
                    os.makedirs(directory)
                torch.save({
                    'iteration': iteration,
                    'en': self.encoder.state_dict(),
                    'de': self.decoder.state_dict(),
                    'en_opt': self.encoder_optimizer.state_dict(),
                    'de_opt': self.decoder_optimizer.state_dict(),
                    'loss': loss,
                    'voc_dict': self.voc.__dict__,
                    'embedding': self.embedding.state_dict()
                }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))

    def train_one_batch(self,
                        input_variable,
                        lengths,
                        target_variable,
                        mask,
                        max_target_len):
        """The train function contains the algorithm for a single training iteration (a single batch
        of inputs)."""
        # Zero gradients
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        # Set device options
        input_variable = input_variable.to(device)
        lengths = lengths.to(device)
        target_variable = target_variable.to(device)
        mask = mask.to(device)

        # Initialize variables
        loss = 0
        print_losses = []
        n_totals = 0

        # Forward pass through encoder
        encoder_outputs, encoder_hidden = self.encoder(input_variable, lengths)

        # Create initial decoder input (start with SOS tokens for each sentence)
        decoder_input = torch.LongTensor([[SOS_token for _ in range(self.batch_size)]])
        decoder_input = decoder_input.to(device)

        # Set initial decoder hidden state to the encoder's final hidden state
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]

        # Determine if we are using teacher forcing this iteration
        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False

        # Forward batch of sequences through decoder one time step at a time
        if use_teacher_forcing:
            for t in range(max_target_len):
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                # Teacher forcing: next input is current target
                decoder_input = target_variable[t].view(1, -1)
                # Calculate and accumulate loss
                mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
                loss += mask_loss
                print_losses.append(mask_loss.item() * nTotal)
                n_totals += nTotal
        else:
            for t in range(max_target_len):
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                # No teacher forcing: next input is decoder's own current output
                _, topi = decoder_output.topk(1)
                decoder_input = torch.LongTensor([[topi[i][0] for i in range(self.batch_size)]])
                decoder_input = decoder_input.to(device)
                # Calculate and accumulate loss
                mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
                loss += mask_loss
                print_losses.append(mask_loss.item() * nTotal)
                n_totals += nTotal

        # Perform backpropatation
        loss.backward()

        # Clip gradients: gradients are modified in place
        _ = nn.utils.clip_grad_norm_(self.encoder.parameters(), self.clip)
        _ = nn.utils.clip_grad_norm_(self.decoder.parameters(), self.clip)

        # Adjust model weights
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return sum(print_losses) / n_totals


if __name__ == '__main__':
    # todo make train config file in json

    # save_dir = Path('nc_data/formatted_texts')
    # corpus_name = 'nc_interviews'
    # datafile = Path('nc_data/texts/interviews.txt')
    # embedding_path = Path('/media/liah/DATA/glove_embeddings/nc.npy')
    # voc, pairs = loadPrepareData(corpus_name,
    #                              datafile,
    #                              save_dir,
    #                              vocab_name='nc',
    #                              save_vocab=False,
    #                              update_vocab=False)
    #
    # model_save_dir = 'nc_chat_bot_model'
    # n_iteration = 20
    # # Run training iterations
    # print("Starting Training!")
    # trainer = modelTrainer()
    # trainer.configure_model(pretrained_embed_path=embedding_path)
    # trainer.configure_train(print_every=1,
    #                         eval_every=5,
    #                         save_every=5,
    #                         learning_rate=0.01,
    #                         teacher_forcing_ratio=0.4,
    #                         decoder_learning_ratio=2,
    #                         batch_size=256)
    # trainer.train(pairs,
    #               model_save_dir,
    #               n_iteration,
    #               corpus_name)

    ############################################################################
    # save_dir = Path('film_data')
    # corpus_name = 'cornell movie-dialogs corpus'
    #
    # datafile = Path('film_data/cornell movie-dialogs corpus/formatted_movie_lines.txt')
    # embedding_path = Path('/media/liah/DATA/glove_embeddings/film.npy')
    # voc, pairs = loadPrepareData(corpus_name,
    #                              datafile,
    #                              save_dir,
    #                              vocab_name='film',
    #                              save_corpus=False,
    #                              save_vocab=True,
    #                              update_vocab=True)
    #
    # model_save_dir = 'film_chat_bot_model'
    # n_iteration = 50
    # # Run training iterations
    # print("Starting Training!")
    # vocab_fname = './film_data/film.pkl'
    # trainer = modelTrainer(vocab_fname=vocab_fname)
    # map_pretrained_embeddings(vocab_fname=vocab_fname)
    # trainer.configure_model(pretrained_embed_path=embedding_path)
    # # trainer.configure_model()
    # trainer.configure_train(print_every=1,
    #                         eval_every=2,
    #                         save_every=5,
    #                         learning_rate=0.0001,
    #                         teacher_forcing_ratio=1.0,
    #                         decoder_learning_ratio=5,
    #                         batch_size=64)
    # trainer.train(pairs,
    #               model_save_dir,
    #               n_iteration,
    #               corpus_name)

    save_dir = Path('nc_data')

    corpus_name = 'nc_interviews'
    datafile = Path('nc_data/texts/interviews.txt')
    embedding_path = Path('/media/liah/DATA/glove_embeddings/nc.npy')
    voc, pairs = loadPrepareData(corpus_name,
                                 datafile,
                                 save_dir,
                                 vocab_name='nc',
                                 save_corpus=True,
                                 save_vocab=True,
                                 update_vocab=True)

    model_save_dir = 'nc_chat_bot_model'
    n_iteration = 50
    # Run training iterations
    print("Starting Training!")

    vocab_fname = './nc_data/nc.pkl'
    trainer = modelTrainer(vocab_fname=vocab_fname)
    map_pretrained_embeddings(vocab_fname=vocab_fname)
    trainer.configure_model(pretrained_embed_path=embedding_path)
    # trainer.configure_model()
    trainer.configure_train(print_every=1,
                            eval_every=2,
                            save_every=5,
                            learning_rate=0.0001,
                            teacher_forcing_ratio=1.0,
                            decoder_learning_ratio=5,
                            batch_size=64)
    trainer.train(pairs,
                  model_save_dir,
                  n_iteration,
                  corpus_name)
