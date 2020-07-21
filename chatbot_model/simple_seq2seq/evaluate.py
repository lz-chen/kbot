from pathlib import Path
from typing import Optional

from torch import nn

from chatbot_model.data_to_tensor import indexesFromSentence
from chatbot_model.format_data import MAX_LENGTH, normalizeString
import torch

from chatbot_model.vocabulary import Vocabulary
from chatbot_model.simple_seq2seq.decoder_rnn import LuongAttnDecoderRNN
from chatbot_model.simple_seq2seq.encoder_rnn import EncoderRNN
from chatbot_model.simple_seq2seq.greedy_search_decoder import GreedySearchDecoder

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
print(f"using cuda: {USE_CUDA}")


class modelEvaluator:
    def __init__(self,
                 load_filename,
                 vocab_fname: str = 'nc_data/formatted_texts/nc.pkl'):
        self.load_filename = load_filename
        self.voc = Vocabulary.from_file(Path(vocab_fname))

    def configure_model(self,
                        attn_model: str = 'dot',
                        pretrained_embed_path: Optional[Path] = None,
                        embed_size: int = 300,
                        hidden_size: int = 200,
                        encoder_n_layers: int = 2,
                        decoder_n_layers: int = 2,
                        dropout: float = 0.1
                        ):
        # attn_model = 'general'
        # attn_model = 'concat'
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.encoder_n_layers = encoder_n_layers
        self.decoder_n_layers = decoder_n_layers
        self.dropout = dropout

        # loadFilename = os.path.join(save_dir, model_name, corpus_name,
        #                            '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers,
        #                            hidden_size),
        #                            '{}_checkpoint.tar'.format(checkpoint_iter))

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
        self.embedding = nn.Embedding(self.voc.num_words, embed_size)
        if self.load_filename:
            self.embedding.load_state_dict(self.embedding_sd)
        # Initialize encoder & decoder models
        self.encoder = EncoderRNN(self.embed_size,
                                  hidden_size,
                                  self.embedding,
                                  encoder_n_layers,
                                  dropout)
        self.decoder = LuongAttnDecoderRNN(attn_model,
                                           self.embedding,
                                           hidden_size,
                                           self.voc.num_words,
                                           decoder_n_layers, dropout)
        if self.load_filename:
            self.encoder.load_state_dict(self.encoder_sd)
            self.decoder.load_state_dict(self.decoder_sd)
        # Use appropriate device
        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)
        print('Models loaded and ready to go!')
        # return encoder, decoder

    @staticmethod
    def evaluate(searcher, sentence, voc: Vocabulary = None):
        # voc = self.voc if voc is None else voc
        ### Format input sentence as a batch
        # words -> indexes
        indexes_batch = [indexesFromSentence(voc, sentence)]
        # Create lengths tensor
        lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
        # Transpose dimensions of batch to match models' expectations
        input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
        # Use appropriate device
        input_batch = input_batch.to(device)
        lengths = lengths.to(device)
        # Decode sentence with searcher
        tokens, scores = searcher(input_batch, lengths, MAX_LENGTH)
        # indexes -> words
        decoded_words = [voc.index2word[token.item()] for token in tokens]
        return decoded_words

    def evaluateInput(self):
        self.encoder.eval()
        self.decoder.eval()
        searcher = GreedySearchDecoder(self.encoder, self.decoder)
        input_sentence = ''
        while (1):
            try:
                # Get input sentence
                input_sentence = input('> ')
                # Check if it is quit case
                if input_sentence == 'q' or input_sentence == 'quit': break
                # Normalize sentence
                input_sentence = normalizeString(input_sentence)
                # Evaluate sentence
                output_words = self.evaluate(searcher, input_sentence)
                # Format and print response sentence
                output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
                print('Bot:', ' '.join(output_words))

            except KeyError:
                print("Error: Encountered unknown word.")

    @staticmethod
    def eval_during_train(encoder, decoder, pairs, voc):
        # encoder.eval()
        # decoder.eval()
        searcher = GreedySearchDecoder(encoder, decoder)
        input_pairs = pairs[345:350]
        for input_pair in input_pairs:
            input_sentence = input_pair[0]
            print(f"Test input: {input_sentence}")
            input_sentence = normalizeString(input_sentence)
            try:
                # Evaluate sentence
                output_words = modelEvaluator.evaluate(searcher, input_sentence, voc=voc)
                # Format and print response sentence
                output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
                print('Bot:', ' '.join(output_words))
            except KeyError:
                print("Error: Encountered unknown word.")
            print(f"Gold output: {input_pair[1]}")
            print("============================")


if __name__ == '__main__':
    embedding_path = Path('/media/liah/DATA/glove_embeddings/nc.npy')
    evaluator = modelEvaluator(
        '/chatbot_model/nc_chat_bot_model/nc_chatbot/nc_interviews/2-2_200/20_checkpoint.tar')
    evaluator.configure_model(pretrained_embed_path=embedding_path)
    evaluator.evaluateInput()
