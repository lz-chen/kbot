from pathlib import Path
from typing import Optional
import numpy as np
from torchtext.vocab import GloVe

from chatbot_model.vocabulary import Vocabulary


def map_pretrained_embeddings(embedding_path: str =
                              '/media/liah/DATA/glove_embeddings/glove.6B.300d.txt',
                              vocab_fname='./nc_data/formatted_texts/nc.pkl',
                              save_to: Optional[str] = '/media/liah/DATA/glove_embeddings'):
    word_vec = GloVe()
    nc_vocab = Vocabulary.from_file(Path(vocab_fname))
    weight_matrix = np.random.normal(size=(nc_vocab.num_words, word_vec.dim))
    found_cnt = 0
    found_idxs = []
    for i, word in enumerate(word_vec.itos):
        if word in nc_vocab.word2index.keys():
            word_idx = nc_vocab.word2index[word]
            weight_matrix[word_idx, :] = word_vec.vectors[i].numpy()
            found_cnt += 1
            found_idxs.append(word_idx)

    print(f'Found pretrained vectors for {found_cnt} word')
    if save_to:
        output_path = Path(save_to).joinpath(nc_vocab.name + '.npy').as_posix()
        np.save(output_path, weight_matrix)
        print(f'Save weight matrix to {output_path}')


if __name__ == '__main__':
    map_pretrained_embeddings(vocab_fname='/home/liah/Documents/katebot/film_data/film.pkl')
