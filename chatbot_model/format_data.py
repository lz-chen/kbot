import re
import unicodedata
from pathlib import Path
from typing import Optional, List

from chatbot_model.vocabulary import Vocabulary

MAX_LENGTH = 30  # Maximum sentence length to consider
MIN_VOCAB_COUNT = 3  # Minimum word count threshold for trimming
ABBRS = ['e.g.', 'i.e.', 'u.s.', 'u.s.s.r', 'i.e', ]


def unicodeToAscii(s: str) -> str:
    """Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427"""
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalizeString(s):
    """Lowercase, trim, and remove non-letter characters"""
    s = unicodeToAscii(s)
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s


def split_line(l: str) -> List[str]:
    # remove speaker name
    l = l.split(':')[-1].lower().strip()
    for abbr in ABBRS:
        l.replace(abbr, ''.join(abbr.split('.')))
    delimiters = ['.', '?', '!']
    regexPattern = '|'.join(map(re.escape, delimiters))
    return [normalizeString(s) for s in re.split(regexPattern, l) if s != '']


def get_sentence_pairs(datafile: Path):
    """Read query/response pairs and return a voc object"""
    print("Reading lines...")
    # Read the file and split into lines
    lines = open(datafile.as_posix(), encoding='utf-8'). \
        read().strip().split('\n')
    # Split every line into pairs and normalize
    if not datafile.as_posix().startswith('film'):
        sentences = []
        for i, l in enumerate(lines):
            if not l.startswith('###'):
                sentences += split_line(l)
        pairs = []
        for i in range(0, len(sentences), 2):
            pairs.append((sentences[i], sentences[min(len(sentences) - 1, i + 1)]))
    else:
        pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    return pairs


def filterPair(p):
    """Returns True iff both sentences in a pair 'p' are under the MAX_LENGTH threshold"""
    # Input sequences need to preserve the last word for EOS token
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH


def filterPairs(pairs):
    """Filter pairs using filterPair condition"""
    return [pair for pair in pairs if filterPair(pair)]


# Using the functions defined above, return a populated voc object and pairs list
def loadPrepareData(corpus_name: str,
                    datafile: Path,
                    save_dir: Path,
                    vocab_name: Optional[str] = None,
                    save_corpus: bool = True,
                    save_vocab: bool = True,
                    update_vocab: bool = True,
                    ):
    if not save_dir.is_dir():
        save_dir.mkdir()
    print("Start preparing training data ...")
    pairs = get_sentence_pairs(datafile)
    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs = filterPairs(pairs)

    vocab_path = save_dir.joinpath(vocab_name + '.pkl')
    if vocab_path.is_file():
        print(f'Load existing vocabulary from {vocab_path.as_posix()}')
        voc = Vocabulary.from_file(vocab_path)
    else:
        voc = Vocabulary(vocab_name)

    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    print("Counting words...")
    for pair in pairs:
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])
    print("Counted words:", voc.num_words)

    pairs = trimRareWords(voc, pairs, MIN_VOCAB_COUNT)
    if save_vocab:
        # vocab_path = save_dir.joinpath(f'{corpus_name}_vocabulary.pkl')
        print(f'Saving vocabulary to {vocab_path}...')
        if vocab_path.is_file() and not update_vocab:
            print('Vocabulary file already exists!'
                  'If you want to update it, please set update_vocab = True')
        else:
            voc.to_file(fpath=vocab_path)

    if save_corpus:
        corpus_path = save_dir.joinpath(f'{corpus_name}_corpus.txt')
        with open(corpus_path.as_posix(), 'w') as f:
            f.writelines('\n'.join(['\t'.join(pair) for pair in pairs]))

    return voc, pairs


def trimRareWords(voc, pairs, MIN_COUNT):
    # Trim words used under the MIN_COUNT from the voc
    voc.trim(MIN_COUNT)
    # Filter out pairs with trimmed words
    keep_pairs = []
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True
        # Check input sentence
        for word in input_sentence.split(' '):
            if word not in voc.word2index:
                keep_input = False
                break
        # Check output sentence
        for word in output_sentence.split(' '):
            if word not in voc.word2index:
                keep_output = False
                break

        # Only keep pairs that do not contain trimmed word(s) in their input or output sentence
        if keep_input and keep_output:
            keep_pairs.append(pair)

    print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keep_pairs),
                                                                len(keep_pairs) / len(pairs)))
    return keep_pairs


if __name__ == '__main__':
    # Load/Assemble voc and pairs
    save_dir = Path('./nc_data/formatted_texts')
    topics = ['debates', 'interviews', 'talks']
    for topic in topics:
        datafile = Path(f'./nc_data/texts/{topic}.txt')
        corpus_name = f'nc_{topic}'
        voc, pairs = loadPrepareData(corpus_name, datafile, save_dir, vocab_name='nc')

    # Print some pairs to validate
    print("\npairs:")
    for pair in pairs[:10]:
        print(pair)
