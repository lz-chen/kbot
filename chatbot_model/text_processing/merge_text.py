"""
Script for merging together texts scraped online, in order to make a single corpus file for
future processing
"""
from pathlib import Path
import re


def clean_sentence(s, chunk_size=50):
    """
    A sentence in Plato book is usually of form SPEAKER: sentence ....
    this function strip away the speaker, keeps only the text. In the case of sentences more than
    {chunk_size} words, it returns several chunks of texts, otherwise one text
    """
    s = s.replace('\n', ' ')
    parts = s.split(':')
    if len(parts) <= 2:  # no speaker or one sentence, does not contain ':'
        text = parts[-1]
    else:  # one sentence contains ':'
        text = ':'.join(parts[1:])
    text = text.strip()
    if len(text.split()) < chunk_size:
        return text
    else:
        texts = []
        sentence_list = re.split(r'[\.\?!]', text)
        len_added = 0
        sent_added = ''
        # add sentences upto 50 words
        for sent in sentence_list:
            sent = sent + '.'
            if len(sent.split()) + len_added < 50:
                sent_added += sent
                len_added += len(sent.split())
            else:
                texts.append(sent_added)
                sent_added = sent
                len_added = len(sent.split())
        if sent_added != '' and len(sent_added) > 1:
            texts.append(sent_added)
        return texts


def clean_dialouge_text(text, file_name):
    # todo Crito.txt doesn't work with this pattern
    file_name = file_name.upper()
    patt = rf'''.*INTRODUCTION.*?{file_name}(.*)End of the|this Project Gutenberg EBook|Etext.*'''
    match = re.search(patt, text, re.DOTALL)
    cleaned_text = ''
    if match:
        content = match.group(1)
        cleaned_sentences = []
        sentences = content.split('\n\n')
        for s in sentences:
            if s == '':
                continue
            cleaned = clean_sentence(s)
            if isinstance(cleaned, str):
                cleaned_sentences.append(cleaned)
            elif isinstance(cleaned, list):
                cleaned_sentences += cleaned
        cleaned_text = '\n'.join(cleaned_sentences)
    return cleaned_text


def merge_plato():
    texts_dir = Path('chatbot_model/texts/plato')
    filenames = texts_dir.glob("*.txt")
    # Add all texts into one file
    with texts_dir.joinpath('corpus.txt').open('w') as outfile:
        for fname in filenames:
            file_name = fname.name.strip('.txt')
            if file_name.startswith('_') or file_name == 'corpus':
                continue
            print(f"processing {fname}...")
            with fname.open('r') as infile:
                lines = infile.read()
                clean_content = clean_dialouge_text(lines, file_name) + '\n'
                outfile.write(clean_content)
    outfile.close()


def merge_nc():
    texts_dir = Path('chatbot_model/texts/nc_texts/texts')
    filenames = texts_dir.glob("*.txt")
    with texts_dir.joinpath('corpus.txt').open('w') as outfile:
        for filename in filenames:
            cleaned_lines = ''
            with filename.open('r') as infile:
                for line in infile:
                    if not line.strip().startswith('#####'):
                        cleaned_lines += line.split(':')[-1]
            outfile.write(cleaned_lines + '\n\n')


if __name__ == '__main__':
    merge_plato()
