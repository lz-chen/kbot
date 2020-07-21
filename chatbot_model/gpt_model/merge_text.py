# Get list of file Names
import glob
from pathlib import Path
import re


def merge():
    texts_dir = Path('chatbot_model/texts/plato_texts')
    filenames = texts_dir.glob("*.txt")
    # Add all texts into one file
    with texts_dir.joinpath('corpus.txt').open('w') as outfile:
        for fname in filenames:
            with fname.open('r') as infile:
                lines = infile.read()
                match = re.search(r'\*\*\*.*?\*\*\*(.*)\*\*\*.*?\*\*\*.*', lines, re.DOTALL)
                if match:
                    content = match.group(1)
                outfile.write(content)
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
    merge_nc()
