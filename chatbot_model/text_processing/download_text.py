import requests
from pathlib import Path
from bs4 import BeautifulSoup
import re

data_path = Path('./chatbot_model/plato_texts')
data_path.mkdir(exist_ok=True)


def get_plato_books():
    re_pattern_ebook = r"""/ebooks/\d+"""
    book_dict = {}
    url = "http://www.gutenberg.org/ebooks/author/93"
    res = requests.get(url)
    if res:
        html = BeautifulSoup(res.content, 'html.parser')
        for link in html.select('a'):
            ref = link.get('href')
            if ref and re.match(re_pattern_ebook, ref):
                book_name = link.text.strip().split('\n')[0]
                book_id = ref.split('/')[-1]
                book_url = f'http://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt'
                book_dict[book_name] = book_url

    return book_dict


book_dict = get_plato_books()
for book_name, url in book_dict.items():
    data = requests.get(url)
    print(f"Downloading {book_name}...")
    with data_path.joinpath(f"{book_name}.txt").open('w') as f:
        f.write(data.text)
    f.close()
