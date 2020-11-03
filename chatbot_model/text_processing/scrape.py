from requests import get
from requests.exceptions import RequestException
from contextlib import closing
from bs4 import BeautifulSoup


def simple_get(url):
    """
    Attempts to get the content at `url` by making an HTTP GET request.
    If the content-type of response is some kind of HTML/XML, return the
    text content, otherwise return None.
    """
    try:
        with closing(get(url, stream=True)) as resp:
            if is_good_response(resp):
                return resp.content
            else:
                return None

    except RequestException as e:
        log_error('Error during requests to {0} : {1}'.format(url, str(e)))
        return None


def is_good_response(resp):
    """
    Returns True if the response seems to be HTML, False otherwise.
    """
    content_type = resp.headers['Content-Type'].lower()
    return (resp.status_code == 200
            and content_type is not None
            and content_type.find('html') > -1)


def log_error(e):
    """
    It is always a good idea to log errors.
    This function just prints them, but you can
    make it do anything.
    """
    print(e)


def get_names():
    """
    Downloads the page where the list of mathematicians is found
    and returns a list of strings, one per mathematician
    """
    url = 'http://www.fabpedigree.com/james/mathmen.htm'
    response = simple_get(url)

    if response is not None:
        html = BeautifulSoup(response, 'html.parser')
        names = set()
        for li in html.select('li'):
            for name in li.text.split('\n'):
                if len(name) > 0:
                    names.add(name.strip())
        return list(names)

    # Raise an exception if we failed to get any data from the url
    raise Exception('Error retrieving contents at {}'.format(url))


def parse_page_text(site_url, save_to='./interviews.txt'):
    response = simple_get(site_url)
    all_texts = {}
    if response is not None:
        html = BeautifulSoup(response, 'html.parser')
        for link in html.select('a'):
            if not link.text.isupper():
                texts = []
                ref = link.get('href')
                transcript_link = '/'.join(site_url.split('/')[:-2] + [ref.split('/')[-2]])
                transcript_page_ref = simple_get(transcript_link)
                if transcript_page_ref is not None:
                    transcript_html = BeautifulSoup(transcript_page_ref)
                    for para in transcript_html.select('p'):
                        if para.text.strip():
                            texts.append(para.text.strip())
                all_texts[link.text] = texts

    with open(save_to, 'w+') as f:
        for k, v in all_texts.items():
            f.write(f'##################{k}##################\n')
            f.write('\n'.join(v) + '\n')


if __name__ == '__main__':
    topic = 'debates'
    site_url = f'https://chomsky.info/{topic}/'
    parse_page_text(site_url, f'./{topic}.txt')
    # print('Getting the list of names....')
    # names = get_names()
    # print('... done.\n')
    #
    # results = []
    #
    # print('Getting stats for each name....')
    #
    # for name in names:
    #     try:
    #         hits = get_hits_on_name(name)
    #         if hits is None:
    #             hits = -1
    #         results.append((hits, name))
    #     except:
    #         results.append((-1, name))
    #         log_error('error encountered while processing '
    #                   '{}, skipping'.format(name))
    #
    # print('... done.\n')
    #
    # results.sort()
    # results.reverse()
    #
    # if len(results) > 5:
    #     top_marks = results[:5]
    # else:
    #     top_marks = results
    #
    # print('\nThe most popular mathematicians are:\n')
    # for (mark, mathematician) in top_marks:
    #     print('{} with {} pageviews'.format(mathematician, mark))
    #
    # no_results = len([res for res in results if res[0] == -1])
    # print('\nBut we did not find results for '
    #       '{} mathematicians on the list'.format(no_results))
