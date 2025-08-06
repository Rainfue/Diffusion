import os
import time
import random
import logging
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from urllib.parse import urljoin, urlparse, quote_plus

# --- Конфигурация ---
TAG = 'Ayanami Rei'
TAG_PATH = quote_plus(TAG)  # "Ayanami+Rei"
SAFEBORU_TAG = TAG.lower().replace(' ', '_')  # "ayanami_rei"
SAFEBORU_BASE = 'https://safebooru.org'
ZEROCHAN_BASE = 'https://www.zerochan.net'

OUTPUT_DIR_SAFEBORU = 'images/safebooru'
OUTPUT_DIR_ZEROCHAN = 'images/zerochan'
PAGES_SAFEBORU = 80  # число страниц для Safebooru
PAGES_ZEROCHAN = 10  # число страниц для Zerochan

# HTTP headers и сессия
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
session = requests.Session()
session.headers.update(HEADERS)

# Логирование
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Создать папки
os.makedirs(OUTPUT_DIR_SAFEBORU, exist_ok=True)
os.makedirs(OUTPUT_DIR_ZEROCHAN, exist_ok=True)


def download_image(url: str, folder: str):
    """Скачивание картинки по URL в указанную папку"""
    try:
        filename = os.path.basename(urlparse(url).path)
        path = os.path.join(folder, filename)
        if os.path.exists(path):
            logging.debug(f'Skipped existing {filename}')
            return
        resp = session.get(url, stream=True, timeout=10)
        resp.raise_for_status()
        with open(path, 'wb') as f:
            for chunk in resp.iter_content(1024):
                f.write(chunk)
    except Exception as e:
        logging.warning(f'Failed to download {url}: {e}')


def parse_safebooru():
    """Парсит Safebooru: для каждой миниатюры заходит на страницу поста и скачивает полную версию картинки"""
    logging.info('Parsing Safebooru...')
    for page in tqdm(range(PAGES_SAFEBORU), desc='Safebooru pages'):
        page = page+30
        listing_url = f"{SAFEBORU_BASE}/index.php?page=post&s=list&tags={SAFEBORU_TAG}&pid={page}"
        try:
            resp = session.get(listing_url, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, 'html.parser')
            thumbs = soup.select('span.thumb a')
            for thumb_link in tqdm(thumbs, desc=f'Page {page}', leave=False):
                post_href = thumb_link.get('href')
                if not post_href:
                    continue
                post_url = urljoin(SAFEBORU_BASE, post_href)
                try:
                    post_resp = session.get(post_url, timeout=10)
                    post_resp.raise_for_status()
                    post_soup = BeautifulSoup(post_resp.text, 'html.parser')
                    img_tag = post_soup.find('img', id='image')
                    if not img_tag:
                        continue
                    full_url = img_tag.get('src')
                    if full_url.startswith('//'):
                        full_url = 'https:' + full_url
                    elif full_url.startswith('/'):
                        full_url = urljoin(SAFEBORU_BASE, full_url)
                    # logging.info(f'Downloading: {full_url}')
                    download_image(full_url, OUTPUT_DIR_SAFEBORU)
                except Exception as e:
                    logging.error(f'Error fetching post {post_url}: {e}')
                time.sleep(random.uniform(0.5, 1.5))
        except Exception as e:
            logging.error(f'Error parsing Safebooru page {page}: {e}')
        time.sleep(random.uniform(1, 2))

if __name__ == '__main__':
    parse_safebooru()
    logging.info('Парсинг завершён.')

