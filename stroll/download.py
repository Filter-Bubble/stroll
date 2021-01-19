import argparse
import logging
import sys
from pathlib import Path
import urllib.request


parser = argparse.ArgumentParser(description='Download a trained SRL model for Dutch')
parser.add_argument(
    '--path',
    dest='path',
    default='models',
    help='Path to the models directory'
)


logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())


def download_srl_model(datapath):
    datapath = Path(datapath)
    fname_fasttext = datapath / 'fasttext.model.bin'
    fname_model = datapath / 'srl.pt'
    if not fname_fasttext.exists():
        url = 'https://surfdrive.surf.nl/files/index.php/s/085yxFcRmn0osMw/download'
        urllib.request.urlretrieve(url, fname_fasttext)
    else:
        logger.info('fasttext.model.bin found')

    if not fname_model.exists():
        url = 'https://surfdrive.surf.nl/files/index.php/s/kOgUm0oEpmx5HiZ/download'
        urllib.request.urlretrieve(url, fname_model)
    else:
        logger.info('srl.pt found')

    return fname_fasttext, fname_model


if __name__ == '__main__':
    logger.setLevel(logging.DEBUG)

    args = parser.parse_args()
    download_srl_model(args.path)
