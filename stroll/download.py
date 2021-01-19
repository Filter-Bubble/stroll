import argparse
import logging
from os import mkdir
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


def download_srl_model(datapath='models', name_model=None, name_fasttext=None):
    datapath = Path(datapath)

    if not datapath.exists():
        mkdir(datapath)

    if name_model:
        # explicitly named model
        fname_model = datapath / name_model
    else:
        # default model
        fname_model = datapath / 'srl.pt'

        if not fname_model.exists():
            # download if not found
            logger.info('Downloading srl.pt')
            url = 'https://surfdrive.surf.nl/files/index.php/s/kOgUm0oEpmx5HiZ/download'
            urllib.request.urlretrieve(url, fname_model)
        else:
            logger.info('srl.pt found')

    if name_fasttext:
        # explicitly named fasttext
        fname_fasttext = datapath / name_fasttext
    else:
        # default fasttext
        fname_fasttext = datapath / 'fasttext.model.bin'

        if not fname_fasttext.exists():
            # download if not found
            logger.info('Downloading fasttext.model.bin')
            url = 'https://surfdrive.surf.nl/files/index.php/s/085yxFcRmn0osMw/download'
            urllib.request.urlretrieve(url, fname_fasttext)
        else:
            logger.info('fasttext.model.bin found')


    return str(fname_fasttext), str(fname_model)


if __name__ == '__main__':
    logger.setLevel(logging.DEBUG)

    args = parser.parse_args()
    download_srl_model(args.path)
