from pathlib import Path
import urllib.request


def download_srl_model(datapath):
    datapath = Path(datapath)
    fname_fasttext = datapath / 'fasttext.model.bin'
    fname_model = datapath / 'srl.pt'
    if not fname_fasttext.exists():
        url = 'https://surfdrive.surf.nl/files/index.php/s/085yxFcRmn0osMw/download'
        urllib.request.urlretrieve(url, fname_fasttext)

    if not fname_model.exists():
        url = 'https://surfdrive.surf.nl/files/index.php/s/kOgUm0oEpmx5HiZ/download'
        urllib.request.urlretrieve(url, fname_model)
    return fname_fasttext, fname_model
