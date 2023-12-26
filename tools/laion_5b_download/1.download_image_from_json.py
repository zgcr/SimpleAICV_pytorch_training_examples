import os
import json

import multiprocessing as mp
import urllib.request

from subprocess import PIPE, Popen
from typing import Tuple
from tqdm import tqdm

download_json_folder_path = r'D:\laion_5b_download\save_jsons'
save_image_folder_path = r'D:\laion_5b_download\save_images'

# max_cpu_num = max(32, mp.cpu_count())

max_cpu_num = 8


def read_json(json_pa):
    with open(json_pa, 'r', encoding='utf-8') as f:
        dic = json.load(f)
    return dic


def sub_download(url, output):
    """Backup download metod timout in 30s.

    Parameters
    ----------
    url : str

    output : str
        output path and file name

    Returns
    -------
    bool
        true if download succeeds
    """
    command = ['wget', '-c', '--timeout=30', '-O', output, url]
    try:
        p = Popen(command, stdout=PIPE, stderr=PIPE, universal_newlines=True)
        out, err = p.communicate()
        rc = p.returncode
        return rc == 0
    except:
        return False


def json_extract(json_file, dir):
    """Extract information from csv.

    Returns
    -------
    Tuple[str, str, str, bool]

    """
    json_path = os.path.join(download_json_folder_path, json_file)
    assert os.path.exists(json_path)

    clip_subset = read_json(json_path)

    for ind, item in tqdm(enumerate(clip_subset)):
        url = item['url']
        id = item['id']
        format = 'jpeg' if 'jpeg' in url else 'jpg'
        output = os.path.join(dir, "laion5b_" + str(id) + '.' + format)
        _exists = os.path.exists(output)
        yield ind, url, output, _exists


def url2file(args):
    """url to files with 2 attempts.

    Parameters
    ----------
    args
        arguments from the output of "csv_extract"

    Returns
    -------
    Tuple[int, str, str]
        status, output path/url, and caption
    """

    ind, url, output, _exists = args

    if _exists:
        return -1, output

    try:
        urllib.request.urlretrieve(url, output)
        # print(output)
        return -1, output
    except:
        if sub_download(url, output):
            return -1, output

    return ind, url


if __name__ == '__main__':
    process_pool = mp.Pool(max_cpu_num)

    if not os.path.exists(save_image_folder_path):
        os.makedirs(save_image_folder_path)

    json_list = os.listdir(download_json_folder_path)

    print(f'total json num: {len(json_list)}')

    for json_file in json_list:
        print(json_file)

        name = json_file.split('.')[0]
        dir = os.path.join(save_image_folder_path, name)

        if not os.path.exists(dir):
            os.mkdir(dir)

        PATH, CAPTIONS, ERROR_IND, ERROR_URL, ERROR_CAP = [], [], [], [], []

        print(f'{name} - start')

        process_pool.imap(url2file, json_extract(json_file, dir))

    process_pool.close()
    process_pool.join()
