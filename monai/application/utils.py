# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import urllib
import hashlib
import tarfile
from monai.utils import process_bar


def download_url(url: str, filepath: str, md5_value: str = None):
    """
    Download file from specified URL link, support process bar and MD5 check.

    Args:
        url: source URL link to download file.
        filepath: target filepath to save the downloaded file.
        md5_value: expected MD5 value to validate the downloaded file.
            if None, skip MD5 validation.

    """

    def _process_hook(blocknum, blocksize, totalsize):
        process_bar(blocknum * blocksize, totalsize)

    try:
        urllib.request.urlretrieve(url, filepath, reporthook=_process_hook)
        print(f"\ndownloaded file: {filepath}.")
    except (urllib.error.URLError, IOError) as e:
        raise e

    if md5_value is not None:
        md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                md5.update(chunk)
        if md5_value != md5.hexdigest():
            raise RuntimeError("MD5 check of downloaded file failed.")


def extractall(filepath: str, output_dir: str = None):
    """
    Extract file to the output directory.

    Args:
        filepath: the file path of compressed file.
        output_dir: target directory to save extracted files.
            defaut is None to save in current directory.
    """
    datafile = tarfile.open(filepath)
    datafile.extractall(output_dir)
    datafile.close()
