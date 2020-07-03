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
from urllib.request import urlretrieve
from urllib.error import URLError
import hashlib
import tarfile
import zipfile
from monai.utils import progress_bar, optional_import

gdown, has_gdown = optional_import("gdown", "3.6")


def check_md5(filepath: str, md5_value: str = None):
    """
    check MD5 signature of specified file.

    Args:
        filepath: path of source file to verify MD5.
        md5_value: expected MD5 value of the file.

    """
    if md5_value is not None:
        md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                md5.update(chunk)
        if md5_value != md5.hexdigest():
            return False
    else:
        print(f"expected MD5 is None, skip MD5 check for file {filepath}.")

    return True


def download_url(url: str, filepath: str, md5_value: str = None):
    """
    Download file from specified URL link, support process bar and MD5 check.

    Args:
        url: source URL link to download file.
        filepath: target filepath to save the downloaded file.
        md5_value: expected MD5 value to validate the downloaded file.
            if None, skip MD5 validation.

    Raises:
        RuntimeError: MD5 check of existing file {filepath} failed, please delete it and try again.
        URLError: See urllib.request.urlopen
        IOError: See urllib.request.urlopen
        RuntimeError: MD5 check of downloaded file failed, URL={url}, filepath={filepath}, expected MD5={md5_value}.

    """
    if os.path.exists(filepath):
        if not check_md5(filepath, md5_value):
            raise RuntimeError(f"MD5 check of existing file {filepath} failed, please delete it and try again.")
        print(f"file {filepath} exists, skip downloading.")
        return
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    if url.startswith("https://drive.google.com"):
        gdown.download(url, filepath, quiet=False)
        if not os.path.exists(filepath):
            raise RuntimeError("download failed due to network issue or permission denied.")
    else:

        def _process_hook(blocknum, blocksize, totalsize):
            progress_bar(blocknum * blocksize, totalsize, f"Downloading {filepath.split('/')[-1]}:")

        try:
            urlretrieve(url, filepath, reporthook=_process_hook)
            print(f"\ndownloaded file: {filepath}.")
        except (URLError, IOError) as e:
            raise e

    if not check_md5(filepath, md5_value):
        raise RuntimeError(
            f"MD5 check of downloaded file failed, \
            URL={url}, filepath={filepath}, expected MD5={md5_value}."
        )


def extractall(filepath: str, output_dir: str, md5_value: str = None):
    """
    Extract file to the output directory.
    Expected file types are: `zip`, `tar.gz` and `tar`.

    Args:
        filepath: the file path of compressed file.
        output_dir: target directory to save extracted files.
        md5_value: expected MD5 value to validate the compressed file.
            if None, skip MD5 validation.

    Raises:
        RuntimeError: MD5 check of compressed file {filepath} failed.
        TypeError: unsupported compressed file type.

    """
    target_file = os.path.join(output_dir, os.path.basename(filepath).split(".")[0])
    if os.path.exists(target_file):
        print(f"extracted file {target_file} exists, skip extracting.")
        return
    if not check_md5(filepath, md5_value):
        raise RuntimeError(f"MD5 check of compressed file {filepath} failed.")

    if filepath.endswith("zip"):
        zip_file = zipfile.ZipFile(filepath)
        zip_file.extractall(output_dir)
        zip_file.close()
    elif filepath.endswith("tar") or filepath.endswith("tar.gz"):
        tar_file = tarfile.open(filepath)
        tar_file.extractall(output_dir)
        tar_file.close()
    else:
        raise TypeError("unsupported compressed file type.")


def download_and_extract(url: str, filepath: str, output_dir: str, md5_value: str = None):
    """
    Download file from URL and extract it to the output directory.

    Args:
        url: source URL link to download file.
        filepath: the file path of compressed file.
        output_dir: target directory to save extracted files.
            defaut is None to save in current directory.
        md5_value: expected MD5 value to validate the downloaded file.
            if None, skip MD5 validation.

    """
    download_url(url=url, filepath=filepath, md5_value=md5_value)
    extractall(filepath=filepath, output_dir=output_dir, md5_value=md5_value)
