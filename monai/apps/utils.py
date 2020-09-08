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

import hashlib
import logging
import os
import shutil
import tarfile
import zipfile
from typing import Optional
from urllib.error import ContentTooShortError, HTTPError, URLError
from urllib.request import Request, urlopen, urlretrieve

from monai.utils import optional_import, progress_bar

gdown, has_gdown = optional_import("gdown", "3.6")


def check_md5(filepath: str, md5_value: Optional[str] = None) -> bool:
    """
    check MD5 signature of specified file.

    Args:
        filepath: path of source file to verify MD5.
        md5_value: expected MD5 value of the file.

    """
    if md5_value is not None:
        md5 = hashlib.md5()
        try:
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    md5.update(chunk)
        except Exception as e:
            print(f"Exception in check_md5: {e}")
            return False
        if md5_value != md5.hexdigest():
            return False
    else:
        print(f"expected MD5 is None, skip MD5 check for file {filepath}.")

    return True


def download_url(url: str, filepath: str, md5_value: Optional[str] = None) -> None:
    """
    Download file from specified URL link, support process bar and MD5 check.

    Args:
        url: source URL link to download file.
        filepath: target filepath to save the downloaded file.
        md5_value: expected MD5 value to validate the downloaded file.
            if None, skip MD5 validation.

    Raises:
        RuntimeError: When the MD5 validation of the ``filepath`` existing file fails.
        RuntimeError: When a network issue or denied permission prevents the
            file download from ``url`` to ``filepath``.
        URLError: See urllib.request.urlretrieve.
        HTTPError: See urllib.request.urlretrieve.
        ContentTooShortError: See urllib.request.urlretrieve.
        IOError: See urllib.request.urlretrieve.
        RuntimeError: When the MD5 validation of the ``url`` downloaded file fails.

    """
    if os.path.exists(filepath):
        if not check_md5(filepath, md5_value):
            raise RuntimeError(f"MD5 check of existing file failed: filepath={filepath}, expected MD5={md5_value}.")
        print(f"file {filepath} exists, skip downloading.")
        return

    if url.startswith("https://drive.google.com"):
        if not has_gdown:
            raise RuntimeError("To download files from Google Drive, please install the gdown dependency.")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        gdown.download(url, filepath, quiet=False)
        if not os.path.exists(filepath):
            raise RuntimeError(
                f"Download of file from {url} to {filepath} failed due to network issue or denied permission."
            )
    elif url.startswith("https://msd-for-monai.s3-us-west-2.amazonaws.com"):
        block_size = 1024 * 1024
        tmp_file_path = filepath + ".part"
        first_byte = os.path.getsize(tmp_file_path) if os.path.exists(tmp_file_path) else 0
        file_size = -1

        try:
            file_size = int(urlopen(url).info().get("Content-Length", -1))
            progress_bar(index=first_byte, count=file_size)

            while first_byte < file_size:
                last_byte = first_byte + block_size if first_byte + block_size < file_size else file_size - 1

                req = Request(url)
                req.headers["Range"] = "bytes=%s-%s" % (first_byte, last_byte)
                data_chunk = urlopen(req, timeout=10).read()
                with open(tmp_file_path, "ab") as f:
                    f.write(data_chunk)
                progress_bar(index=last_byte, count=file_size)
                first_byte = last_byte + 1
        except IOError as e:
            logging.debug("IO Error - %s" % e)
        finally:
            if file_size == os.path.getsize(tmp_file_path):
                if md5_value and not check_md5(tmp_file_path, md5_value):
                    raise Exception("Error validating the file against its MD5 hash")
                shutil.move(tmp_file_path, filepath)
            elif file_size == -1:
                raise Exception("Error getting Content-Length from server: %s" % url)
    else:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        def _process_hook(blocknum: int, blocksize: int, totalsize: int):
            progress_bar(blocknum * blocksize, totalsize, f"Downloading {filepath.split('/')[-1]}:")

        try:
            urlretrieve(url, filepath, reporthook=_process_hook)
            print(f"\ndownloaded file: {filepath}.")
        except (URLError, HTTPError, ContentTooShortError, IOError) as e:
            print(f"download failed from {url} to {filepath}.")
            raise e

    if not check_md5(filepath, md5_value):
        raise RuntimeError(
            f"MD5 check of downloaded file failed: URL={url}, filepath={filepath}, expected MD5={md5_value}."
        )


def extractall(filepath: str, output_dir: str, md5_value: Optional[str] = None) -> None:
    """
    Extract file to the output directory.
    Expected file types are: `zip`, `tar.gz` and `tar`.

    Args:
        filepath: the file path of compressed file.
        output_dir: target directory to save extracted files.
        md5_value: expected MD5 value to validate the compressed file.
            if None, skip MD5 validation.

    Raises:
        RuntimeError: When the MD5 validation of the ``filepath`` compressed file fails.
        ValueError: When the ``filepath`` file extension is not one of [zip", "tar.gz", "tar"].

    """
    target_file = os.path.join(output_dir, os.path.basename(filepath).split(".")[0])
    if os.path.exists(target_file):
        print(f"extracted file {target_file} exists, skip extracting.")
        return
    if not check_md5(filepath, md5_value):
        raise RuntimeError(f"MD5 check of compressed file failed: filepath={filepath}, expected MD5={md5_value}.")

    if filepath.endswith("zip"):
        zip_file = zipfile.ZipFile(filepath)
        zip_file.extractall(output_dir)
        zip_file.close()
    elif filepath.endswith("tar") or filepath.endswith("tar.gz"):
        tar_file = tarfile.open(filepath)
        tar_file.extractall(output_dir)
        tar_file.close()
    else:
        raise ValueError('Unsupported file extension, available options are: ["zip", "tar.gz", "tar"].')


def download_and_extract(url: str, filepath: str, output_dir: str, md5_value: Optional[str] = None) -> None:
    """
    Download file from URL and extract it to the output directory.

    Args:
        url: source URL link to download file.
        filepath: the file path of compressed file.
        output_dir: target directory to save extracted files.
            default is None to save in current directory.
        md5_value: expected MD5 value to validate the downloaded file.
            if None, skip MD5 validation.

    """
    download_url(url=url, filepath=filepath, md5_value=md5_value)
    extractall(filepath=filepath, output_dir=output_dir, md5_value=md5_value)
