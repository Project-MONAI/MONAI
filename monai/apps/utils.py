# Copyright (c) MONAI Consortium
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
import sys
import tarfile
import tempfile
import warnings
import zipfile
from pathlib import Path
from typing import List, TYPE_CHECKING, Optional
from urllib.error import ContentTooShortError, HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import urlretrieve

from monai.config.type_definitions import PathLike
from monai.utils import look_up_option, min_version, optional_import

gdown, has_gdown = optional_import("gdown", "4.4")
requests_get, has_requests = optional_import("requests", name="get")

if TYPE_CHECKING:
    from tqdm import tqdm

    has_tqdm = True
else:
    tqdm, has_tqdm = optional_import("tqdm", "4.47.0", min_version, "tqdm")

__all__ = ["check_hash", "download_url", "extractall", "download_and_extract", "get_logger", "SUPPORTED_HASH_TYPES"]

DEFAULT_FMT = "%(asctime)s - %(levelname)s - %(message)s"
SUPPORTED_HASH_TYPES = {"md5": hashlib.md5, "sha1": hashlib.sha1, "sha256": hashlib.sha256, "sha512": hashlib.sha512}


def get_logger(
    module_name: str = "monai.apps",
    fmt: str = DEFAULT_FMT,
    datefmt: Optional[str] = None,
    logger_handler: Optional[logging.Handler] = None,
):
    """
    Get a `module_name` logger with the specified format and date format.
    By default, the logger will print to `stdout` at the INFO level.
    If `module_name` is `None`, return the root logger.
    `fmt` and `datafmt` are passed to a `logging.Formatter` object
    (https://docs.python.org/3/library/logging.html#formatter-objects).
    `logger_handler` can be used to add an additional handler.
    """
    logger = logging.getLogger(module_name)
    logger.propagate = False
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    if logger_handler is not None:
        logger.addHandler(logger_handler)
    return logger


# apps module-level default logger
logger = get_logger("monai.apps")
__all__.append("logger")


def _basename(p: PathLike) -> str:
    """get the last part of the path (removing the trailing slash if it exists)"""
    sep = os.path.sep + (os.path.altsep or "") + "/ "
    return Path(f"{p}".rstrip(sep)).name


def _download_with_progress(url, filepath, progress: bool = True):
    """
    Retrieve file from `url` to `filepath`, optionally showing a progress bar.
    """
    try:
        if has_tqdm and progress:

            class TqdmUpTo(tqdm):
                """
                Provides `update_to(n)` which uses `tqdm.update(delta_n)`.
                Inspired by the example in https://github.com/tqdm/tqdm.
                """

                def update_to(self, b: int = 1, bsize: int = 1, tsize: Optional[int] = None):
                    """
                    Args:
                        b: number of blocks transferred so far, default: 1.
                        bsize: size of each block (in tqdm units), default: 1.
                        tsize: total size (in tqdm units). if None, remains unchanged.
                    """
                    if tsize is not None:
                        self.total = tsize
                    self.update(b * bsize - self.n)  # will also set self.n = b * bsize

            with TqdmUpTo(unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc=_basename(filepath)) as t:
                urlretrieve(url, filepath, reporthook=t.update_to)
        else:
            if not has_tqdm and progress:
                warnings.warn("tqdm is not installed, will not show the downloading progress bar.")
            urlretrieve(url, filepath)
    except (URLError, HTTPError, ContentTooShortError, OSError) as e:
        logger.error(f"Download failed from {url} to {filepath}.")
        raise e


def check_hash(filepath: PathLike, val: Optional[str] = None, hash_type: str = "md5") -> bool:
    """
    Verify hash signature of specified file.

    Args:
        filepath: path of source file to verify hash value.
        val: expected hash value of the file.
        hash_type: type of hash algorithm to use, default is `"md5"`.
            The supported hash types are `"md5"`, `"sha1"`, `"sha256"`, `"sha512"`.
            See also: :py:data:`monai.apps.utils.SUPPORTED_HASH_TYPES`.

    """
    if val is None:
        logger.info(f"Expected {hash_type} is None, skip {hash_type} check for file {filepath}.")
        return True
    actual_hash_func = look_up_option(hash_type.lower(), SUPPORTED_HASH_TYPES)
    actual_hash = actual_hash_func()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                actual_hash.update(chunk)
    except Exception as e:
        logger.error(f"Exception in check_hash: {e}")
        return False
    if val != actual_hash.hexdigest():
        logger.error(f"check_hash failed {actual_hash.hexdigest()}.")
        return False

    logger.info(f"Verified '{_basename(filepath)}', {hash_type}: {val}.")
    return True


def get_tcia_metadata(query: str, attribute: Optional[str] = None):
    """
    Achieve metadata of a public The Cancer Imaging Archive (TCIA) dataset.

    This function makes use of The National Biomedical Imaging Archive (NBIA) REST APIs to access the metadata
    of objects in the TCIA database.
    Please refer to the following link for more details:
    https://wiki.cancerimagingarchive.net/display/Public/NBIA+Search+REST+API+Guide

    This function relys on `requests` package.

    Args:
        query: queries used to achieve the corresponding metadata. A query is consisted with query name and
            query parameters. The format is like: <query name>?<parameter 1>&<parameter 2>.
            For example: "getSeries?Collection=C4KC-KiTS&Modality=SEG"
            Please refer to the section of Image Metadata APIs in the link mentioned
            above for more details.
        attribute: Achieved metadata may contain multiple attributes, if specifying an attribute name, other attributes
            will be ignored.

    """

    if has_requests:
        baseurl = "https://services.cancerimagingarchive.net/nbia-api/services/v1/"
        full_url = baseurl + query
        resp = requests_get(full_url)
        resp.raise_for_status()
    else:
        raise ValueError("requests package is necessary, please install it.")
    metadata_list: List = []
    if len(resp.text) == 0:
        return metadata_list
    for d in resp.json():
        if attribute is not None and attribute in d:
            metadata_list.append(d[attribute])
        else:
            metadata_list.append(d)

    return metadata_list


def download_url(
    url: str,
    filepath: PathLike = "",
    hash_val: Optional[str] = None,
    hash_type: str = "md5",
    progress: bool = True,
    **gdown_kwargs,
) -> None:
    """
    Download file from specified URL link, support process bar and hash check.

    Args:
        url: source URL link to download file.
        filepath: target filepath to save the downloaded file (including the filename).
            If undefined, `os.path.basename(url)` will be used.
        hash_val: expected hash value to validate the downloaded file.
            if None, skip hash validation.
        hash_type: 'md5' or 'sha1', defaults to 'md5'.
        progress: whether to display a progress bar.
        gdown_kwargs: other args for `gdown` except for the `url`, `output` and `quiet`.
            these args will only be used if download from google drive.
            details of the args of it:
            https://github.com/wkentaro/gdown/blob/main/gdown/download.py

    Raises:
        RuntimeError: When the hash validation of the ``filepath`` existing file fails.
        RuntimeError: When a network issue or denied permission prevents the
            file download from ``url`` to ``filepath``.
        URLError: See urllib.request.urlretrieve.
        HTTPError: See urllib.request.urlretrieve.
        ContentTooShortError: See urllib.request.urlretrieve.
        IOError: See urllib.request.urlretrieve.
        RuntimeError: When the hash validation of the ``url`` downloaded file fails.

    """
    if not filepath:
        filepath = Path(".", _basename(url)).resolve()
        logger.info(f"Default downloading to '{filepath}'")
    filepath = Path(filepath)
    if filepath.exists():
        if not check_hash(filepath, hash_val, hash_type):
            raise RuntimeError(
                f"{hash_type} check of existing file failed: filepath={filepath}, expected {hash_type}={hash_val}."
            )
        logger.info(f"File exists: {filepath}, skipped downloading.")
        return
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_name = Path(tmp_dir, _basename(filepath))
            if urlparse(url).netloc == "drive.google.com":
                if not has_gdown:
                    raise RuntimeError("To download files from Google Drive, please install the gdown dependency.")
                gdown.download(url, f"{tmp_name}", quiet=not progress, **gdown_kwargs)
            else:
                _download_with_progress(url, tmp_name, progress=progress)
            if not tmp_name.exists():
                raise RuntimeError(
                    f"Download of file from {url} to {filepath} failed due to network issue or denied permission."
                )
            file_dir = filepath.parent
            if file_dir:
                os.makedirs(file_dir, exist_ok=True)
            shutil.move(f"{tmp_name}", f"{filepath}")  # copy the downloaded to a user-specified cache.
    except (PermissionError, NotADirectoryError):  # project-monai/monai issue #3613 #3757 for windows
        pass
    logger.info(f"Downloaded: {filepath}")
    if not check_hash(filepath, hash_val, hash_type):
        raise RuntimeError(
            f"{hash_type} check of downloaded file failed: URL={url}, "
            f"filepath={filepath}, expected {hash_type}={hash_val}."
        )


def extractall(
    filepath: PathLike,
    output_dir: PathLike = ".",
    hash_val: Optional[str] = None,
    hash_type: str = "md5",
    file_type: str = "",
    has_base: bool = True,
) -> None:
    """
    Extract file to the output directory.
    Expected file types are: `zip`, `tar.gz` and `tar`.

    Args:
        filepath: the file path of compressed file.
        output_dir: target directory to save extracted files.
        hash_val: expected hash value to validate the compressed file.
            if None, skip hash validation.
        hash_type: 'md5' or 'sha1', defaults to 'md5'.
        file_type: string of file type for decompressing. Leave it empty to infer the type from the filepath basename.
        has_base: whether the extracted files have a base folder. This flag is used when checking if the existing
            folder is a result of `extractall`, if it is, the extraction is skipped. For example, if A.zip is unzipped
            to folder structure `A/*.png`, this flag should be True; if B.zip is unzipped to `*.png`, this flag should
            be False.

    Raises:
        RuntimeError: When the hash validation of the ``filepath`` compressed file fails.
        NotImplementedError: When the ``filepath`` file extension is not one of [zip", "tar.gz", "tar"].

    """
    if has_base:
        # the extracted files will be in this folder
        cache_dir = Path(output_dir, _basename(filepath).split(".")[0])
    else:
        cache_dir = Path(output_dir)
    if cache_dir.exists() and next(cache_dir.iterdir(), None) is not None:
        logger.info(f"Non-empty folder exists in {cache_dir}, skipped extracting.")
        return
    filepath = Path(filepath)
    if hash_val and not check_hash(filepath, hash_val, hash_type):
        raise RuntimeError(
            f"{hash_type} check of compressed file failed: " f"filepath={filepath}, expected {hash_type}={hash_val}."
        )
    logger.info(f"Writing into directory: {output_dir}.")
    _file_type = file_type.lower().strip()
    if filepath.name.endswith("zip") or _file_type == "zip":
        zip_file = zipfile.ZipFile(filepath)
        zip_file.extractall(output_dir)
        zip_file.close()
        return
    if filepath.name.endswith("tar") or filepath.name.endswith("tar.gz") or "tar" in _file_type:
        tar_file = tarfile.open(filepath)
        tar_file.extractall(output_dir)
        tar_file.close()
        return
    raise NotImplementedError(
        f'Unsupported file type, available options are: ["zip", "tar.gz", "tar"]. name={filepath} type={file_type}.'
    )


def download_and_extract(
    url: str,
    filepath: PathLike = "",
    output_dir: PathLike = ".",
    hash_val: Optional[str] = None,
    hash_type: str = "md5",
    file_type: str = "",
    has_base: bool = True,
    progress: bool = True,
) -> None:
    """
    Download file from URL and extract it to the output directory.

    Args:
        url: source URL link to download file.
        filepath: the file path of the downloaded compressed file.
            use this option to keep the directly downloaded compressed file, to avoid further repeated downloads.
        output_dir: target directory to save extracted files.
            default is the current directory.
        hash_val: expected hash value to validate the downloaded file.
            if None, skip hash validation.
        hash_type: 'md5' or 'sha1', defaults to 'md5'.
        file_type: string of file type for decompressing. Leave it empty to infer the type from url's base file name.
        has_base: whether the extracted files have a base folder. This flag is used when checking if the existing
            folder is a result of `extractall`, if it is, the extraction is skipped. For example, if A.zip is unzipped
            to folder structure `A/*.png`, this flag should be True; if B.zip is unzipped to `*.png`, this flag should
            be False.
        progress: whether to display progress bar.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        filename = filepath or Path(tmp_dir, _basename(url)).resolve()
        download_url(url=url, filepath=filename, hash_val=hash_val, hash_type=hash_type, progress=progress)
        extractall(filepath=filename, output_dir=output_dir, file_type=file_type, has_base=has_base)
