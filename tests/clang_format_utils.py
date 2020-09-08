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

# this file is adapted from
# github/pytorch/pytorch/blob/63d62d3e44a0a4ec09d94f30381d49b78cc5b095/tools/clang_format_utils.py

import hashlib
import os
import platform
import stat
import sys
import urllib.error
import urllib.request

# String representing the host platform (e.g. Linux, Darwin).
HOST_PLATFORM = platform.system()

# PyTorch directory root, derived from the location of this file.
PYTORCH_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# This dictionary maps each platform to the S3 object URL for its clang-format binary.
PLATFORM_TO_CF_URL = {
    "Darwin": "https://oss-clang-format.s3.us-east-2.amazonaws.com/mac/clang-format-mojave",
    "Linux": "https://oss-clang-format.s3.us-east-2.amazonaws.com/linux64/clang-format-linux64",
}

# This dictionary maps each platform to a relative path to a file containing its reference hash.
# github/pytorch/pytorch/tree/63d62d3e44a0a4ec09d94f30381d49b78cc5b095/tools/clang_format_hash
PLATFORM_TO_HASH = {
    "Darwin": "020c7f38f14665c2ed82f3e8976c9074c2cfac0a",
    "Linux": "d1365110da598d148d8143a7f2ccfd8bac7df499",
}

# Directory and file paths for the clang-format binary.
CLANG_FORMAT_DIR = os.path.join(PYTORCH_ROOT, ".clang-format-bin")
CLANG_FORMAT_PATH = os.path.join(CLANG_FORMAT_DIR, "clang-format")


def compute_file_sha1(path):
    """Compute the SHA1 hash of a file and return it as a hex string."""
    # If the file doesn't exist, return an empty string.
    if not os.path.exists(path):
        return ""

    hash = hashlib.sha1()

    # Open the file in binary mode and hash it.
    with open(path, "rb") as f:
        for b in f:
            hash.update(b)

    # Return the hash as a hexadecimal string.
    return hash.hexdigest()


def report_download_progress(chunk_number, chunk_size, file_size):
    """
    Pretty printer for file download progress.
    """
    if file_size != -1:
        percent = min(1, (chunk_number * chunk_size) / file_size)
        bar = "#" * int(64 * percent)
        sys.stdout.write("\r0% |{:<64}| {}%".format(bar, int(percent * 100)))


def download_clang_format(path):
    """
    Downloads a clang-format binary appropriate for the host platform and stores it at the given location.
    """
    if HOST_PLATFORM not in PLATFORM_TO_CF_URL:
        print("Unsupported platform: {}".format(HOST_PLATFORM))
        return False

    cf_url = PLATFORM_TO_CF_URL[HOST_PLATFORM]
    filename = os.path.join(path, "clang-format")

    # Try to download clang-format.
    print("Downloading clang-format to {}".format(path))
    try:
        urllib.request.urlretrieve(
            cf_url, filename, reporthook=report_download_progress if sys.stdout.isatty() else None
        )
    except urllib.error.URLError as e:
        print("Error downloading {}: {}".format(filename, str(e)))
        return False
    finally:
        print()

    return True


def get_and_check_clang_format(verbose=False):
    """
    Download a platform-appropriate clang-format binary if one doesn't already exist at the expected location and verify
    that it is the right binary by checking its SHA1 hash against the expected hash.
    """
    if not os.path.exists(CLANG_FORMAT_DIR):
        # If the directory doesn't exist, try to create it.
        try:
            os.mkdir(CLANG_FORMAT_DIR)
        except os.OSError as e:
            print("Unable to create directory for clang-format binary: {}".format(CLANG_FORMAT_DIR))
            return False
        finally:
            if verbose:
                print("Created directory {} for clang-format binary".format(CLANG_FORMAT_DIR))

        # If the directory didn't exist, neither did the binary, so download it.
        ok = download_clang_format(CLANG_FORMAT_DIR)

        if not ok:
            return False
    else:
        # If the directory exists but the binary doesn't, download it.
        if not os.path.exists(CLANG_FORMAT_PATH):
            ok = download_clang_format(CLANG_FORMAT_DIR)

            if not ok:
                return False
        else:
            if verbose:
                print("Found pre-existing clang-format binary, skipping download")

    # Now that the binary is where it should be, hash it.
    actual_bin_hash = compute_file_sha1(CLANG_FORMAT_PATH)

    # If the host platform is not in PLATFORM_TO_HASH, it is unsupported.
    if HOST_PLATFORM not in PLATFORM_TO_HASH:
        print("Unsupported platform: {}".format(HOST_PLATFORM))
        return False

    # Load the reference hash and compare the actual hash to it.
    reference_bin_hash = PLATFORM_TO_HASH[HOST_PLATFORM]

    if verbose:
        print("Reference Hash: {}".format(reference_bin_hash))
        print("Actual Hash: {}".format(actual_bin_hash))

    if reference_bin_hash != actual_bin_hash:
        print("The downloaded binary is not what was expected!")

        # Err on the side of caution and try to delete the downloaded binary.
        try:
            os.unlink(CLANG_FORMAT_PATH)
        except os.OSError as e:
            print("Failed to delete binary: {}".format(str(e)))
            print("Delete this binary as soon as possible and do not execute it!")

        return False
    else:
        # Make sure the binary is executable.
        mode = os.stat(CLANG_FORMAT_PATH).st_mode
        mode |= stat.S_IXUSR
        os.chmod(CLANG_FORMAT_PATH, mode)
        print("Using clang-format located at {}".format(CLANG_FORMAT_PATH))

    return True


if __name__ == "__main__":
    ok = get_and_check_clang_format(True)
    sys.exit(int(not ok))
