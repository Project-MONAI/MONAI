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

import os
from typing import List, Optional

import monai
from monai.config.type_definitions import PathLike
from monai.utils import optional_import

requests_get, has_requests = optional_import("requests", name="get")
pd, has_pandas = optional_import("pandas")

__all__ = ["get_tcia_metadata", "download_tcia_series_instance", "get_tcia_ref_uid", "match_tcia_ref_uid_in_study"]

BASE_URL = "https://services.cancerimagingarchive.net/nbia-api/services/v1/"


def get_tcia_metadata(query: str, attribute: Optional[str] = None):
    """
    Achieve metadata of a public The Cancer Imaging Archive (TCIA) dataset.

    This function makes use of The National Biomedical Imaging Archive (NBIA) REST APIs to access the metadata
    of objects in the TCIA database.
    Please refer to the following link for more details:
    https://wiki.cancerimagingarchive.net/display/Public/NBIA+Search+REST+API+Guide

    This function relies on `requests` package.

    Args:
        query: queries used to achieve the corresponding metadata. A query is consisted with query name and
            query parameters. The format is like: <query name>?<parameter 1>&<parameter 2>.
            For example: "getSeries?Collection=C4KC-KiTS&Modality=SEG"
            Please refer to the section of Image Metadata APIs in the link mentioned
            above for more details.
        attribute: Achieved metadata may contain multiple attributes, if specifying an attribute name, other attributes
            will be ignored.

    """

    if not has_requests:
        raise ValueError("requests package is necessary, please install it.")
    full_url = f"{BASE_URL}{query}"
    resp = requests_get(full_url)
    resp.raise_for_status()
    metadata_list: List = []
    if len(resp.text) == 0:
        return metadata_list
    for d in resp.json():
        if attribute is not None and attribute in d:
            metadata_list.append(d[attribute])
        else:
            metadata_list.append(d)

    return metadata_list


def download_tcia_series_instance(
    series_uid: str,
    download_dir: PathLike,
    output_dir: PathLike,
    check_md5: bool = False,
    hashes_filename: str = "md5hashes.csv",
    progress: bool = True,
):
    """
    Download a dicom series from a public The Cancer Imaging Archive (TCIA) dataset.
    The downloaded compressed file will be stored in `download_dir`, and the uncompressed folder will be saved
    in `output_dir`.

    Args:
        series_uid: SeriesInstanceUID of a dicom series.
        download_dir: the path to store the downloaded compressed file. The full path of the file is:
            `os.path.join(download_dir, f"{series_uid}.zip")`.
        output_dir: target directory to save extracted dicom series.
        check_md5: whether to download the MD5 hash values as well. If True, will check hash values for all images in
            the downloaded dicom series.
        hashes_filename: file that contains hashes.
        progress: whether to display progress bar.

    """
    query_name = "getImageWithMD5Hash" if check_md5 else "getImage"
    download_url = f"{BASE_URL}{query_name}?SeriesInstanceUID={series_uid}"

    monai.apps.utils.download_and_extract(
        url=download_url,
        filepath=os.path.join(download_dir, f"{series_uid}.zip"),
        output_dir=output_dir,
        progress=progress,
    )
    if check_md5:
        if not has_pandas:
            raise ValueError("pandas package is necessary, please install it.")
        hashes_df = pd.read_csv(os.path.join(output_dir, hashes_filename))
        for dcm, md5hash in hashes_df.values:
            monai.apps.utils.check_hash(filepath=os.path.join(output_dir, dcm), val=md5hash, hash_type="md5")


def get_tcia_ref_uid(ds, find_sop: bool = False, ref_series_uid_tag=(0x0020, 0x000E), ref_sop_uid_tag=(0x0008, 0x1155)):
    """
    Achieve the referenced UID from the referenced Series Sequence for the input pydicom dataset object.
    The referenced UID could be Series Instance UID or SOP Instance UID. The UID will be detected from
    the data element of the input object. If the data element is a sequence, each dataset within the sequence
    will be detected iteratively. The first detected UID will be returned.

    Args:
        ds: a pydicom dataset object.
        find_sop: whether to achieve the referenced SOP Instance UID.
        ref_series_uid_tag: tag of the referenced Series Instance UID.
        ref_sop_uid_tag: tag of the referenced SOP Instance UID.

    """
    ref_uid_tag = ref_sop_uid_tag if find_sop else ref_series_uid_tag
    output = ""

    for elem in ds:
        if elem.VR == "SQ":
            for item in elem:
                output = get_tcia_ref_uid(item, find_sop)
        if elem.tag == ref_uid_tag:
            return elem.value

    return output


def match_tcia_ref_uid_in_study(study_uid, ref_sop_uid):
    """
    Match the SeriesInstanceUID from all series in a study according to the input SOPInstanceUID.

    Args:
        study_uid: StudyInstanceUID.
        ref_sop_uid: SOPInstanceUID.

    """
    series_list = get_tcia_metadata(query=f"getSeries?StudyInstanceUID={study_uid}", attribute="SeriesInstanceUID")
    for series_id in series_list:
        sop_id_list = get_tcia_metadata(
            query=f"getSOPInstanceUIDs?SeriesInstanceUID={series_id}", attribute="SOPInstanceUID"
        )
        if ref_sop_uid in sop_id_list:
            return series_id
    return ""
