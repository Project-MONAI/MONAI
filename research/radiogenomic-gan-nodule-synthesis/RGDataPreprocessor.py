# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import csv
import os

import numpy as np


def _load_rna_text(data_dir):
    default_rna_filename = "GSE103584_R01_NSCLC_RNAseq.txt"
    full_path = os.path.join(data_dir, default_rna_filename)
    assert os.path.isfile(
        full_path
    ), f"No RNA data {default_rna_filename} file in {data_dir}. Please see readme for data download instructions."
    rows = []
    with open(full_path, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            rows.append(row)  # ignore index value

    cols = rows[0]  # Patient names
    gene_codes = np.asarray(rows[1:])
    # Remove genes that have 'NA' value for patients.
    gene_codes = [seq for seq in gene_codes if not np.isin("NA", seq)]
    patient_codes = np.transpose(gene_codes)  # Move gene codes to cols.

    embeddings = {}
    for idx in range(len(cols)):
        embeddings[cols[idx]] = patient_codes[idx]

    return cols[1:], embeddings


def _load_tcia_images(data_dir, rna_patient_names):
    """
    Crawl the NSCLC Radiogenomics data, grab CT images and segments, return dict.
    """
    patients = []
    nsclc_foldername = "NSCLC Radiogenomics"
    nsclc_dir = os.path.join(data_dir, nsclc_foldername)
    assert os.path.isdir(
        nsclc_dir
    ), f"No TCIA 'NSCLC Radiogenomics' folder in {data_dir}. Please see readme for data download instructions."
    # Select patients with rna data.
    patients_with_rna_data = [patient for patient in os.listdir(nsclc_dir) if np.isin(patient, rna_patient_names)]
    # Build patient datapoint.
    for patient in patients_with_rna_data:
        datapoint = {}
        datapoint["patient"] = patient
        patient_folder = os.path.join(nsclc_dir, patient)  # Patient data.
        patient_scans = os.listdir(patient_folder)  # Each patient has CT and PET scans.
        for scan_dirname in patient_scans:
            # CT scans variate in name, but all PET scans have PET in name.
            if scan_dirname.find("PET") == -1:
                ct_scan = os.path.join(patient_folder, scan_dirname)  # Contains CT image and segment.
                for folder in os.listdir(ct_scan):
                    names = sorted(os.listdir(os.path.join(ct_scan, folder)))
                    names = [fname for fname in names if fname.startswith("1-")]  # Keep 1st image.
                    names = [os.path.join(ct_scan, folder, name) for name in names]
                    if len(names) > 1:
                        datapoint["image"] = names
                    else:
                        datapoint["seg"] = names[0]
                if len(datapoint.keys()) == 3:
                    # ensure name, image, and segmentation
                    patients.append(datapoint)
    return patients


def load_rg_data(data_dir):
    """
    Load CT images, segmentations, and rna embeddings for patients from the NSCLC Radiogenomics dataset.

    Args:
        data_dir: Patch to input data directory with rna sequences text and downloaded patient images/segments.

    Returns:
        List[Dict] of patient datasamples. Example item
        [{
            "patient": "R01-041",
            "image": ["inputdir/NSCLC Radiogenomics/R01-041/ct_scan/image/1-001.dcm", "inputdir/.../1-002.dcm", ...],
            "seg": "inputdir/NSCLC Radiogenomics/R01-041/ct_scan/segmentation/1-1.dcm",
            "rna_embedding": [12.814, 12.231, ...]
        }]
    """
    rna_patient_names, embeddings = _load_rna_text(data_dir)
    patient_data = _load_tcia_images(data_dir, rna_patient_names)
    for datapoint in patient_data:
        patient = datapoint["patient"]
        datapoint["rna_embedding"] = embeddings[patient]
    return patient_data


if __name__ == "__main__":
    data_dir = "/nvdata/NSCLC-Radiogenomics-Fresh"
    print("Data dir: ", data_dir)
    data = load_rg_data(data_dir)
    # Note: We select patient R01-023 where Ziyue did not.
    print("done")
