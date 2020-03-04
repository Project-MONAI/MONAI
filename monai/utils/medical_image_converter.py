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
import argparse
import numpy as np
import SimpleITK as Sitk

import ai4med.utils.custom_rounding as cp

ALLOWED_SRC_FORMATS = ['.nii', '.nii.gz', '.mhd', '.mha', '.dcm']
ALLOWED_DST_FORMATS = ['.nii', '.nii.gz', '.mhd', '.mha']


def standardize_ext(ext):
    if not ext.startswith('.'):
        ext = '.' + ext

    return ext


def contain_dicom(path):
    items = os.listdir(path)
    for item in items:
        if item.lower().endswith('.dcm'):
            return True

    return False


def get_dicom_dir_list(source_dir):
    dicom_dir_list = []

    # Generate a list of folders that contains dicom files, start with the root folder
    if contain_dicom(source_dir):
        dicom_dir_list.append(source_dir)

    # Go into the subfolders to find dicom files
    for root, dirs, files in os.walk(source_dir, topdown=False):

        for name in dirs:
            full_dir = os.path.join(root, name)
            if contain_dicom(full_dir):
                dicom_dir_list.append(full_dir)

    return dicom_dir_list


def get_image_file_list(source_dir, ext):
    image_file_list = []

    # Go into the subfolders to find dicom files
    for root, dirs, files in os.walk(source_dir, topdown=False):

        for name in files:
            if name.endswith(ext):
                full_path = os.path.join(root, name)
                image_file_list.append(full_path)

    return image_file_list


def resample_image(img, tgt_res):
    # Compute resampling factor, new size and resample using SimpleITK
    factor = np.asarray(img.GetSpacing()) / tgt_res
    new_size = np.asarray(img.GetSize() * factor, dtype=int)
    resampler.SetReferenceImage(img)
    resampler.SetOutputSpacing(tgt_res)
    resampler.SetSize(new_size.tolist())
    return resampler.Execute(img)    


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Convert medical image formats with option to resample image. \
    Supported input formats: .dcm, .nii, .nii.gz, .mha, .mhd. \
    Supported output formats: .nii, .nii.gz, .mha, .mhd')
    parser.add_argument('--dir', '-d', required=True,
                        help='Directory of dicom files to be converted')
    parser.add_argument('--res', '-r', nargs='+', type=float,
                        help='Target resolution. If not provided, dicom resolution will be preserved. '
                             'If only one value is provided, target resolution will be isotrophic.')
    parser.add_argument('--src_ext', '-s', default='.nii',
                        help='Input file format, can be .dcm, .nii, .nii.gz, .mha, .mhd')
    parser.add_argument('--dst_ext', '-e', default='.nii',
                        help='Output file format, can be .nii, .nii.gz, .mha, .mhd')
    parser.add_argument('--output', '-o', default='.',
                        help='Output directory')
    parser.add_argument('--force', '-f', action='store_true',
                        help='Option to force overwriting exsting files')
    parser.add_argument('--label', '-l', action='store_true',
                        help='flag indicating converting label data (nearest neighbor interpolator will be used)')
    parser.add_argument('--first_decimal', '-n', type=int, default=None,
                        help='number to indicate the first decimal digit')
    parser.add_argument('--precision', '-p', type=int, default=3,
                        help='number to indicate the rounding precision')
    args = parser.parse_args()

    target_res = args.res
    src_dir = args.dir
    dst_dir = args.output
    src_ext = standardize_ext(args.src_ext)
    dst_ext = standardize_ext(args.dst_ext)
    convert_label = args.label
    force_write = args.force
    first_decimal_digit = args.first_decimal
    rounding_precision = args.precision

    # Check if extension is supported
    if src_ext.lower() not in ALLOWED_SRC_FORMATS:
        raise ValueError('Unsupported output extension: {}'.format(src_ext))
    if dst_ext.lower() not in ALLOWED_DST_FORMATS:
        raise ValueError('Unsupported output extension: {}'.format(dst_ext))

    # Create data list, image reader, writer and resampler
    if src_ext == '.dcm':
        data_list = get_dicom_dir_list(src_dir)
        reader = Sitk.ImageSeriesReader()
    else:
        data_list = get_image_file_list(src_dir, src_ext)
        reader = Sitk.ImageFileReader()
    writer = Sitk.ImageFileWriter()

    if target_res is not None:
        if len(target_res) == 1:
            target_res *= 3
        resampler = Sitk.ResampleImageFilter()
        if convert_label:
            resampler.SetInterpolator(Sitk.sitkNearestNeighbor)

    # Load dicom series in every folder
    skip_exist = False
    N = len(data_list)
    for i, data_path in enumerate(data_list):

        # Output file is named by the folder name containing the dicom files. Folder structure is preserved.
        rel_path = os.path.relpath(data_path, src_dir)
        if rel_path[-len(src_ext):] == src_ext:
            rel_path = rel_path[0:-len(src_ext)]  # remove src_ext if file ends with provided src_ext
        output_path = os.path.join(dst_dir, rel_path + dst_ext)
        output_dir = os.path.dirname(output_path)

        # If the file already exists, check the force and skip options, and ask the user what to do.
        if os.path.isfile(output_path):
            if skip_exist:
                continue

            if not force_write:
                while True:
                    print('{} already exists!'.format(output_path))
                    overwrite = input('Overwrite? (Y)es, (N)o, (A)lways overwrite, (S)kip all: ')
                    overwrite = overwrite.lower()
                    if overwrite in ['y', 'n', 'a', 's']:
                        break

                if overwrite == 'a':
                    force_write = True
                elif overwrite == 's':
                    skip_exist = True
                    continue
                elif overwrite == 'n':
                    continue

        # Get dicom file names, sorted based on slice location
        if src_ext.endswith('.dcm'):
            print('Converting dicom series {} of {} in {}'.format(i + 1, N, os.path.relpath(data_path, src_dir)))
            dicom_names = reader.GetGDCMSeriesFileNames(data_path, useSeriesDetails=True)
            reader.SetFileNames(dicom_names)
        else:
            print('Converting image {} of {}: {}'.format(i + 1, N, os.path.relpath(data_path, src_dir)))
            reader.SetFileName(data_path)

        try:
            image = reader.Execute()

            if target_res is not None:
                if len(image.GetSize()) > 3:  # time series
                    n_timepoints = image.GetSize()[-1]
                    resampled_timepoints = []
                    for n in range(n_timepoints):
                        if n_timepoints == 1:
                            _img = image
                        else:
                            _img = Sitk.Extract(image, image.GetSize()[:3] + (0,), [0, 0, 0, n])

                        spacing = _img.GetSpacing()
                        # solution 1
                        # new_spacing = [np.float16(item) for item in spacing]
                        # new_spacing = [np.float64(item) for item in new_spacing]
                        # solution 2
                        new_spacing = [cp.customized_rounding(item, first_decimal_digit=first_decimal_digit,
                                                              rounding_precision=rounding_precision) for item in spacing]
                        _img.SetSpacing(new_spacing)

                        _img = resample_image(_img, target_res)
                        resampled_timepoints.append(_img)
                    # join resampled timepoints back together
                    join = Sitk.JoinSeriesImageFilter()
                    image = join.Execute(resampled_timepoints)
                else:  # single time

                    spacing = image.GetSpacing()
                    # solution 1
                    # new_spacing = [np.float16(item) for item in spacing]
                    # new_spacing = [np.float64(item) for item in new_spacing]
                    # solution 2
                    new_spacing = [cp.customized_rounding(item, first_decimal_digit=first_decimal_digit,
                                                          rounding_precision=rounding_precision) for item in spacing]
                    image.SetSpacing(new_spacing)

                    image = resample_image(image, target_res)

            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)
            writer.SetFileName(output_path)
            writer.Execute(image)
        except RuntimeError:
            print("Failed to convert image file: {}. Skipped!".format(data_path))
