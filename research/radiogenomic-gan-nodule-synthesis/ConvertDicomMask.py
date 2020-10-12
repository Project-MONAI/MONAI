#!/usr/bin/env python

import ast
import os

import nibabel as nib
import numpy as np
import pydicom
from pydicom.data import get_testdata_files


def findTag(dataset, tag, range_z):
    for data_element in dataset:
        if data_element.VR == "SQ":
            for sequence_item in data_element.value:
                findTag(sequence_item, tag, range_z)
        else:
            if data_element.name in tag:
                repr_value = repr(data_element.value)
                range_z.append(repr_value)
                # print("{0:s} = {1:s}".format(data_element.name, repr_value))
                # print(range_z)
    return range_z


def convert_one_pair(imgFile, mskFile, outFile):
    if os.path.isfile(mskFile):
            i = 121
            data = pydicom.dcmread(mskFile)
            seg = data.pixel_array
            seg = np.swapaxes(seg, 0, 2)

            image = nib.load(imgFile)
            affine = image.affine
            img = image.dataobj

            if np.shape(seg) == np.shape(img):
                nib.save(nib.Nifti1Image(seg.astype(np.uint8), affine), outFile)
            elif i == 83:
                segOut = np.zeros(np.shape(img))
                segOut[: seg.shape[0], : seg.shape[1], : seg.shape[2]] = seg
                nib.save(nib.Nifti1Image(segOut.astype(np.uint8), affine), outFile)
            else:
                tag = ["Image Position (Patient)"]
                range_z = findTag(data, tag, [])
                print("Case", i, "different in dimension")
                print("Seg Size \n", np.shape(seg))
                print("Img Size \n", np.shape(img))
                print("Matrix \n", affine)

                spaZ = affine[2][2]
                oriZ = affine[2][3]
                if spaZ < 0:
                    spaZ = -spaZ
                    oriZ = -oriZ

                startIdx = range_z[0]
                startIdx = float(ast.literal_eval(startIdx)[2])
                startIdx = int((startIdx - oriZ) / spaZ)

                endIdx = range_z[len(range_z) - 1]
                endIdx = float(ast.literal_eval(endIdx)[2])
                endIdx = int((endIdx - oriZ) / spaZ)

                print(startIdx, endIdx)

                segOut = np.zeros(np.shape(img))
                segOut[: seg.shape[0], : seg.shape[1], startIdx : (startIdx + np.shape(seg)[2])] = seg
                nib.save(nib.Nifti1Image(segOut.astype(np.uint8), affine), outFile)


def processFolder():
    ct = 0
    for i in range(98, 163):
        mskFile = "/media/ziyue/Research/Data/NSCLC_Radiogenomics/ImagesTrain/R01-%03d/Mask/000000.dcm" % (i)
        imgFile = "/media/ziyue/Research/Data/NSCLC_Radiogenomics/ImagesTrain/R01-%03d/image.nii.gz" % (i)
        outFile = "/media/ziyue/Research/Data/NSCLC_Radiogenomics/ImagesTrain/R01-%03d/mask.nii.gz" % (i)
        if os.path.isfile(mskFile):
            ct = ct + 1
            data = pydicom.dcmread(mskFile)
            seg = data.pixel_array
            seg = np.swapaxes(seg, 0, 2)

            image = nib.load(imgFile)
            affine = image.affine
            img = image.dataobj

            if np.shape(seg) == np.shape(img):
                nib.save(nib.Nifti1Image(seg.astype(np.uint8), affine), outFile)
            elif i == 83:
                segOut = np.zeros(np.shape(img))
                segOut[: seg.shape[0], : seg.shape[1], : seg.shape[2]] = seg
                nib.save(nib.Nifti1Image(segOut.astype(np.uint8), affine), outFile)
            else:
                tag = ["Image Position (Patient)"]
                range_z = findTag(data, tag, [])
                print("Case", i, "different in dimension")
                print("Seg Size \n", np.shape(seg))
                print("Img Size \n", np.shape(img))
                print("Matrix \n", affine)

                spaZ = affine[2][2]
                oriZ = affine[2][3]
                if spaZ < 0:
                    spaZ = -spaZ
                    oriZ = -oriZ

                startIdx = range_z[0]
                startIdx = float(ast.literal_eval(startIdx)[2])
                startIdx = int((startIdx - oriZ) / spaZ)

                endIdx = range_z[len(range_z) - 1]
                endIdx = float(ast.literal_eval(endIdx)[2])
                endIdx = int((endIdx - oriZ) / spaZ)

                print(startIdx, endIdx)

                segOut = np.zeros(np.shape(img))
                segOut[: seg.shape[0], : seg.shape[1], startIdx : (startIdx + np.shape(seg)[2])] = seg
                nib.save(nib.Nifti1Image(segOut.astype(np.uint8), affine), outFile)

            print("Total image processed: ", ct)


if __name__ == "__main__":
    # processFolder()
    imgfile = '/nvdata/NSCLC-Radiogenomics-Fresh/test/120_chest.nii.gz'
    segfile = '/nvdata/NSCLC-Radiogenomics-Fresh/test/120_seg.dcm'
    outfile = '/nvdata/NSCLC-Radiogenomics-Fresh/test/120_mask_b.nii.gz'
    convert_one_pair(imgfile, segfile, outfile)