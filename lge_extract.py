import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pydicom
import utils

# Command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--directory", required=True, help="path to input directory")
args = vars(ap.parse_args())

# Load lge dicoms
def load_lge_data(directory):
    """
    Args:
     directory: the path to the folder where dicom images are stored
    Return:
        list of images and indices
    """

    dicomStackList = []
    indicesStackList = []
    dicomSingleList = []
    indicesSingleList = []
    dicomrawList = []
    dicomSrawList = []

    dir_paths = sorted(glob.glob(os.path.join(directory, "*")))
    for dir_path in dir_paths:
        file_paths = sorted(glob.glob(os.path.join(dir_path, "*.dcm")))

        if len(file_paths) > 4:
            folder = os.path.split(dir_path)[1]
            print("\nWorking on ", folder)
            dlist = []
            drlist = []
            for file_path in file_paths:
                imgraw = pydicom.read_file(file_path)
                drlist.append(imgraw)
                img = imgraw.pixel_array
                dlist.append(img)
            dicomrawList.append(drlist)
            dicomSingleList.append(dlist)
            indicesSingleList.append(folder)

        else:
            folder = os.path.split(dir_path)[1]
            print("\nWorking on ", folder)
            dlist = []
            drlist = []
            for file_path in file_paths:
                imgraw = pydicom.read_file(file_path)
                drlist.append(imgraw)
                img = imgraw.pixel_array
                dlist.append(img)
            dicomSrawList.append(drlist)
            dicomStackList.append(dlist)
            indicesStackList.append(folder)

    return dicomrawList, dicomSrawList, dicomSingleList, indicesSingleList, dicomStackList, indicesStackList

dicomrawList, dicomSrawList, dicomSingleList, indicesSingleList, dicomStackList, indicesStackList = load_lge_data(args["directory"])

# Extract LGE dicoms and save as png
# Start with stacked dicoms
for draw, i in zip(dicomStackList, indicesStackList):
    keys = range(len(draw))
    for k in keys:
        dlge = draw[k]
        files = range(len(dlge[:, ]))
        for f in files:
            img = utils.centre_crop(dlge[f])
            dir = f"{i}"
            path = os.path.join('lge_img', dir)
            os.makedirs(path, exist_ok=True)
            plt.imshow(img, cmap='gray')
            plt.savefig(f"lge_img/{dir}/{k}_{f}.png")

for draw, i in zip(dicomSingleList, indicesSingleList):
    keys = range(len(draw))
    for k in keys:
        img = utils.centre_crop(draw[k])
        dir = f"{i}"
        path = os.path.join('lge_img', dir)
        os.makedirs(path, exist_ok=True)
        plt.imshow(img, cmap='gray')
        plt.savefig(f"lge_img/{dir}/{k}.png")
