import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import pydicom
import utils


def centre_crop(img, new_width=None, new_height=None):

    width = img.shape[1]
    height = img.shape[0]

    if new_width is None:
        new_width = width//1.5

    if new_height is None:
        new_height = height//1.5

    left = int(np.ceil((width - new_width) / 2))
    right = width - int(np.floor((width - new_width) / 2))

    top = int(np.ceil((height - new_height) / 2))
    bottom = height - int(np.floor((height - new_height) / 2))

    if len(img.shape) == 2:
        centre_cropped_img = img[top:bottom, left:right]
    else:
        centre_cropped_img = img[top:bottom, left:right, ...]

    return centre_cropped_img


def load_lge_data(directory, df, im_size):
    """
    Args:
     directory: the path to the folder where dicom images are stored
    Return:
        list of images and indices
    """

    images = []
    indices = []
    
    dir_paths = sorted(glob.glob(os.path.join(directory, "*")))
    for dir_path in dir_paths:
        file_paths = sorted(glob.glob(os.path.join(dir_path, "*.dcm")))

        if len(file_paths) > 4:
            folder = os.path.split(dir_path)[1]
            folder_strip = folder.rstrip('_')
            print("\nWorking on ", folder)
            dlist = []
            for file_path in file_paths:
                img = pydicom.read_file(file_path)
                img = img.pixel_array
                img = centre_crop(img)
                img = resize(img, (im_size, im_size))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                dlist.append(img)
            imgStack = np.stack(dlist, axis=2)
            images.append(imgStack)
            indices.append(folder_strip)

        else:
            folder = os.path.split(dir_path)[1]
            folder_strip = folder.rstrip('_')
            print("\nWorking on ", folder)
            dlist = []
            for i in file_paths[0:]:
                # Read stacked dicom and add to list
                img = pydicom.read_file(os.path.join(dir_path, i), force=True)
                img = img.pixel_array
                img = centre_crop(img)
                img = resize(img, (im_size, im_size))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                dlist.append(img)
            imgStack = np.stack(dlist, axis=2)
            images.append(imgStack)
            indices.append(folder_strip)
            
    idx_df = pd.DataFrame(indices, columns=['ID'])
    info_df = pd.merge(df, idx_df, on=['ID'])
    info_df['LGE'] = images

    return (info_df)
