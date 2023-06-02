import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import pydicom
import utils
import matplotlib.image as mpimg


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
                mpimg.imread(os.path.join(dir_path, file))
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
                img = mpimg.imread(os.path.join(dir_path, i))
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
