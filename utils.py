import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import pydicom
import utils
import matplotlib.image as mpimg

# Load lge images
def load_lge_data(directory, df, im_size):
    """
    Args:
     directory: the path to the folder where dicom images are stored
    Return:
        list of images and indices
    """

    images = []
    indices = []

    # Loop over folders and files
    for root, dirs, files in os.walk(directory, topdown=True):
        if '.DS_Store' in files:
            files.remove('.DS_Store')
        for dir in dirs:
            imgList = []
            folder_strip = dir.rstrip('_')
            dir_path = os.path.join(directory, dir)
            files = os.listdir(dir_path)
            l = len(files)
            for file in files:
                file_name = os.path.basename(file)
                file_name = file_name[:file_name.find('.')]

                if file_name in ('0_1', '1_1', '2_1', '3_1'):
                    img = mpimg.imread(os.path.join(dir_path, file))
                    img = resize(img, (im_size, im_size))
                    imgList.append(img)

                elif file_name == f'{l-1}':
                    img = mpimg.imread(os.path.join(dir_path, file))
                    img = resize(img, (im_size, im_size))
                    imgList.append(img)
                elif file_name == f'{l-3}':
                    img = mpimg.imread(os.path.join(dir_path, file))
                    img = resize(img, (im_size, im_size))
                    imgList.append(img)
                elif file_name == f'{l-6}':
                    img = mpimg.imread(os.path.join(dir_path, file))
                    img = resize(img, (im_size, im_size))
                    imgList.append(img)
                elif file_name == f'{l-9}':
                    img = mpimg.imread(os.path.join(dir_path, file))
                    img = resize(img, (im_size, im_size))
                    imgList.append(img)

                else:
                    continue

            images.append(imgList)
            indices.append(int(folder_strip))

    Images = []
    for image_list in images:
        img = cv2.vconcat(image_list)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = resize(gray, (224, 224))
        out = cv2.merge([gray, gray, gray])
        # out = gray[..., np.newaxis]
        Images.append(out)
        #plt.imshow(img)
        #plt.show()

    idx_df = pd.DataFrame(indices, columns=['ID'])
    idx_df['LGE'] = Images
    info_df = pd.merge(df, idx_df, on=['ID'])

    return (info_df)
