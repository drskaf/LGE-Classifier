import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import utils
import tf_cnns
import argparse
from sklearn.model_selection import train_test_split
from skimage.transform import resize
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from datetime import datetime
from keras import backend as K
from keras.losses import BinaryCrossentropy, SparseCategoricalCrossentropy, CategoricalCrossentropy
from keras.utils import to_categorical
import cv2
from imblearn.over_sampling import RandomOverSampler, SMOTE
from collections import Counter
from random import sample


# Command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--target", required=False, help="name of the target field")
args = vars(ap.parse_args())

# Set parameters
INPUT_DIM = 224
WIDTH = 224
HEIGHT = 224
DEPTH = 13
BATCH_SIZE = 32
NUM_EPOCHS = 500
N_CLASSES = 17

# Load dataframe and creating classes
patient_info = pd.read_csv('/Users/ebrahamalskaf/Documents/patient_info.csv')

          ### Training apical AHA segments classification ###

# Load images and label them
(df) = utils.load_lge_images('/Users/ebrahamalskaf/Documents/**LGE_CLASSIFICATION**/LGE_images', patient_info, INPUT_DIM)
print(len(df))

#''' Fine tuning step '''

import ssl
from keras import Model, Sequential
from keras.layers import Dropout, Flatten, Dense, BatchNormalization, Activation

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_virtual_device_configuration(gpu, [tf.config.experimental.VirtualDeviceConfiguration
                                                                      (memory_limit=4096)])
# Splitting data
aha_list = ['LGE_basal anterior','LGE_basal anteroseptum','LGE_basal inferoseptum','LGE_basal inferior'
                        ,'LGE_basal inferolateral', 'LGE_basal anterolateral','LGE_mid anterior','LGE_mid anteroseptum','LGE_mid inferoseptum','LGE_mid inferior',
                                       'LGE_mid inferolateral','LGE_mid anterolateral','LGE_apical anterior', 'LGE_apical septum','LGE_apical inferior','LGE_apical lateral', 'True_apex_x']

df['aha_lge'] = df[aha_list].apply(lambda x:'{}'.format(np.sum(x)), axis=1)
df['aha_lge'] = df['aha_lge'].astype(int)
for c in aha_list:
    df[c] = df[c].replace(to_replace=2, value=1)

(df_train, df_valid) = train_test_split(df, train_size=0.8, random_state=42, shuffle=True)

X_train = np.array([x for x in df_train['LGE']])
print(X_train.shape)
X_valid = np.array([x for x in df_valid['LGE']])
print(X_valid.shape)
y_train = np.array(df_train[aha_list])
y_valid = np.array(df_valid[aha_list])
print(y_train[:10])
print(y_valid[:10])

# Data augmentation
aug = ImageDataGenerator(rotation_range=10,
                         width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.1, zoom_range
                         =0.1, horizontal_flip=True, fill_mode="nearest")

v_aug = ImageDataGenerator()

# Initialise the optimiser and model
print("[INFO] compiling model ...")

METRICS = [
    tf.keras.metrics.TruePositives(name='tp'),
    tf.keras.metrics.FalsePositives(name='fp'),
    tf.keras.metrics.TrueNegatives(name='tn'),
    tf.keras.metrics.FalseNegatives(name='fn'),
    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
    tf.keras.metrics.AUC(name='auc'),
    tf.keras.metrics.AUC(name='prc', curve='PR'),  # precision-recall curve
]

Opt = Adam(lr=0.001)
def my_tf_loss_fn(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    return tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, pos_weight=7)
Loss = BinaryCrossentropy()
model = tf_cnns.ResNet((INPUT_DIM, INPUT_DIM, DEPTH), num_layers=50, OUTPUT=N_CLASSES)
model.compile(loss=Loss, optimizer=Opt, metrics=METRICS)
weigth_path = "{}_my_model.best.hdf5".format("models/multilabel_aha")
checkpoint = ModelCheckpoint(weigth_path, monitor='val_prc', save_best_only=True, mode='max', save_weights_only=False)
early = EarlyStopping(monitor='val_prc', mode='max', patience=50)
callbacks_list = [checkpoint]

logdir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=logdir, histogram_freq=1)

# Training the model
print("[INFO] Training the model ...")
history = model.fit_generator(aug.flow(X_train, y_train, batch_size=BATCH_SIZE), validation_data=v_aug.flow(X_valid, y_valid),
                          epochs=NUM_EPOCHS,callbacks=[early, checkpoint, tensorboard_callback],
                           verbose=1)

# summarize history for loss
plt.plot(history.history['prc'])
plt.plot(history.history['val_prc'])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('AHA perfusion classification CNN')
plt.ylabel('prc')
plt.xlabel('epoch')
plt.legend(['train prc', 'validation prc', 'train loss', 'validation loss'], loc='upper right')
plt.show()

# Saving model data
model_json = model.to_json()
with open("models/multilabel_aha.json", "w") as json_file:
    json_file.write(model_json)

K.clear_session()
