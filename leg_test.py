import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, MinMaxScaler
import tensorflow as tf
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from keras.utils import to_categorical
from keras.models import model_from_json, load_model
import matplotlib.image as mpimg
from skimage.transform import resize
import cv2
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, roc_curve, classification_report, precision_recall_curve, average_precision_score
import pickle
from random import sample
from sklearn.model_selection import train_test_split
import scipy.stats
import utils
from sklearn.preprocessing import StandardScaler
from keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import cohen_kappa_score

# Command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--directory", required=True, help="path to input directory")
args = vars(ap.parse_args())

# Load dataframe and creating classes
patient_info = pd.read_csv('/Users/ebrahamalskaf/Documents/patient_info.csv')

# Load trained models
# AHA1
(testX, survival_yhat1) = utils.load_lge_data(args["directory"], patient_info, 224)
json_file = open('models/AHA1/aha1_VGG19.json','r')
model1_json = json_file.read()
json_file.close()
model1 = model_from_json(model1_json)
model1.load_weights("models/AHA1/aha1_VGG19_my_model.best.hdf5")
# Predict with aha1 model
preds1 = model1.predict(testX)
# Predict with multilabel models
json_fileMul1 = open('models/MultiLabel/aha_VGG19.json','r')
modelMul1_json = json_fileMul1.read()
json_fileMul1.close()
modelMul1 = model_from_json(modelMul1_json)
modelMul1.load_weights("models/MultiLabel/aha_VGG19_my_model.best.hdf5")
predsMul1 = np.expand_dims(modelMul1.predict(testX)[:,0], axis=1)

# AHA2
json_file = open('models/AHA2/aha2_VGG19.json','r')
model2_json = json_file.read()
json_file.close()
model2 = model_from_json(model2_json)
model2.load_weights("models/AHA2/aha2_VGG19_my_model.best.hdf5")
# Predict with aha1 model
preds2 = model2.predict(testX)
# Predict with multilabel models
predsMul2 = np.expand_dims(modelMul1.predict(testX)[:,1], axis=1)

# AHA3
json_file = open('models/AHA3/aha3_VGG19.json','r')
model3_json = json_file.read()
json_file.close()
model3 = model_from_json(model3_json)
model3.load_weights("models/AHA3/aha3_VGG19_my_model.best.hdf5")
# Predict with aha1 model
preds3 = model3.predict(testX)
# Predict with multilabel models
predsMul3 = np.expand_dims(modelMul1.predict(testX)[:,2], axis=1)

# AHA4
json_file = open('models/AHA4/aha4_VGG19.json','r')
model4_json = json_file.read()
json_file.close()
model4 = model_from_json(model4_json)
model4.load_weights("models/AHA4/aha4_VGG19_my_model.best.hdf5")
# Predict with aha1 model
preds4 = model4.predict(testX)
# Predict with multilabel models
predsMul4 = np.expand_dims(modelMul1.predict(testX)[:,3], axis=1)

# AHA5
json_file = open('models/AHA5/aha5_VGG19.json','r')
model5_json = json_file.read()
json_file.close()
model5 = model_from_json(model5_json)
model5.load_weights("models/AHA5/aha5_VGG19_my_model.best.hdf5")
# Predict with aha1 model
preds5 = model5.predict(testX)
# Predict with multilabel models
predsMul5 = np.expand_dims(modelMul1.predict(testX)[:,4], axis=1)

# AHA6
json_file = open('models/AHA6/aha6_VGG19.json','r')
model6_json = json_file.read()
json_file.close()
model6 = model_from_json(model6_json)
model6.load_weights("models/AHA6/aha6_VGG19_my_model.best.hdf5")
# Predict with aha1 model
preds6 = model6.predict(testX)
# Predict with multilabel models
predsMul6 = np.expand_dims(modelMul1.predict(testX)[:,5], axis=1)

# AHA7
json_file = open('models/AHA7/aha7_VGG19.json','r')
model7_json = json_file.read()
json_file.close()
model7 = model_from_json(model7_json)
model7.load_weights("models/AHA7/aha7_VGG19_my_model.best.hdf5")
# Predict with aha1 model
preds7 = model7.predict(testX)
# Predict with multilabel models
predsMul7 = np.expand_dims(modelMul1.predict(testX)[:,6], axis=1)

# AHA8
json_file = open('models/AHA8/aha8_VGG19.json','r')
model8_json = json_file.read()
json_file.close()
model8 = model_from_json(model8_json)
model8.load_weights("models/AHA8/aha8_VGG19_my_model.best.hdf5")
# Predict with aha1 model
preds8 = model8.predict(testX)
# Predict with multilabel models
predsMul8 = np.expand_dims(modelMul1.predict(testX)[:,7], axis=1)

# AHA9
json_file = open('models/AHA9/aha9_VGG19.json','r')
model9_json = json_file.read()
json_file.close()
model9 = model_from_json(model9_json)
model9.load_weights("models/AHA9/aha9_VGG19_my_model.best.hdf5")
# Predict with aha1 model
preds9 = model9.predict(testX)
# Predict with multilabel models
predsMul9 = np.expand_dims(modelMul1.predict(testX)[:,8], axis=1)

# AHA10
json_file = open('models/AHA10/aha10_VGG19.json','r')
model10_json = json_file.read()
json_file.close()
model10 = model_from_json(model10_json)
model10.load_weights("models/AHA10/aha10_VGG19_my_model.best.hdf5")
# Predict with aha1 model
preds10 = model10.predict(testX)
# Predict with multilabel models
predsMul10 = np.expand_dims(modelMul1.predict(testX)[:,9], axis=1)

# AHA11
json_file = open('models/AHA11/aha11_VGG19.json','r')
model11_json = json_file.read()
json_file.close()
model11 = model_from_json(model11_json)
model11.load_weights("models/AHA11/aha11_VGG19_my_model.best.hdf5")
# Predict with aha1 model
preds11 = model11.predict(testX)
# Predict with multilabel models
predsMul11 = np.expand_dims(modelMul1.predict(testX)[:,10], axis=1)

# AHA12
json_file = open('models/AHA12/aha12_VGG19.json','r')
model12_json = json_file.read()
json_file.close()
model12 = model_from_json(model12_json)
model12.load_weights("models/AHA12/aha12_VGG19_my_model.best.hdf5")
# Predict with aha1 model
preds12 = model12.predict(testX)
# Predict with multilabel models
predsMul12 = np.expand_dims(modelMul1.predict(testX)[:,11], axis=1)

# AHA13
json_file = open('models/AHA13/aha13_VGG19.json','r')
model13_json = json_file.read()
json_file.close()
model13 = model_from_json(model13_json)
model13.load_weights("models/AHA13/aha13_VGG19_my_model.best.hdf5")
# Predict with aha1 model
preds13 = model13.predict(testX)
# Predict with multilabel models
predsMul13 = np.expand_dims(modelMul1.predict(testX)[:,12], axis=1)

# AHA14
json_file = open('models/AHA14/aha14_VGG19.json','r')
model14_json = json_file.read()
json_file.close()
model14 = model_from_json(model14_json)
model14.load_weights("models/AHA14/aha14_VGG19_my_model.best.hdf5")
# Predict with aha1 model
preds14 = model14.predict(testX)
# Predict with multilabel models
predsMul14 = np.expand_dims(modelMul1.predict(testX)[:,13], axis=1)

# AHA15
json_file = open('models/AHA15/aha15_VGG19.json','r')
model15_json = json_file.read()
json_file.close()
model15 = model_from_json(model15_json)
model15.load_weights("models/AHA15/aha15_VGG19_my_model.best.hdf5")
# Predict with aha1 model
preds15 = model15.predict(testX)
# Predict with multilabel models
predsMul15 = np.expand_dims(modelMul1.predict(testX)[:,14], axis=1)

# AHA16
json_file = open('models/AHA16/aha16_VGG19.json','r')
model16_json = json_file.read()
json_file.close()
model16 = model_from_json(model16_json)
model16.load_weights("models/AHA16/aha16_VGG19_my_model.best.hdf5")
# Predict with aha1 model
preds16 = model16.predict(testX)
# Predict with multilabel models
predsMul16 = np.expand_dims(modelMul1.predict(testX)[:,15], axis=1)

# Concatenate predictions and ground truth
predictions = np.concatenate((preds1, preds2, preds3, preds4, preds5, preds6, preds7, preds8, preds9, preds10, preds11, preds12, preds13, preds14, preds15, preds16))
predictionsMul = np.concatenate((predsMul1, predsMul2, predsMul3, predsMul4, predsMul5, predsMul6, predsMul7, predsMul8, predsMul9, predsMul10, predsMul11, predsMul12, predsMul13, predsMul14, predsMul15, predsMul16))
ground_truth = np.concatenate((survival_yhat1, survival_yhat2, survival_yhat3, survival_yhat4, survival_yhat5, survival_yhat6, survival_yhat7, survival_yhat8, survival_yhat9, survival_yhat10, survival_yhat11, survival_yhat12, survival_yhat13, survival_yhat14, survival_yhat15, survival_yhat16))

lad_pred = np.concatenate((preds1, preds2, preds7, preds8, preds13, preds14))
lad_gt = np.concatenate((survival_yhat1, survival_yhat2, survival_yhat7, survival_yhat8, survival_yhat13, survival_yhat14))

rca_pred = np.concatenate((preds3, preds4, preds9, preds10, preds15))
rca_gt = np.concatenate((survival_yhat3, survival_yhat4, survival_yhat9, survival_yhat10, survival_yhat15))

lcx_pred = np.concatenate((preds5, preds6, preds11, preds12, preds16))
lcx_gt = np.concatenate((survival_yhat5, survival_yhat6, survival_yhat11, survival_yhat12, survival_yhat16))

# Plot ROC
fpr, tpr, _ = roc_curve(ground_truth, predictions[:,0])
auc = round(roc_auc_score(ground_truth, predictions[:,0]), 2)
plt.plot(fpr, tpr, label="AHA Cluster Classifiers AUC="+str(auc), color='navy')
fpr, tpr, _ = roc_curve(ground_truth, predictionsMul[:,0])
auc = round(roc_auc_score(ground_truth, predictionsMul[:,0]), 2)
plt.plot(fpr, tpr, label="AHA Multilabel Classifier AUC="+str(auc), color='gold')
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.legend()
plt.xlabel('1 - Specificity')
plt.ylabel('Sensitivity')
plt.grid()
plt.show()

# Calculate Cohen Kappa agreeement
import statsmodels.api as sm

lad_pred = np.array(list(map(lambda x: 0 if x<0.5 else 1, lad_pred)))
lad_gt = np.squeeze(lad_gt)
print(cohen_kappa_score(lad_pred, lad_gt))
rca_pred = np.array(list(map(lambda x: 0 if x<0.5 else 1, rca_pred)))
rca_gt = np.squeeze(rca_gt)
print(cohen_kappa_score(rca_pred, rca_gt))
lcx_pred = np.array(list(map(lambda x: 0 if x<0.5 else 1, lcx_pred)))
lcx_gt = np.squeeze(lcx_gt)
print(cohen_kappa_score(lcx_pred, lcx_gt))




