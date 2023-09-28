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
ap.add_argument("-d", "--directory", required=False, help="path to input directory")
args = vars(ap.parse_args())

# Load dataframe and creating classes
patient_info = pd.read_csv('/Users/ebrahamalskaf/Documents/patient_info.csv')
patient_info['Ventricular_fibrillation_(disorder)'] = patient_info['Ventricular_fibrillation_(disorder)'].astype(int)
patient_info['Ventricular_tachycardia_(disorder)'] = patient_info['Ventricular_tachycardia_(disorder)'].astype(int)
patient_info['VT'] = patient_info[['Ventricular_fibrillation_(disorder)', 'Ventricular_tachycardia_(disorder)']].apply(lambda x:'{}'.format(np.max(x)), axis=1)
patient_info['VT'] = patient_info['VT'].astype(int)
aha_list = ['LGE_basal anterior','LGE_basal anteroseptum','LGE_basal inferoseptum','LGE_basal inferior'
                        ,'LGE_basal inferolateral', 'LGE_basal anterolateral','LGE_mid anterior','LGE_mid anteroseptum','LGE_mid inferoseptum','LGE_mid inferior',
                                       'LGE_mid inferolateral','LGE_mid anterolateral','LGE_apical anterior', 'LGE_apical septum','LGE_apical inferior','LGE_apical lateral', 'True_apex_x']
for c in aha_list:
    patient_info[c] = patient_info[c].replace(to_replace=2, value=1)
categorical_col_list = ['Chronic_kidney_disease_(disorder)','Essential_hypertension', 'Gender', 'Heart_failure_(disorder)', 'Smoking_history',
'Dyslipidaemia', 'Myocardial_infarction_(disorder)', 'Diabetes_mellitus_(disorder)', 'Cerebrovascular_accident_(disorder)']
numerical_col_list= ['Age_on_20.08.2021_x', 'LVEF_(%)']

def process_attributes(df):
    continuous = numerical_col_list
    categorical = categorical_col_list
    cs = MinMaxScaler()
    testContinuous = cs.fit_transform(df[continuous])

    # One-hot encode categorical data
    catBinarizer = LabelBinarizer().fit(df[categorical])
    testCategorical = catBinarizer.transform(df[categorical])

    # Construct our training and testing data points by concatenating
    # the categorical features with the continous features
    testX = np.hstack([testCategorical, testContinuous])

    return (testX)

# Load trained models
# AHA1
(df) = utils.load_lge_images('/Users/ebrahamalskaf/Documents/test1', patient_info, 224)
testX = np.array([x for x in df['LGE']])
survival_yhat1 = np.array(df['LGE_basal anterior'])
json_file = open('models/AHA1/aha1.json','r')
model1_json = json_file.read()
json_file.close()
model1 = model_from_json(model1_json)
model1.load_weights("models/AHA1/aha1_my_model.best.hdf5")
# Predict with aha1 model
preds1 = model1.predict(testX)
# Predict with multilabel models
json_fileMul = open('models/MultiLabel/multilabel_aha.json','r')
modelMul_json = json_fileMul.read()
json_fileMul.close()
modelMul = model_from_json(modelMul_json)
modelMul.load_weights("models/MultiLabel/multilabel_aha_my_model.best.hdf5")
predsMul1 = np.expand_dims(modelMul.predict(testX)[:,0], axis=1)

# AHA2
json_file = open('models/AHA2/aha2.json','r')
model2_json = json_file.read()
json_file.close()
model2 = model_from_json(model2_json)
model2.load_weights("models/AHA2/aha2_my_model.best.hdf5")
# Predict with aha1 model
preds2 = model2.predict(testX)
# Predict with multilabel models
survival_yhat2 = np.array(df['LGE_basal anteroseptum'])
predsMul2 = np.expand_dims(modelMul.predict(testX)[:,1], axis=1)

# AHA3
json_file = open('models/AHA3/aha3.json','r')
model3_json = json_file.read()
json_file.close()
model3 = model_from_json(model3_json)
model3.load_weights("models/AHA3/aha3_my_model.best.hdf5")
# Predict with aha1 model
preds3 = model3.predict(testX)
# Predict with multilabel models
survival_yhat3 = np.array(df['LGE_basal inferoseptum'])
predsMul3 = np.expand_dims(modelMul.predict(testX)[:,2], axis=1)

# AHA4
json_file = open('models/AHA4/aha4.json','r')
model4_json = json_file.read()
json_file.close()
model4 = model_from_json(model4_json)
model4.load_weights("models/AHA4/aha4_my_model.best.hdf5")
# Predict with aha1 model
preds4 = model4.predict(testX)
# Predict with multilabel models
survival_yhat4 = np.array(df['LGE_basal inferior'])
predsMul4 = np.expand_dims(modelMul.predict(testX)[:,3], axis=1)

#fpr, tpr, _ = roc_curve(survival_yhat4, preds4[:,0])
#auc = round(roc_auc_score(survival_yhat4, preds4[:,0]), 2)
#plt.plot(fpr, tpr, label="AHA Cluster Classifiers AUC="+str(auc), color='navy')
#fpr, tpr, _ = roc_curve(survival_yhat4, predsMul4[:,0])
#auc = round(roc_auc_score(survival_yhat4, predsMul4[:,0]), 2)
#plt.plot(fpr, tpr, label="AHA Multilabel Classifier AUC="+str(auc), color='gold')
#plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
#plt.legend()
#plt.xlabel('1 - Specificity')
#plt.ylabel('Sensitivity')
#plt.grid()
#plt.show()

# AHA5
json_file = open('models/AHA5/aha5.json','r')
model5_json = json_file.read()
json_file.close()
model5 = model_from_json(model5_json)
model5.load_weights("models/AHA5/aha5_my_model.best.hdf5")
# Predict with aha1 model
preds5 = model5.predict(testX)
# Predict with multilabel models
survival_yhat5 = np.array(df['LGE_basal inferolateral'])
predsMul5 = np.expand_dims(modelMul.predict(testX)[:,4], axis=1)

# AHA6
json_file = open('models/AHA6/aha6.json','r')
model6_json = json_file.read()
json_file.close()
model6 = model_from_json(model6_json)
model6.load_weights("models/AHA6/aha6_my_model.best.hdf5")
# Predict with aha1 model
preds6 = model6.predict(testX)
# Predict with multilabel models
survival_yhat6 = np.array(df['LGE_basal anterolateral'])
predsMul6 = np.expand_dims(modelMul.predict(testX)[:,5], axis=1)

# AHA7
json_file = open('models/AHA7/aha7.json','r')
model7_json = json_file.read()
json_file.close()
model7 = model_from_json(model7_json)
model7.load_weights("models/AHA7/aha7_my_model.best.hdf5")
# Predict with aha1 model
preds7 = model7.predict(testX)
# Predict with multilabel models
survival_yhat7 = np.array(df['LGE_mid anterior'])
predsMul7 = np.expand_dims(modelMul.predict(testX)[:,6], axis=1)

# AHA8
json_file = open('models/AHA8/aha8.json','r')
model8_json = json_file.read()
json_file.close()
model8 = model_from_json(model8_json)
model8.load_weights("models/AHA8/aha8_my_model.best.hdf5")
# Predict with aha1 model
preds8 = model8.predict(testX)
# Predict with multilabel models
survival_yhat8 = np.array(df['LGE_mid anteroseptum'])
predsMul8 = np.expand_dims(modelMul.predict(testX)[:,7], axis=1)

# AHA9
json_file = open('models/AHA9/aha9.json','r')
model9_json = json_file.read()
json_file.close()
model9 = model_from_json(model9_json)
model9.load_weights("models/AHA9/aha9_my_model.best.hdf5")
# Predict with aha1 model
preds9 = model9.predict(testX)
# Predict with multilabel models
survival_yhat9 = np.array(df['LGE_mid inferoseptum'])
predsMul9 = np.expand_dims(modelMul.predict(testX)[:,8], axis=1)

# AHA10
json_file = open('models/AHA10/aha10.json','r')
model10_json = json_file.read()
json_file.close()
model10 = model_from_json(model10_json)
model10.load_weights("models/AHA10/aha10_my_model.best.hdf5")
# Predict with aha1 model
preds10 = model10.predict(testX)
# Predict with multilabel models
survival_yhat10 = np.array(df['LGE_mid inferior'])
predsMul10 = np.expand_dims(modelMul.predict(testX)[:,9], axis=1)

# AHA11
json_file = open('models/AHA11/aha11.json','r')
model11_json = json_file.read()
json_file.close()
model11 = model_from_json(model11_json)
model11.load_weights("models/AHA11/aha11_my_model.best.hdf5")
# Predict with aha1 model
preds11 = model11.predict(testX)
# Predict with multilabel models
survival_yhat11 = np.array(df['LGE_mid inferolateral'])
predsMul11 = np.expand_dims(modelMul.predict(testX)[:,10], axis=1)

# AHA12
json_file = open('models/AHA12/aha12.json','r')
model12_json = json_file.read()
json_file.close()
model12 = model_from_json(model12_json)
model12.load_weights("models/AHA12/aha12_my_model.best.hdf5")
# Predict with aha1 model
preds12 = model12.predict(testX)
# Predict with multilabel models
survival_yhat12 = np.array(df['LGE_mid anterolateral'])
predsMul12 = np.expand_dims(modelMul.predict(testX)[:,11], axis=1)

# AHA13
json_file = open('models/AHA13/aha13.json','r')
model13_json = json_file.read()
json_file.close()
model13 = model_from_json(model13_json)
model13.load_weights("models/AHA13/aha13_my_model.best.hdf5")
# Predict with aha1 model
preds13 = model13.predict(testX)
# Predict with multilabel models
survival_yhat13 = np.array(df['LGE_apical anterior'])
predsMul13 = np.expand_dims(modelMul.predict(testX)[:,12], axis=1)

# AHA14
json_file = open('models/AHA14/aha14.json','r')
model14_json = json_file.read()
json_file.close()
model14 = model_from_json(model14_json)
model14.load_weights("models/AHA14/aha14_my_model.best.hdf5")
# Predict with aha1 model
preds14 = model14.predict(testX)
# Predict with multilabel models
survival_yhat14 = np.array(df['LGE_apical septum'])
predsMul14 = np.expand_dims(modelMul.predict(testX)[:,13], axis=1)

# AHA15
json_file = open('models/AHA15/aha15.json','r')
model15_json = json_file.read()
json_file.close()
model15 = model_from_json(model15_json)
model15.load_weights("models/AHA15/aha15_my_model.best.hdf5")
# Predict with aha1 model
preds15 = model15.predict(testX)
# Predict with multilabel models
survival_yhat15 = np.array(df['LGE_apical inferior'])
predsMul15 = np.expand_dims(modelMul.predict(testX)[:,14], axis=1)

# AHA16
json_file = open('models/AHA16/aha16.json','r')
model16_json = json_file.read()
json_file.close()
model16 = model_from_json(model16_json)
model16.load_weights("models/AHA16/aha16_my_model.best.hdf5")
# Predict with aha1 model
preds16 = model16.predict(testX)
# Predict with multilabel models
survival_yhat16 = np.array(df['LGE_apical lateral'])
predsMul16 = np.expand_dims(modelMul.predict(testX)[:,15], axis=1)

# AHA17
json_file = open('models/AHA17/aha17.json','r')
model17_json = json_file.read()
json_file.close()
model17 = model_from_json(model17_json)
model17.load_weights("models/AHA17/aha17_my_model.best.hdf5")
# Predict with aha1 model
preds17 = model17.predict(testX)
# Predict with multilabel models
survival_yhat17 = np.array(df['True_apex_x'])
predsMul17 = np.expand_dims(modelMul.predict(testX)[:,16], axis=1)

# Concatenate predictions and ground truth
predictions = np.concatenate((preds1, preds2, preds3, preds4, preds5, preds6, preds7, preds8, preds9, preds10, preds11, preds12, preds13, preds14, preds15, preds16, preds17))
predictionsMul = np.concatenate((predsMul1, predsMul2, predsMul3, predsMul4, predsMul5, predsMul6, predsMul7, predsMul8, predsMul9, predsMul10, predsMul11, predsMul12, predsMul13, predsMul14, predsMul15, predsMul16, predsMul17))
ground_truth = np.concatenate((survival_yhat1, survival_yhat2, survival_yhat3, survival_yhat4, survival_yhat5, survival_yhat6, survival_yhat7, survival_yhat8, survival_yhat9, survival_yhat10, survival_yhat11, survival_yhat12, survival_yhat13, survival_yhat14, survival_yhat15, survival_yhat16, survival_yhat17))

#lad_pred = np.concatenate((preds1, preds2, preds7, preds8, preds13, preds14))
#lad_gt = np.concatenate((survival_yhat1, survival_yhat2, survival_yhat7, survival_yhat8, survival_yhat13, survival_yhat14))

#rca_pred = np.concatenate((preds3, preds4, preds9, preds10, preds15))
#rca_gt = np.concatenate((survival_yhat3, survival_yhat4, survival_yhat9, survival_yhat10, survival_yhat15))

#lcx_pred = np.concatenate((preds5, preds6, preds11, preds12, preds16))
#lcx_gt = np.concatenate((survival_yhat5, survival_yhat6, survival_yhat11, survival_yhat12, survival_yhat16))

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
# Calculate Cohen Kappa agreeement
import statsmodels.api as sm

predictions = np.array(list(map(lambda x: 0 if x<0.5 else 1, predictionsMul)))
ground_truth = np.squeeze(ground_truth)
print('Cohen Kappa Score:', cohen_kappa_score(predictions, ground_truth))

#lad_pred = np.array(list(map(lambda x: 0 if x<0.5 else 1, lad_pred)))
#lad_gt = np.squeeze(lad_gt)
#print(cohen_kappa_score(lad_pred, lad_gt))
#rca_pred = np.array(list(map(lambda x: 0 if x<0.5 else 1, rca_pred)))
#rca_gt = np.squeeze(rca_gt)
#print(cohen_kappa_score(rca_pred, rca_gt))
#lcx_pred = np.array(list(map(lambda x: 0 if x<0.5 else 1, lcx_pred)))
#lcx_gt = np.squeeze(lcx_gt)
#print(cohen_kappa_score(lcx_pred, lcx_gt))


# HNN
# Mortality
testImageX = testX
testAttrX = process_attributes(df)
testAttrX = np.array(testAttrX)
survival_yhatMM = np.array(df['Event_x'])

json_file = open('models/Mortality/lge_mortality.json','r')
modelMM_json = json_file.read()
json_file.close()
modelMM = model_from_json(modelMM_json)
modelMM.load_weights('models/Mortality/lge_mortality_my_model.best.hdf5')

# Predict with model
predsMM = modelMM.predict([testAttrX, testImageX])
pred_test_clMM = np.array(list(map(lambda x: 0 if x<0.5 else 1, predsMM)))
print(pred_test_clMM[:5])

prob_outputsMM = {
    "pred": pred_test_clMM,
    "actual_value": survival_yhatMM
}

prob_output_dfMM = pd.DataFrame(prob_outputsMM)
print(prob_output_dfMM.head())

# VT
survival_yhatVT = np.array(df['VT'])
json_file = open('models/VA/lge_VA.json','r')
modelVA_json = json_file.read()
json_file.close()
modelVA = model_from_json(modelVA_json)
modelVA.load_weights('models/VA/lge_VA_my_model.best.hdf5')

# Predict with model
predsVA = modelVA.predict([testAttrX, testImageX])
pred_test_clVA = np.array(list(map(lambda x: 0 if x<0.5 else 1, predsVA)))
print(pred_test_clVA[:5])

prob_outputsVA = {
    "pred": pred_test_clVA,
    "actual_value": survival_yhatVT
}

prob_output_dfVA = pd.DataFrame(prob_outputsVA)
print(prob_output_dfVA.head())

fpr, tpr, _ = roc_curve(survival_yhatMM, predsMM[:,0])
auc = round(roc_auc_score(survival_yhatMM, predsMM[:,0]), 2)
plt.plot(fpr, tpr, label="HNN Mortality AUC="+str(auc), color='blue')
fpr, tpr, _ = roc_curve(survival_yhatVT, predsVA[:,0])
auc = round(roc_auc_score(survival_yhatVT, predsVA[:,0]), 2)
plt.plot(fpr, tpr, label="HNN Ventricular Arrhythmia AUC="+str(auc), color='red')
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.legend()
plt.xlabel('1 - Specificity')
plt.ylabel('Sensitivity')
plt.grid()
plt.show()
