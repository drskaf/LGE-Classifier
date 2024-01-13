import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, MinMaxScaler
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json, load_model
import matplotlib.image as mpimg
from skimage.transform import resize
import cv2
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, roc_curve, classification_report, precision_recall_curve, average_precision_score, precision_score, recall_score
import pickle
from random import sample
from sklearn.model_selection import train_test_split
import utils
from sklearn.preprocessing import StandardScaler
import tempfile
from statsmodels.stats.contingency_tables import mcnemar
from mlxtend.evaluate import mcnemar_table, mcnemar
from sklearn.metrics import RocCurveDisplay, auc, confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_selection import mutual_info_regression

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

(df) = utils.load_lge_images('/Users/ebrahamalskaf/Documents/**LGE_CLASSIFICATION**/LGE_test', patient_info, 224)
testX = np.array([x for x in df['LGE']])

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
precision, recall, thresholds = precision_recall_curve(survival_yhatMM, predsMM[:,0])
pred_test_clMM = np.array(list(map(lambda x: 0 if x<np.mean(thresholds) else 1, predsMM)))
print(pred_test_clMM[:5])

prob_outputsMM = {
    "pred": pred_test_clMM,
    "actual_value": survival_yhatMM
}

prob_output_dfMM = pd.DataFrame(prob_outputsMM)
print(prob_output_dfMM.head())

# Evaluate model
print(classification_report(survival_yhatMM, pred_test_clMM))
print('Mortality HNN ROCAUC score:',roc_auc_score(survival_yhatMM, predsMM[:,0]))
print('Mortality HNN Accuracy score:',accuracy_score(survival_yhatMM, pred_test_clMM))
print('Mortality HNN Precision:', np.mean(precision))
print('Mortality HNN recall:', np.mean(recall))
print('Mortality HNN F1 Score:',average_precision_score(survival_yhatMM, predsMM[:,0]))

# plot confusion matrix
cm = confusion_matrix(survival_yhatMM, pred_test_clMM)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# Define mortality predictors
dir = os.listdir('/Users/ebrahamalskaf/Documents/**LGE_CLASSIFICATION**/LGE_images')
dirOne = []
for d in dir:
    if '.DS_Store' in dir:
        dir.remove('.DS_Store')
    d = d.rstrip('_')
    dirOne.append(d)

df1 = pd.DataFrame(dirOne, columns=['index'])
df1['ID'] = df1['index'].astype(int)

# Create dataframe
data = patient_info.merge(df1, on=['ID'])
print(len(data))

trainx, testx = utils.patient_dataset_splitter(data, patient_key='patient_TrustNumber')
y_train = np.array(data['Event_x'])
y_test = np.array(df['Event_x'])
x_train = np.array(process_attributes(data))
x_test = np.array(process_attributes(df))

# fit Linear model
lr_model = LogisticRegression()
lr_model.fit(x_train, y_train)
print('LR Intercept:', lr_model.intercept_)
print('LR Coefficient:', lr_model.coef_)
lr_predict = lr_model.predict(x_test)
print(lr_predict[:5])
lr_preds = lr_model.predict_proba(x_test)[:,1]
print(lr_preds[:5])

# Information gain graph
continuous = data[numerical_col_list]
categorical = data[categorical_col_list]
Xtrain = pd.concat([continuous, categorical], axis=1)
Xtrain = Xtrain.rename(columns={"Chronic_kidney_disease_(disorder)": "CKD", "Essential_hypertension": "HTN",
              "Heart_failure_(disorder)": "Heart Failure", "Smoking_history": "Smoking",
              "Myocardial_infarction_(disorder)": "Myocardial Infarction", "Diabetes_mellitus_(disorder)": "DM",
              "Cerebrovascular_accident_(disorder)": "CVA", "Age_on_20.08.2021_x": "Age", "LVEF_(%)": "LV Ejection Fraction"} )
Ytrain = data['Event_x']
mutual_info = mutual_info_regression(Xtrain, Ytrain)
mutual_info = pd.Series(mutual_info)
mutual_info.index = Xtrain.columns
mi = mutual_info.sort_values(ascending=False)
print("Predictors importance:", mutual_info)
plt.barh(y=mutual_info.index, width=mi)
plt.title("Predictors information gain - mortality")
plt.show()

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

# Evaluate model
print(classification_report(survival_yhatVT, pred_test_clVA))
print('VT HNN ROCAUC score:',roc_auc_score(survival_yhatVT, predsVA[:,0]))
print('VT HNN Accuracy score:',accuracy_score(survival_yhatVT, pred_test_clVA))
print('VT HNN Precision:', np.mean(precision))
print('VT HNN recall:', np.mean(recall))
print('VT HNN F1 Score:',average_precision_score(survival_yhatVT, predsVA[:,0]))

# plot confusion matrix
cm = confusion_matrix(survival_yhatVT, pred_test_clVA)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# Define mortality predictors
dir = os.listdir('/Users/ebrahamalskaf/Documents/**LGE_CLASSIFICATION**/LGE_images')
dirOne = []
for d in dir:
    if '.DS_Store' in dir:
        dir.remove('.DS_Store')
    d = d.rstrip('_')
    dirOne.append(d)

df1 = pd.DataFrame(dirOne, columns=['index'])
df1['ID'] = df1['index'].astype(int)

# Create dataframe
data = patient_info.merge(df1, on=['ID'])
print(len(data))

trainx, testx = utils.patient_dataset_splitter(data, patient_key='patient_TrustNumber')
y_train = np.array(data['VT'])
y_test = np.array(df['VT'])
x_train = np.array(process_attributes(data))
x_test = np.array(process_attributes(df))

# fit Linear model
lr_model = LogisticRegression()
lr_model.fit(x_train, y_train)
print('LR Intercept:', lr_model.intercept_)
print('LR Coefficient:', lr_model.coef_)
lr_predict = lr_model.predict(x_test)
print(lr_predict[:5])
lr_preds = lr_model.predict_proba(x_test)[:,1]
print(lr_preds[:5])

# Information gain graph
continuous = data[numerical_col_list]
categorical = data[categorical_col_list]
Xtrain = pd.concat([continuous, categorical], axis=1)
Xtrain = Xtrain.rename(columns={"Chronic_kidney_disease_(disorder)": "CKD", "Essential_hypertension": "HTN",
              "Heart_failure_(disorder)": "Heart Failure", "Smoking_history": "Smoking",
              "Myocardial_infarction_(disorder)": "Myocardial Infarction", "Diabetes_mellitus_(disorder)": "DM",
              "Cerebrovascular_accident_(disorder)": "CVA", "Age_on_20.08.2021_x": "Age", "LVEF_(%)": "LV Ejection Fraction"} )
Ytrain = data['VT']
mutual_info = mutual_info_regression(Xtrain, Ytrain)
mutual_info = pd.Series(mutual_info)
mutual_info.index = Xtrain.columns
mi = mutual_info.sort_values(ascending=False)
print("Predictors importance:", mutual_info)
plt.barh(y=mutual_info.index, width=mi)
plt.title("Predictors information gain - ventricular arrhythmia")
plt.show()

# plot ROC curves
import scipy.stats
def mean_confidence_interval(data, confidence=0.95):
    n = len(data)
    se = scipy.stats.sem(data)
    m = data
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m-h, m+h

fpr, tpr, _ = roc_curve(survival_yhatMM, predsMM[:,0])
tprs_lower, tprs_upper = mean_confidence_interval(tpr)
auc = round(roc_auc_score(survival_yhatMM, predsMM[:,0]), 2)
plt.plot(fpr, tpr, label="HNN Mortality AUC="+str(auc), color='navy')
plt.fill_between(fpr, tprs_lower,tprs_upper, color='navy', alpha=.20)
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.legend()
plt.xlabel('1 - Specificity')
plt.ylabel('Sensitivity')
plt.title("AUC for mortality HNN")
plt.grid()
plt.show()

fpr, tpr, _ = roc_curve(survival_yhatVT, predsVA[:,0])
tprs_lower, tprs_upper = mean_confidence_interval(tpr)
auc = round(roc_auc_score(survival_yhatVT, predsVA[:,0]), 2)
plt.plot(fpr, tpr, label="HNN Ventricular Arrhythmia AUC="+str(auc), color='red')
plt.fill_between(fpr, tprs_lower,tprs_upper, color='red', alpha=.20)
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.legend()
plt.xlabel('1 - Specificity')
plt.ylabel('Sensitivity')
plt.title("AUC for ventricular arrhythmia HNN")
plt.grid()
plt.show()
