import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import seaborn as sns
    
# Define indices
dirs_1 = os.listdir('/Users/ebrahamalskaf/Documents/**LGE_CLASSIFICATION**/lge_img')
dirs_2 = os.listdir('/Users/ebrahamalskaf/Documents/**LGE_CLASSIFICATION**/lge_test')
dir = dirs_2 + dirs_1
dirs = []
for i in dir:
    d = i.rstrip('_')
    dirs.append(d)
    if '.DS_Store' in dir:
        dir.remove('.DS_Store')
if '.\\20220511_indices.csv' in dir:
    dir.remove('.\\20220511_indices.csv')

# Create dataframe
df = pd.DataFrame(dirs, columns=['index'])
df['ID'] = df['index'].astype(int)
patient_df = pd.read_csv('/Users/ebrahamalskaf/Documents/patient_info.csv')
data = pd.merge(df, patient_df, on=['ID'])
aha_list = ['LGE_basal anterior','LGE_basal anteroseptum','LGE_basal inferoseptum','LGE_basal inferior'
                        ,'LGE_basal inferolateral', 'LGE_basal anterolateral','LGE_mid anterior','LGE_mid anteroseptum','LGE_mid inferoseptum','LGE_mid inferior',
                                       'LGE_mid inferolateral','LGE_mid anterolateral','LGE_apical anterior', 'LGE_apical septum','LGE_apical inferior','LGE_apical lateral']

data_aha = data[aha_list]
data_aha = data_aha.rename(columns={'LGE_basal anterior':'AHA1', 'LGE_basal anteroseptum':'AHA2', 'LGE_basal inferoseptum':'AHA3',
                                    'LGE_basal inferior':'AHA4', 'LGE_basal inferolateral':'AHA5', 'LGE_basal anterolateral':'AHA6',
                                    'LGE_mid anterior':'AHA7', 'LGE_mid anteroseptum':'AHA8', 'LGE_mid inferoseptum':'AHA9',
                                    'LGE_mid inferior':'AHA10', 'LGE_mid inferolateral':'AHA11', 'LGE_mid anterolateral':'AHA12',
                                    'LGE_apical anterior':'AHA13', 'LGE_apical septum':'AHA14', 'LGE_apical inferior':'AHA15',
                                    'LGE_apical lateral':'AHA16'})

# Plot AHA labels
categories = list(data_aha.columns.values)
sns.set(font_scale = 1)
plt.figure(figsize=(15,8))
ax= sns.barplot(x=categories, y=data_aha.iloc[:,:].sum().values)
plt.title("Number of positive LGE cases in each AHA segment", fontsize=18)
plt.ylabel('Number of cases', fontsize=14)
plt.xlabel('AHA segment IDs ', fontsize=14)
#adding the text labels
rects = ax.patches
labels = data_aha.iloc[:,:].sum().values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom', fontsize=18)
plt.show()

rowSums = data_aha.sum(axis=1)
multiLabel_counts = rowSums.value_counts()
multiLabel_counts = multiLabel_counts.iloc[:]
sns.set(font_scale = 1)
plt.figure(figsize=(15,8))
ax = sns.barplot(x=multiLabel_counts.index, y=multiLabel_counts.values)
plt.title("Number of cases of multilabel positive segments ")
plt.ylabel('Number of cases', fontsize=14)
plt.xlabel('Number of positive labels', fontsize=14)
#adding the text labels
for i in ax.containers:
    ax.bar_label(i,)
plt.show()

x = {'1.5T Philips':0, '1.5T Siemens':0, '1.5T':0, '3T':1, '3T Philips':1, '3T Siemens':1}
data['Field_strength_x'] = data['Field_strength_x'].map(x)
data['Field_strength_x'] = data['Field_strength_x'].astype('category')
data['Field_strength_x'] = data['Field_strength_x'].cat.codes
y = {'A':0, 'na':0, 'R': 1, 'A + R': 1}
data['Stress_agent_x'] = data['Stress_agent_x'].map(y)
data['Stress_agent_x'] = data['Stress_agent_x'].astype('category')
data['Stress_agent_x'] = data['Stress_agent_x'].cat.codes
data['Ventricular_tachycardia'] = data['Ventricular_tachycardia_(disorder)'].astype('int')
data['Ventricular_fibrillation'] = data['Ventricular_fibrillation_(disorder)'].astype('int')
data['VT'] = data[['Ventricular_tachycardia','Ventricular_fibrillation']].apply(lambda x:'{}'.format(np.max(x)), axis=1)
data['VT'] = data['VT'].astype('int')
print(len(data[data['VT']==1]))

# Data extraction

# Dividing into age group
age_group1 = data[data['Age_on_20.08.2021_x'] <65]
age_group2 = data[(data['Age_on_20.08.2021_x'] >=65) & (data['Age_on_20.08.2021_x'] <=75)]
age_group3 = data[data['Age_on_20.08.2021_x'] >75]
print('Number of cases < 65 years:\n{}'.format(len(age_group1)))
print('Number of cases 65-75 years:\n{}'.format(len(age_group2)))
print('Number of cases > 75 years:\n{}'.format(len(age_group3)))

# calculating events
event1 = (age_group1[age_group1['Event_x']==1])
event2 = (age_group2[age_group2['Event_x']==1])
event3 = (age_group3[age_group3['Event_x']==1])
print('Number of events in group 1:\n{}'.format(len(event1)))
print('Number of events in group 2:\n{}'.format(len(event2)))
print('Number of events in group 3:\n{}'.format(len(event3)))
print('Percentage group 1: \n{}'.format(len(event1)/len(age_group1)))
print('Percentage group 2: \n{}'.format(len(event2)/len(age_group2)))
print('Percentage group 3: \n{}'.format(len(event3)/len(age_group3)))
print('Kruskal Wallis significance test: \n{}'.format(stats.kruskal((age_group1['Event_x'].values), age_group2['Event_x'].values, age_group3['Event_x'].values)))

# calculating gender
female1 = (age_group1[age_group1['Gender']==0])
female2 = (age_group2[age_group2['Gender']==0])
female3 = (age_group3[age_group3['Gender']==0])
print('Number of female in group 1:\n{}'.format(len(female1)))
print('Number of females in group 2:\n{}'.format(len(female2)))
print('Number of females in group 3:\n{}'.format(len(female3)))
print('Percentage female 1: \n{}'.format(len(female1)/len(age_group1)))
print('Percentage female 2: \n{}'.format(len(female2)/len(age_group2)))
print('Percentage female 3: \n{}'.format(len(female3)/len(age_group3)))

male1 = (age_group1[age_group1['Gender']==1])
male2 = (age_group2[age_group2['Gender']==1])
male3 = (age_group3[age_group3['Gender']==1])
print('Number of males in group 1:\n{}'.format(len(male1)))
print('Number of males in group 2:\n{}'.format(len(male2)))
print('Number of males in group 3:\n{}'.format(len(male3)))
print('Percentage male 1: \n{}'.format(len(male1)/len(age_group1)))
print('Percentage male 2: \n{}'.format(len(male2)/len(age_group2)))
print('Percentage male 3: \n{}'.format(len(male3)/len(age_group3)))
print('Kruskal Wallis significance test: \n{}'.format(stats.kruskal((age_group1['Gender'].values), age_group2['Gender'].values, age_group3['Gender'].values)))

# calculating smoking
smoke1 = (age_group1[age_group1['Smoking_history']==1])
smoke2 = (age_group2[age_group2['Smoking_history']==1])
smoke3 = (age_group3[age_group3['Smoking_history']==1])
print('Number of smoking in group 1:\n{}'.format(len(smoke1)))
print('Number of smoking in group 2:\n{}'.format(len(smoke2)))
print('Number of smoking in group 3:\n{}'.format(len(smoke3)))
print('Percentage smoking 1: \n{}'.format(len(smoke1)/len(age_group1)))
print('Percentage smoking 2: \n{}'.format(len(smoke2)/len(age_group2)))
print('Percentage smoking 3: \n{}'.format(len(smoke3)/len(age_group3)))
print('Kruskal Wallis significance test: \n{}'.format(stats.kruskal((age_group1['Smoking_history'].values), age_group2['Smoking_history'].values, age_group3['Smoking_history'].values)))

# calculating diabetes
dm1 = (age_group1[age_group1['Diabetes_mellitus_(disorder)']==1])
dm2 = (age_group2[age_group2['Diabetes_mellitus_(disorder)']==1])
dm3 = (age_group3[age_group3['Diabetes_mellitus_(disorder)']==1])
print('Number of DM in group 1:\n{}'.format(len(dm1)))
print('Number of DM in group 2:\n{}'.format(len(dm2)))
print('Number of DM in group 3:\n{}'.format(len(dm3)))
print('Percentage DM 1: \n{}'.format(len(dm1)/len(age_group1)))
print('Percentage DM 2: \n{}'.format(len(dm2)/len(age_group2)))
print('Percentage DM 3: \n{}'.format(len(dm3)/len(age_group3)))
print('Kruskal Wallis significance test: \n{}'.format(stats.kruskal((age_group1['Diabetes_mellitus_(disorder)'].values), age_group2['Diabetes_mellitus_(disorder)'].values, age_group3['Diabetes_mellitus_(disorder)'].values)))

# calculating hypertension
htn1 = (age_group1[age_group1['Essential_hypertension']==1])
htn2 = (age_group2[age_group2['Essential_hypertension']==1])
htn3 = (age_group3[age_group3['Essential_hypertension']==1])
print('Number of HTN in group 1:\n{}'.format(len(htn1)))
print('Number of HTN in group 2:\n{}'.format(len(htn2)))
print('Number of HTN in group 3:\n{}'.format(len(htn3)))
print('Percentage HTN 1: \n{}'.format(len(htn1)/len(age_group1)))
print('Percentage HTN 2: \n{}'.format(len(htn2)/len(age_group2)))
print('Percentage HTN 3: \n{}'.format(len(htn3)/len(age_group3)))
print('Kruskal Wallis significance test: \n{}'.format(stats.kruskal((age_group1['Essential_hypertension'].values), age_group2['Essential_hypertension'].values, age_group3['Essential_hypertension'].values)))

# calculating dyslipidaemia
lip1 = (age_group1[age_group1['Dyslipidaemia']==1])
lip2 = (age_group2[age_group2['Dyslipidaemia']==1])
lip3 = (age_group3[age_group3['Dyslipidaemia']==1])
print('Number of Dyslipidaemia in group 1:\n{}'.format(len(lip1)))
print('Number of Dyslipidaemia in group 2:\n{}'.format(len(lip2)))
print('Number of Dyslipidaemia in group 3:\n{}'.format(len(lip3)))
print('Percentage Dyslipidaemia 1: \n{}'.format(len(lip1)/len(age_group1)))
print('Percentage Dyslipidaemia 2: \n{}'.format(len(lip2)/len(age_group2)))
print('Percentage Dyslipidaemia 3: \n{}'.format(len(lip3)/len(age_group3)))
print('Kruskal Wallis significance test: \n{}'.format(stats.kruskal((age_group1['Dyslipidaemia'].values), age_group2['Dyslipidaemia'].values, age_group3['Dyslipidaemia'].values)))

# calculating CVA
cva1 = (age_group1[age_group1['Cerebrovascular_accident_(disorder)']==1])
cva2 = (age_group2[age_group2['Cerebrovascular_accident_(disorder)']==1])
cva3 = (age_group3[age_group3['Cerebrovascular_accident_(disorder)']==1])
print('Number of CVA in group 1:\n{}'.format(len(cva1)))
print('Number of CVA in group 2:\n{}'.format(len(cva2)))
print('Number of CVA in group 3:\n{}'.format(len(cva3)))
print('Percentage CVA 1: \n{}'.format(len(cva1)/len(age_group1)))
print('Percentage CVA 2: \n{}'.format(len(cva2)/len(age_group2)))
print('Percentage CVA 3: \n{}'.format(len(cva3)/len(age_group3)))
print('Kruskal Wallis significance test: \n{}'.format(stats.kruskal((age_group1['Cerebrovascular_accident_(disorder)'].values), age_group2['Cerebrovascular_accident_(disorder)'].values, age_group3['Cerebrovascular_accident_(disorder)'].values)))

# calculating CKD
ckd1 = (age_group1[age_group1['Chronic_kidney_disease_(disorder)']==1])
ckd2 = (age_group2[age_group2['Chronic_kidney_disease_(disorder)']==1])
ckd3 = (age_group3[age_group3['Chronic_kidney_disease_(disorder)']==1])
print('Number of CKD in group 1:\n{}'.format(len(ckd1)))
print('Number of CKD in group 2:\n{}'.format(len(ckd2)))
print('Number of CKD in group 3:\n{}'.format(len(ckd3)))
print('Percentage CKD 1: \n{}'.format(len(ckd1)/len(age_group1)))
print('Percentage CKD 2: \n{}'.format(len(ckd2)/len(age_group2)))
print('Percentage CKD 3: \n{}'.format(len(ckd3)/len(age_group3)))
print('Kruskal Wallis significance test: \n{}'.format(stats.kruskal((age_group1['Chronic_kidney_disease_(disorder)'].values), age_group2['Chronic_kidney_disease_(disorder)'].values, age_group3['Chronic_kidney_disease_(disorder)'].values)))

# calculating heart failure
hf1 = (age_group1[age_group1['Heart_failure_(disorder)']==1])
hf2 = (age_group2[age_group2['Heart_failure_(disorder)']==1])
hf3 = (age_group3[age_group3['Heart_failure_(disorder)']==1])
print('Number of HF in group 1:\n{}'.format(len(hf1)))
print('Number of HF in group 2:\n{}'.format(len(hf2)))
print('Number of HF in group 3:\n{}'.format(len(hf3)))
print('Percentage HF 1: \n{}'.format(len(hf1)/len(age_group1)))
print('Percentage HF 2: \n{}'.format(len(hf2)/len(age_group2)))
print('Percentage HF 3: \n{}'.format(len(hf3)/len(age_group3)))
print('Kruskal Wallis significance test: \n{}'.format(stats.kruskal((age_group1['Heart_failure_(disorder)'].values), age_group2['Heart_failure_(disorder)'].values, age_group3['Heart_failure_(disorder)'].values)))

# calculating previous myocardial infarction
mi1 = (age_group1[age_group1['Myocardial_infarction_(disorder)']==1])
mi2 = (age_group2[age_group2['Myocardial_infarction_(disorder)']==1])
mi3 = (age_group3[age_group3['Myocardial_infarction_(disorder)']==1])
print('Number of MI in group 1:\n{}'.format(len(mi1)))
print('Number of MI in group 2:\n{}'.format(len(mi2)))
print('Number of MI in group 3:\n{}'.format(len(mi3)))
print('Percentage MI 1: \n{}'.format(len(mi1)/len(age_group1)))
print('Percentage MI 2: \n{}'.format(len(mi2)/len(age_group2)))
print('Percentage MI 3: \n{}'.format(len(mi3)/len(age_group3)))
print('Kruskal Wallis significance test: \n{}'.format(stats.kruskal((age_group1['Myocardial_infarction_(disorder)'].values), age_group2['Myocardial_infarction_(disorder)'].values, age_group3['Myocardial_infarction_(disorder)'].values)))

# calculating atrial fibrillation
af1 = (age_group1[age_group1['Atrial_fibrillation_(disorder)']==1])
af2 = (age_group2[age_group2['Atrial_fibrillation_(disorder)']==1])
af3 = (age_group3[age_group3['Atrial_fibrillation_(disorder)']==1])
print('Number of AF in group 1:\n{}'.format(len(af1)))
print('Number of AF in group 2:\n{}'.format(len(af2)))
print('Number of AF in group 3:\n{}'.format(len(af3)))
print('Percentage AF 1: \n{}'.format(len(af1)/len(age_group1)))
print('Percentage AF 2: \n{}'.format(len(af2)/len(age_group2)))
print('Percentage AF 3: \n{}'.format(len(af3)/len(age_group3)))
print('Kruskal Wallis significance test: \n{}'.format(stats.kruskal((age_group1['Atrial_fibrillation_(disorder)'].values), age_group2['Atrial_fibrillation_(disorder)'].values, age_group3['Atrial_fibrillation_(disorder)'].values)))

# calculating atrial flutter
afl1 = (age_group1[age_group1['Atrial_flutter_(disorder)']==1])
afl2 = (age_group2[age_group2['Atrial_flutter_(disorder)']==1])
afl3 = (age_group3[age_group3['Atrial_flutter_(disorder)']==1])
print('Number of AFL in group 1:\n{}'.format(len(afl1)))
print('Number of AFL in group 2:\n{}'.format(len(afl2)))
print('Number of AFL in group 3:\n{}'.format(len(afl3)))
print('Percentage AFL 1: \n{}'.format(len(afl1)/len(age_group1)))
print('Percentage AFL 2: \n{}'.format(len(afl2)/len(age_group2)))
print('Percentage AFL 3: \n{}'.format(len(afl3)/len(age_group3)))
print('Kruskal Wallis significance test: \n{}'.format(stats.kruskal((age_group1['Atrial_flutter_(disorder)'].values), age_group2['Atrial_flutter_(disorder)'].values, age_group3['Atrial_flutter_(disorder)'].values)))

# calculating VT
vt1 = (age_group1[age_group1['Ventricular_tachycardia_(disorder)']==1])
vt2 = (age_group2[age_group2['Ventricular_tachycardia_(disorder)']==1])
vt3 = (age_group3[age_group3['Ventricular_tachycardia_(disorder)']==1])
print('Number of VT in group 1:\n{}'.format(len(vt1)))
print('Number of VT in group 2:\n{}'.format(len(vt2)))
print('Number of VT in group 3:\n{}'.format(len(vt3)))
print('Percentage VT 1: \n{}'.format(len(vt1)/len(age_group1)))
print('Percentage VT 2: \n{}'.format(len(vt2)/len(age_group2)))
print('Percentage VT 3: \n{}'.format(len(vt3)/len(age_group3)))
print('Kruskal Wallis significance test: \n{}'.format(stats.kruskal((age_group1['Ventricular_tachycardia_(disorder)'].values), age_group2['Ventricular_tachycardia_(disorder)'].values, age_group3['Ventricular_tachycardia_(disorder)'].values)))

# calculating VF
vf1 = (age_group1[age_group1['Ventricular_fibrillation_(disorder)']==1])
vf2 = (age_group2[age_group2['Ventricular_fibrillation_(disorder)']==1])
vf3 = (age_group3[age_group3['Ventricular_fibrillation_(disorder)']==1])
print('Number of VF in group 1:\n{}'.format(len(vf1)))
print('Number of VF in group 2:\n{}'.format(len(vf2)))
print('Number of VF in group 3:\n{}'.format(len(vf3)))
print('Percentage VF 1: \n{}'.format(len(vf1)/len(age_group1)))
print('Percentage VF 2: \n{}'.format(len(vf2)/len(age_group2)))
print('Percentage VF 3: \n{}'.format(len(vf3)/len(age_group3)))
print('Kruskal Wallis significance test: \n{}'.format(stats.kruskal((age_group1['Ventricular_fibrillation_(disorder)'].values), age_group2['Ventricular_fibrillation_(disorder)'].values, age_group3['Ventricular_fibrillation_(disorder)'].values)))

# calculating field strength
lowf1 = (age_group1[age_group1['Field_strength_x']==0])
lowf2 = (age_group2[age_group2['Field_strength_x']==0])
lowf3 = (age_group3[age_group3['Field_strength_x']==0])
print('Number of 1.5T in group 1:\n{}'.format(len(lowf1)))
print('Number of 1.5T in group 2:\n{}'.format(len(lowf2)))
print('Number of 1.5T in group 3:\n{}'.format(len(lowf3)))
print('Percentage 1.5T 1: \n{}'.format(len(lowf1)/len(age_group1)))
print('Percentage 1.5T 2: \n{}'.format(len(lowf2)/len(age_group2)))
print('Percentage 1.5T 3: \n{}'.format(len(lowf3)/len(age_group3)))
hf1 = (age_group1[age_group1['Field_strength_x']==1])
hf2 = (age_group2[age_group2['Field_strength_x']==1])
hf3 = (age_group3[age_group3['Field_strength_x']==1])
print('Number of 3T in group 1:\n{}'.format(len(hf1)))
print('Number of 3T in group 2:\n{}'.format(len(hf2)))
print('Number of 3T in group 3:\n{}'.format(len(hf3)))
print('Percentage 3T 1: \n{}'.format(len(hf1)/len(age_group1)))
print('Percentage 3T 2: \n{}'.format(len(hf2)/len(age_group2)))
print('Percentage 3T 3: \n{}'.format(len(hf3)/len(age_group3)))
print('Kruskal Wallis significance test: \n{}'.format(stats.kruskal((age_group1['Field_strength_x'].values), age_group2['Field_strength_x'].values, age_group3['Field_strength_x'].values)))

# calculating stress agent
a1 = (age_group1[age_group1['Stress_agent_x']==0])
a2 = (age_group2[age_group2['Stress_agent_x']==0])
a3 = (age_group3[age_group3['Stress_agent_x']==0])
print('Number of Adenosine in group 1:\n{}'.format(len(a1)))
print('Number of Adenosine in group 2:\n{}'.format(len(a2)))
print('Number of Adenosine in group 3:\n{}'.format(len(a3)))
print('Percentage Adenosine1: \n{}'.format(len(a1)/len(age_group1)))
print('Percentage Adenosine 2: \n{}'.format(len(a2)/len(age_group2)))
print('Percentage Adenosine 3: \n{}'.format(len(a3)/len(age_group3)))
reg1 = (age_group1[age_group1['Stress_agent_x']==1])
reg2 = (age_group2[age_group2['Stress_agent_x']==1])
reg3 = (age_group3[age_group3['Stress_agent_x']==1])
print('Number of Regadenosine in group 1:\n{}'.format(len(reg1)))
print('Number of Regadenosine in group 2:\n{}'.format(len(reg2)))
print('Number of Regadenosine in group 3:\n{}'.format(len(reg3)))
print('Percentage Regadenosine 1: \n{}'.format(len(reg1)/len(age_group1)))
print('Percentage Regadenosine 2: \n{}'.format(len(reg2)/len(age_group2)))
print('Percentage Regadenosine 3: \n{}'.format(len(reg3)/len(age_group3)))
print('Kruskal Wallis significance test: \n{}'.format(stats.kruskal((age_group1['Stress_agent_x'].values), age_group2['Stress_agent_x'].values, age_group3['Stress_agent_x'].values)))

# calculating LV function
lvm1 = pd.Series(data=age_group1['LVEF_(%)'].astype('int'))
print('LVEF mean for group 1:\n{}'.format(lvm1.mean()))
print('LVEF SD for group 1:\n{}'.format(lvm1.std()))
lvm2 = pd.Series(data=age_group2['LVEF_(%)'].astype('int'))
print('LVEF mean for group 2:\n{}'.format(lvm2.mean()))
print('LVEF SD for group 2:\n{}'.format(lvm2.std()))
lvm3 = pd.Series(data=age_group3['LVEF_(%)'].astype('int'))
print('LVEF mean for group 3:\n{}'.format(lvm3.mean()))
print('LVEF SD for group 3:\n{}'.format(lvm3.std()))
print('One Way ANOVA test LVEF: \n{}'.format(stats.f_oneway(lvm1, lvm2, lvm3)))
lvm = pd.Series(data=data['LVEF_(%)'].astype('int'))
print('LVEF mean for total:\n{}'.format(lvm.mean()))
print('LVEF SD for total:\n{}'.format(lvm.std()))

# calculating RV function
rvm1 = pd.Series(data=age_group1['RVEF_(%)'].astype('int'))
print('RVEF mean for group 1:\n{}'.format(rvm1.mean()))
print('RVEF SD for group 1:\n{}'.format(rvm1.std()))
rvm2 = pd.Series(data=age_group2['RVEF_(%)'].astype('int'))
print('RVEF mean for group 2:\n{}'.format(rvm2.mean()))
print('RVEF SD for group 2:\n{}'.format(rvm2.std()))
rvm3 = pd.Series(data=age_group3['RVEF_(%)'].astype('int'))
print('RVEF mean for group 3:\n{}'.format(rvm3.mean()))
print('RVEF SD for group 3:\n{}'.format(rvm3.std()))
print('One Way ANOVA test RVEF: \n{}'.format(stats.f_oneway(rvm1, rvm2, rvm3)))
rvm = pd.Series(data=data['RVEF_(%)'].astype('int'))
print('RVEF mean for total:\n{}'.format(rvm.mean()))
print('RVEF SD for total:\n{}'.format(rvm.std()))

# calculating perfusion
perf1 = (age_group1[age_group1['Positive_perf']==1])
perf2 = (age_group2[age_group2['Positive_perf']==1])
perf3 = (age_group3[age_group3['Positive_perf']==1])
print('Number of +ve perf in group 1:\n{}'.format(len(perf1)))
print('Number of +ve perf in group 2:\n{}'.format(len(perf2)))
print('Number of +ve perf in group 3:\n{}'.format(len(perf3)))
print('Percentage +ve perf 1: \n{}'.format(len(perf1)/len(age_group1)))
print('Percentage +ve perf 2: \n{}'.format(len(perf2)/len(age_group2)))
print('Percentage +ve perf 3: \n{}'.format(len(perf3)/len(age_group3)))
print('Kruskal Wallis significance test: \n{}'.format(stats.kruskal((age_group1['Positive_perf'].values), age_group2['Positive_perf'].values, age_group3['Positive_perf'].values)))

# calculating LGE
lge1 = (age_group1[age_group1['Positive_LGE']==1])
lge2 = (age_group2[age_group2['Positive_LGE']==1])
lge3 = (age_group3[age_group3['Positive_LGE']==1])
print('Number of +ve LGE in group 1:\n{}'.format(len(lge1)))
print('Number of +ve LGE in group 2:\n{}'.format(len(lge2)))
print('Number of +ve LGE in group 3:\n{}'.format(len(lge3)))
print('Percentage +ve LGE 1: \n{}'.format(len(lge1)/len(age_group1)))
print('Percentage +ve LGE 2: \n{}'.format(len(lge2)/len(age_group2)))
print('Percentage +ve LGE 3: \n{}'.format(len(lge3)/len(age_group3)))
print('Kruskal Wallis significance test: \n{}'.format(stats.kruskal((age_group1['Positive_LGE'].values), age_group2['Positive_LGE'].values, age_group3['Positive_LGE'].values)))

# Plot subgroups
data['Genders'] = data['patient_GenderCode_y']
fig, axs = plt.subplots(1,2)
bins = [0,65,75,np.inf]
names = ['<65', '65-75', '>75']
data['Age'] = pd.cut(data['Age_on_20.08.2021_x'], bins, labels=names)
sns.catplot(x="Age", y="Positive_LGE", hue='Genders', kind="bar", data=data, ax=axs[1], palette=sns.color_palette(['darkslateblue', 'lavender']))
plt.text(0.5, 0.85, "p value <0.001", horizontalalignment='left', size='medium', color='black', weight='normal')
plt.ylabel('Positive ischemic LGE')
plt.show()
