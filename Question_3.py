# -*- coding: utf-8 -*-
# Q3 for the capstone project
#%% import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%% load data
data = pd.read_csv("DrowningData.csv")
# Get the values that some important columns can take
waterTypes      = np.unique(data['Types_of_waters'].to_numpy())
drowningReasons = np.unique(data['Drowning_reasons'].to_numpy())
drowningResults = np.unique(data['Drowning_results'].to_numpy())
drowningResults = np.unique(data['Drowning_results'].to_numpy())

#%% Keep only the columns that we are interested in
attributes = ['Drowning_results', 'CC_Month', 'Hour', 'Gender', 'Age', 
              'Swimming_skills', 'Types_of_waters', 'Drowning_reasons']
data = data[attributes]

#%% Data formatting
data = data[data['Gender']!='Unknown'] # drop entries with unknown gender
data = data[data['Age']!='Unknown']    # drop entries with unknown age
data = data[data['Types_of_waters']!='Others'] # drop entries with 'others' types of waters
data = data[data['Drowning_reasons']!='Others'] # drop entries with 'others' types of reasons
data = data[data['Drowning_reasons']!='Floating Corpse'] # drop entries with 'floaing corpse' types of reasons

# data = data[data['Swimming_skills']!='Unknown'] # drop entries with unknown swimming skills

# Encode drowning results by Dead = 0, Missing = 0, Rescued = 1
data = data.replace({'Drowning_results':{'Dead':0, 'Missing':0, 'Rescued':1}})
# Encode gender by Male = 1, Female = 0
data = data.replace({'Gender':{'Male':1, 'Female':0}})
# convert Age to number
data['Age'] = pd.to_numeric(data['Age']) 
# Encode swimming skills by No = -1, Unknown = 0, Yes = 1
data = data.replace({'Swimming_skills':{'No':-1, 'Unknown':0, 'Yes':1}})
# data = data.replace({'Swimming_skills':{'No':0, 'Yes':1}})

# One-hot encode the last two variables
# data = pd.get_dummies(data, columns=['Types_of_waters', 'Drowning_reasons'], sparse=False)
# Use dummy variables instead to avoid sparsity
waterDict = dict(zip(waterTypes,np.arange(len(waterTypes))))
reasonDict = dict(zip(drowningReasons,np.arange(len(drowningReasons))))
data = data.replace({'Types_of_waters':waterDict})
data = data.replace({'Drowning_reasons':reasonDict})

#%%
X = data.drop(columns=['Drowning_results'])
y = data['Drowning_results'].to_numpy()

#%% Make training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#%% Train a random forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

#%%
paramGrid = {'n_estimators': [500], 
             # 'criterion': ['gini', 'entropy'],
             'max_features': np.arange(1,5),
             'min_samples_leaf': [1,2,3] ,
             'max_depth': [7,10,13] }
rfc = RandomForestClassifier() # initialize the classifier
rfc_grid = GridSearchCV(estimator=rfc, param_grid=paramGrid, cv=5, n_jobs=5, verbose=2)
rfc_grid.fit(X_train, y_train)

#%% Print the best parameters
bestParam = rfc_grid.best_params_
trainScore = rfc_grid.score(X_train, y_train)
testScore = rfc_grid.score(X_test, y_test)
print('Best Parameters: ', bestParam)
print('Train: %0.5f'% trainScore )
print('Test: %0.5f'% testScore)

#%% Train the model based on the best parameters
rfc = RandomForestClassifier(n_estimators=1000, max_features=2, criterion='entropy',\
                             min_samples_leaf=1, max_depth=10, random_state=10)
rfc.fit(X_train, y_train)
print('Train Accuracy: ', rfc.score(X_train, y_train))
print('Test Accuracy: ', rfc.score(X_test, y_test))


#%% Plot the confusion matrix
import seaborn as sb
from sklearn.metrics import confusion_matrix
# print(rfc_grid.classes_) # make sure the order of the classes are correct
resc = ['Not Rescued', 'Rescued']
y_pred = rfc.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
sb.heatmap(cm, annot=True, fmt='d', xticklabels=resc, yticklabels=resc, cbar=False)
plt.xlabel('Prediction')
plt.ylabel('Truth')
plt.title('Confusion Matrix of Random Forest Prediction')
# plt.savefig('ConfusionMatrix.png')

sensitivity = cm[0,0]/(cm[0,0]+cm[1,0])
specificity = cm[1,1]/(cm[0,1]+cm[1,1])
#%% Calculate the ROC AUC score
from sklearn.metrics import roc_curve, roc_auc_score
rfc_probs = rfc.predict_proba(X_test) # probabilistic prediction
y_probs = rfc_probs[:, 1] # keep probabilities for positive outcome
rfc_auc = roc_auc_score(y_test, y_probs)
print('ROC AUC Score: ', rfc_auc)
# Plot the ROC curve
fpr, tpr, _ = roc_curve(y_test, y_probs)
plt.plot(fpr, tpr, marker='.', label='Random Forest (AUROC =%0.3f)'%rfc_auc)
plt.plot([0,1],[0,1], linestyle='--', c='black')
plt.legend(loc='lower right')
plt.title('ROC Curve')
# plt.savefig('ROCCurve.png')

#%% Feature importance based on mean decrease in impurity
# make a dataframe representing the importance for each feature
featImp = pd.DataFrame({'Factor':X.columns, 'Importance':rfc.feature_importances_})
# add the standard deviations to the feature importance
std = np.std([tree.feature_importances_ for tree in rfc.estimators_], axis=0)
featImp['std']=std
# sort them in decreasing order
featImp = featImp.sort_values(by='Importance')
# plot 
featImp.plot.barh(x='Factor', y='Importance', xerr='std', rot=0, color='red')
plt.title('Feature importance (mean decrease in impurity)')
# plt.savefig('FeatureImportance.png')



#%% 
from scipy import stats
data2 = pd.read_csv("DrowningData.csv")
data2 = data2.replace({'Drowning_results':{'Dead':0, 'Missing':0, 'Rescued':1}})

#%% Are those who know how to swim more likely to get rescued?
swimming = data2[['Swimming_skills', 'Drowning_results']]
swim_yes = swimming[swimming['Swimming_skills']=='Yes']['Drowning_results']
swim_no  = swimming[swimming['Swimming_skills']=='No']['Drowning_results']
t, p = stats.ttest_ind(swim_yes, swim_no, alternative = 'greater', equal_var=False)
print(p)
# p=0.002911596380526864, so yes
#%% Are those rescued drowning people gendered?
sex = data2[['Gender', 'Drowning_results']]
sex = sex[sex['Gender']!='Unknown']
sex_m = sex[sex['Gender']=='Male']['Drowning_results']
sex_f = sex[sex['Gender']=='Female']['Drowning_results']
t, p = stats.ttest_ind(sex_m, sex_f, alternative='less', equal_var=False)
print(p)
# p=2.74e-12, so it is gendered



