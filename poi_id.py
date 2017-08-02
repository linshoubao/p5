#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary', 'deferral_payments', 'total_payments', 'bonus',
                 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive',
                 'restricted_stock','to_messages', 'from_poi_to_this_person', 'from_messages',
                 'from_this_person_to_poi', 'shared_receipt_with_poi'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
   
##explore the data_dict
print 'the number of dataset:',len(data_dict)
#146
print 'the number of features:',len(data_dict['METTS MARK'])
#21

num_poi=0
for name in data_dict:
    if data_dict[name]['poi']==1:
        num_poi +=1
print 'the num of poi:',num_poi
#18
count_na=0
for name in data_dict:
    if data_dict[name]['loan_advances'] =='NaN':
        count_na +=1
print 'the num of NaN:',count_na
#142

### Task 2: Remove outliers
data_dict.pop('TOTAL',0)
data_dict.pop('LOCKHART, EUGENE E',0) # all of values are 'NaN'
data_dict.pop('THE TRAVEL AGENCY IN THE PARK',0) # not person
### Task 3: Create new feature(s)
#create fraction_from_poi and fraction_to_poi
for  name in data_dict:
    if data_dict[name]['from_poi_to_this_person']=='NaN' or data_dict[name]['from_this_person_to_poi']=='NaN' or \
       data_dict[name]['from_messages']=='NaN' or data_dict[name]['to_messages']=='NaN':
        data_dict[name]['fraction_from_poi']=0
        data_dict[name]['fraction_to_poi']=0
    else:
        data_dict[name]['fraction_from_poi']=float(data_dict[name]['from_poi_to_this_person'])/data_dict[name]['to_messages']
        data_dict[name]['fraction_to_poi']=float(data_dict[name]['from_this_person_to_poi'])/data_dict[name]['from_messages']
    
### Store to my_dataset for easy export below.
my_dataset = data_dict
features_list=features_list+['fraction_from_poi','fraction_to_poi']
### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

#scale features
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
features=scaler.fit_transform(features)

#select best features
from sklearn.feature_selection import SelectKBest
skb=SelectKBest(k=5)
skb.fit(features,labels)
features_final=sorted(zip(skb.get_support(),features_list[1:],skb.scores_),key=lambda x:x[2],reverse=True)
print 'the best features:',features_final
# 'exercised_stock_options','total_stock_value','bonus','salary','fraction_to_poi' is the best features

#create my features
my_features=['poi','exercised_stock_options','total_stock_value','bonus','salary','fraction_to_poi']
data = featureFormat(my_dataset, my_features, sort_keys = True)
labels, features = targetFeatureSplit(data)

#split features,labels
from sklearn.cross_validation import StratifiedShuffleSplit
cv=StratifiedShuffleSplit(labels,test_size=0.3,random_state=42)
for train_idx,test_idx in cv:
    features_train=[]
    features_test =[]
    labels_train  =[]
    labels_test   =[]
    for  ii in train_idx:
        features_train.append(features[ii])
        labels_train.append(labels[ii])
    for jj in test_idx:
        features_test.append(features[jj])
        labels_test.append(labels[jj])

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
def calculate_precision_recall(clf,features_test,labels_test):
    true_negatives=0
    false_negatives=0
    true_positives=0
    false_positives=0
    predictions=clf.predict(features_test)
    for prediction,truth in zip(predictions,labels_test):
        if prediction==0 and truth==0:
            true_negatives +=1
        elif prediction==0 and truth==1:
            false_negatives +=1
        elif prediction==1 and truth==0:
            false_positives +=1
        elif prediction==1 and truth==1:
            true_positives +=1
    print true_negatives , false_negatives , false_positives , true_positives        
    total_predictions=true_negatives + false_negatives + false_positives + true_positives            
    accuracy=1.0*(true_positives+true_negatives)/total_predictions
    precision=1.0*true_positives/(true_positives+false_positives)
    recall=1.0*true_positives/(true_positives+false_negatives)
    return precision, recall


# Provided to give you a starting point. Try a variety of classifiers.
#sklearn_one
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(features_train,labels_train)
print 'precision=%s,recall=%s' % calculate_precision_recall(clf,features_test,labels_test)
# precision=0.33,recall=0.2

#sklearn_two
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
parameters={'min_samples_split':[2,10,20],
            'criterion':['entropy','gini'],
            'max_depth':[None,2,5,10],
            'min_samples_leaf':[1,5,10],
            'max_leaf_nodes':[None,5,10,20]}
dtc=DecisionTreeClassifier()
clf=GridSearchCV(dtc,parameters)
clf.fit(features_train,labels_train)
print 'the best parameters:',clf.best_params_

'''
best_parameters={'min_samples_split': 2,
                'max_leaf_nodes': None,
                'max_depth': None,
                'min_samples_leaf': 10}
'''

print 'precision=%s,recall=%s' % calculate_precision_recall(clf,features_test,labels_test)
# precision=0.5,recall=0.4

#sklearn_three
from sklearn.neighbors import KNeighborsClassifier
parameters={'n_neighbors':[3,5,10,15]}
rfr=KNeighborsClassifier()
clf=GridSearchCV(rfr,parameters)
clf.fit(features_train,labels_train)
print 'the best parameters:',clf.best_params_
#the best parameters: {'n_neighbors': 5}

print 'precision=%s,recall=%s' % calculate_precision_recall(clf,features_test,labels_test)
# precision=0.666666666667,recall=0.4

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3,random_state=42)

clf=GaussianNB()
features_list=my_features

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)

