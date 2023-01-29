#!/usr/bin/env python
# coding: utf-8

# In[50]:

import numpy as np
import pandas as pd

# In[51]:

def safelog(x):
    return(np.log(x + 1e-100))

# In[52]:

data_set = pd.read_csv("hw01_data_points.csv",header=None)
labels = np.genfromtxt("hw01_class_labels.csv", delimiter = "\n")

# In[53]:

dict_nucl = {'A': 1, 'C': 2, 'G': 3, 'T': 4}

def transform(r):
    return r.map(dict_nucl)
nucleobases = data_set.apply(lambda row: transform(row), axis=1).to_numpy()
#print(nucleobases)

num_class = 2
sequence_length = nucleobases[0].size
training_length = 300

training_data = nucleobases[:training_length]
test_data = nucleobases[training_length:]
training_labels = labels[:training_length]
test_labels = labels[training_length:]

# In[54]:

class_priors = [np.mean((training_labels==c+1)) for c in range(num_class)]

def pXcd(corresponding_num):
    a = np.zeros((num_class,sequence_length))
    for c in range(num_class):
        for d in range(sequence_length):
            a[c][d]=(np.sum([(training_data[i][d]==corresponding_num)*(training_labels[i]==c+1) for i in range(training_length)])
                     /((training_labels==c+1).sum()))

    return a

pAcd = pXcd(dict_nucl.get('A'))
pCcd = pXcd(dict_nucl.get('C'))
pGcd = pXcd(dict_nucl.get('G'))
pTcd = pXcd(dict_nucl.get('T'))
print("### Model Parameters ###")
print(f"pAcd: \n{pAcd}\npCcd: \n{pCcd}\npGcd: \n{pGcd}\npTcd: \n{pTcd}")
print(f"class priors: \n{class_priors}\n")

# In[55]:

def calculate_confusion_matrix(data, p1, p2, p3, p4, labels, class_priors):    
    score = [np.dot(data==1, safelog(p1[c])) + 
             np.dot(data==2, safelog(p2[c])) + 
             np.dot(data==3, safelog(p3[c])) + 
             np.dot(data==4, safelog(p4[c])) + np.log(class_priors[c]) for c in range(num_class)]
    y_predicted = np.argmax(np.transpose(score), axis = 1) + 1
    confusion_matrix = pd.crosstab(y_predicted, labels, rownames = ['y_pred'], colnames = ['y_truth'])
    return confusion_matrix

# In[56]:

print("### Confusion Matrix for the training data ###")
confusion_train = calculate_confusion_matrix(training_data,pAcd,pCcd,pGcd,pTcd,training_labels,class_priors)
print(confusion_train)

# In[57]:

print("### Confusion Matrix for the test data ###")
confusion_test = calculate_confusion_matrix(test_data,pAcd,pCcd,pGcd,pTcd,test_labels,class_priors)
print(confusion_test)
