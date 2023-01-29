#!/usr/bin/env python
# coding: utf-8

# In[63]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[64]:


def safelog(x):
    return(np.log(x + 1e-100))


# In[65]:


images = np.genfromtxt("hw02_data_points.csv", delimiter = ",")
labels= np.genfromtxt("hw02_class_labels.csv")


# In[66]:


training_length = 10000

training_data = images[:training_length]
training_label = labels[:training_length]
test_data = images[training_length:]
test_label = labels[training_length:]

data_set = np.vstack((training_data.T,training_label)).T

N = data_set.shape[0]
D = data_set.shape[1]-1

X = data_set[:, 0:D]
y_truth = data_set[:, D:(D + 1)].astype(int)

K = np.max(y_truth)
Y_truth = np.zeros((N, K)).astype(int)
Y_truth[range(N), y_truth[:, 0] - 1] = 1


# In[67]:


eta = 1e-5
iteration_count = 1000


# In[68]:


W = np.genfromtxt("hw02_W_initial.csv", delimiter = ",")
w0 = np.genfromtxt("hw02_w0_initial.csv", delimiter = ",")


# In[69]:


## Gradient of the error should be matrix multiplied by differentiation of the sigmoid function.
# Y_predicted is the output of the sigmoid fuction here, so (diff(sig) := Y_p * (1 - Y_p)
def gradient_W(X, Y_truth, Y_predicted):
    return (np.asarray([-np.matmul((Y_truth[:,c] - Y_predicted[:,c])  * Y_predicted[:,c] * (1 - Y_predicted[:,c]), X) for c in range(K)]).transpose())

def gradient_w0(Y_truth, Y_predicted):
    return(-np.sum((Y_truth - Y_predicted) * Y_predicted * (1-Y_predicted), axis = 0))

def sigmoid(X, w, w0):
    return(1 / (1 + np.exp(-(np.matmul(X, w) + w0))))


# In[70]:


objective_values = []
# instead of while, for loop is used due to epsilon does not exist
for i in range(iteration_count):
    Y_predicted = sigmoid(X, W, w0)
    
    objective_values = np.append(objective_values, np.sum(0.5*((Y_truth-Y_predicted)**2)))

    W_old = W
    w0_old = w0

    W = W - eta * gradient_W(X, Y_truth, Y_predicted)
    w0 = w0 - eta * gradient_w0(Y_truth, Y_predicted)

print(f"W:\n{W}")
print(f"w0:\n{w0}")


# In[71]:


plt.figure(figsize = (16, 8))
plt.plot(range(1, iteration_count+1), objective_values, "k-")
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.show()


# In[72]:


# calculate confusion matrix
y_predicted = np.argmax(Y_predicted, axis = 1) + 1
confusion_matrix = pd.crosstab(y_predicted, y_truth.T,
                               rownames = ["y_pred"],
                               colnames = ["y_truth"])
print(f"confusion_matrix:\n{confusion_matrix}")


# In[73]:


Y_predicted_test = sigmoid(test_data, W, w0)
y_truth_test = test_label


# In[74]:


y_predicted_test = np.argmax(Y_predicted_test, axis = 1) + 1
confusion_test = pd.crosstab(y_predicted_test, y_truth_test.T,
                               rownames = ["y_pred"],
                               colnames = ["y_truth"])
print(f"confusion_test:\n{confusion_test}")


# In[ ]:




