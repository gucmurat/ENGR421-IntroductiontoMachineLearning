#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import math


# In[2]:


data_set = np.genfromtxt("hw04_data_set.csv", delimiter = ",", skip_header = 1)


# In[3]:


training_length = 150
test_length = 122

x_train = data_set[:training_length, 0]
y_train = data_set[:training_length, 1].astype(int)

x_test = data_set[training_length:, 0]
y_test = data_set[training_length:, 1].astype(int)

X = np.concatenate((x_train, x_test), axis=0)
y = np.concatenate((y_train, y_test), axis=0).astype(int)
data_points = np.linspace(np.min(X), np.max(X), 10000)


# In[4]:


P=25

def learning_algorithm(P):
    node_indices = {}
    is_terminal = {}
    need_split = {}
    
    node_means = {}
    node_splits = {}
    
    node_indices[1] = np.array(range(training_length))
    is_terminal[1] = False
    need_split[1] = True
    
    while True:
        split_nodes = [key for key, value in need_split.items() if value == True]
        
        if len(split_nodes) == 0:
            break
        
        for split_node in split_nodes:
            data_indices = node_indices[split_node]
            need_split[split_node] = False
            node_mean = np.mean(y_train[data_indices])
            
            if x_train[data_indices].size <= P:
                is_terminal[split_node] = True
                node_means[split_node] = node_mean
            else:
                is_terminal[split_node] = False
                unique_values = np.sort(np.unique(x_train[data_indices]))
            
                if len(unique_values) == 1:
                    is_terminal[split_node] = True
                    node_means[split_node] = node_mean
                    continue
                
                split_positions = (unique_values[1:len(unique_values)] + unique_values[0:(len(unique_values)-1)])/2
                split_scores = np.repeat(0.0, len(split_positions))
                
                for split in range(len(split_positions)):
                    left_indices = data_indices[x_train[data_indices] < split_positions[split]]
                    right_indices = data_indices[x_train[data_indices] >= split_positions[split]]
                    sum_of_len_indices = len(left_indices)+len(right_indices)
                    split_scores[split] = ((np.sum((y_train[left_indices] - np.mean(y_train[left_indices])) ** 2)) +
                                        (np.sum((y_train[right_indices] - np.mean(y_train[right_indices])) ** 2)))/sum_of_len_indices
                    
                best_split = split_positions[np.argmin(split_scores)]
                node_splits[split_node] = best_split
                
                left_indices = data_indices[(x_train[data_indices] < best_split)]
                node_indices[2 * split_node] = left_indices
                is_terminal[2 * split_node]  = False
                need_split[2 * split_node] = True

                right_indices = data_indices[(x_train[data_indices] >= best_split)]
                node_indices[2 * split_node + 1] = right_indices
                is_terminal[2 * split_node + 1] = False
                need_split[2 * split_node + 1]  =True
       
    return node_means, node_splits, is_terminal


def predict(p, _node_means, _node_splits, _is_terminal):
    index = 1 
    while True:
        if _is_terminal[index]:
            break
        else:
            if _node_splits[index] < p:
                index = index*2 + 1 
            else:
                index = index*2 
    return _node_means[index]   

def rmse(y1, y2):
    return np.sqrt(np.mean((y1 - y2)**2))


# In[5]:


node_means, node_splits,is_terminal = learning_algorithm(P)

y_pred = [predict(data_points[i], node_means, node_splits, is_terminal) for i in range(len(data_points))]
plt.figure(figsize = (12,6))
plt.plot(data_points, y_pred, "k")
plt.scatter(x_train, y_train, color = "blue", label="training")
plt.scatter(x_test, y_test, color = "red", label="test")
plt.xlabel("Eruption Time (min)")
plt.ylabel("Waiting time to next eruption (min)")
plt.legend()
plt.show()


# In[6]:


y_pred = [predict(x_train[i], node_means, node_splits, is_terminal) for i in range(training_length)]
print(f"RMSE is on training set {rmse(y_train,y_pred)} when P is {str(P)}")
y_pred = [predict(x_test[i], node_means, node_splits, is_terminal) for i in range(test_length)]
print(f"RMSE is on training set {rmse(y_test,y_pred)} when P is {str(P)}")


# In[7]:


points = range(5,55,5)


# In[8]:


rmse_test = []
rmse_train = []

for point in points:
    node_means, node_splits, is_terminal = learning_algorithm(point)
    pred_train = np.array([predict(x,node_means, node_splits, is_terminal) for x in x_train])
    pred_test = np.array([predict(x,node_means, node_splits, is_terminal) for x in x_test])
    rmse_test.append(rmse(y_test,pred_test))
    rmse_train.append(rmse(y_train,pred_train))
    
plt.figure(figsize = (8,8))
plt.plot(points,rmse_train, color= "blue")
plt.scatter(points,rmse_train, color= "blue") 
plt.plot(points,rmse_test, color= "red")
plt.scatter(points,rmse_test, color= "red") 

plt.xlabel("Pre-pruning size (P)")
plt.ylabel("RMSE")
plt.show()

