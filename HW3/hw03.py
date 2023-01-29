#!/usr/bin/env python
# coding: utf-8

# In[25]:


import numpy as np
import matplotlib.pyplot as plt
import math


# In[26]:


data_set = np.genfromtxt("hw03_data_set.csv", delimiter = ",", skip_header = 1)


# In[27]:


training_length = 150
test_length = 122

x_train = data_set[:training_length, 0]
y_train = data_set[:training_length, 1].astype(int)

x_test = data_set[training_length:, 0]
y_test = data_set[training_length:, 1].astype(int)


# In[28]:


bin_width = 0.37
origin = 1.5
x_max = max(x_train)

left_borders = np.arange(start = origin,
                         stop = x_max,
                         step = bin_width)
right_borders = np.arange(start = origin + bin_width,
                          stop = x_max + bin_width,
                          step = bin_width)
g_x = np.zeros(len(left_borders))

for i in range(len(left_borders)):
    num = np.sum(((left_borders[i] < x_train) & (x_train <= right_borders[i])) * y_train)
    den = np.sum((left_borders[i] < x_train) & (x_train <= right_borders[i]))
    g_x[i]= num/den


# In[29]:


plt.figure(figsize = (15, 6))
plt.plot(x_train,
         y_train,
         "b.", markersize = 10,label='training')
plt.legend()
plt.plot(x_test,
         y_test,
         "r.", markersize = 10,label='test')
plt.legend()
for b in range(len(left_borders)):
    plt.plot([left_borders[b], right_borders[b]], [g_x[b], g_x[b]], "k-")
for b in range(len(left_borders) - 1):
    plt.plot([right_borders[b], right_borders[b]], [g_x[b], g_x[b + 1]], "k-")    
    
plt.ylabel("Waiting time to next eruption (min)")
plt.xlabel("Eruption time (min)")    

plt.show() 


# In[30]:


rmse = 0
y = [np.matmul((x_test[i] < right_borders) & (x_test[i] > left_borders), g_x) for i in range(test_length)]

for i in range(test_length):
    rmse += ((y[i] - y_test[i])**2)/test_length
rmse=np.sqrt(rmse)

print(f"Regressogram => RMSE is {rmse} when h is {bin_width}")


# In[31]:


data_particle_length=0.01
data_interval = np.linspace(1.5, 5.1, 361)

ms = np.zeros(len(data_interval))

for i in range(len(data_interval)):
    num = np.sum(((np.abs(data_interval[i] - x_train)/bin_width)<=0.5) * y_train)
    den = np.sum((np.abs(data_interval[i] - x_train)/bin_width)<=0.5)
    ms[i]= num/den


# In[32]:


def visualize_with_nonparam_solution(_arr):
    plt.figure(figsize = (15, 6))
    plt.plot(x_train,
             y_train,
             "b.", markersize = 10,label='training')
    plt.legend()
    plt.plot(x_test,
             y_test,
             "r.", markersize = 10,label='test')
    plt.legend()

    plt.plot(data_interval,
             _arr,
             "k-", markersize = 0.5)

    plt.ylabel("Waiting time to next eruption (min)")
    plt.xlabel("Eruption time (min)")    

    plt.show() 

visualize_with_nonparam_solution(ms)


# In[33]:


rmse = 0
ind=0
for x in x_test:
    ms_ind = round((x-origin)/data_particle_length)
    rmse += ((ms[ms_ind] - y_test[ind])**2)/test_length
    ind+=1
rmse=np.sqrt(rmse) 
print(f"Running Mean => RMSE is {rmse} when h is {bin_width}")


# In[34]:


ks = np.zeros(len(data_interval))
for i in range(len(data_interval)):
    num = np.sum((1/np.sqrt(2*math.pi) * np.exp(-0.5*(data_interval[i] - x_train)**2 / bin_width**2)) * y_train)
    den = np.sum(1/np.sqrt(2*math.pi) * np.exp(-0.5*(data_interval[i] - x_train)**2 / bin_width**2))
    ks[i]= num/den


# In[35]:


visualize_with_nonparam_solution(ks)


# In[36]:


rmse = 0
ind=0
for x in x_test:
    ks_ind = round((x-origin)/data_particle_length)
    rmse += ((ks[ks_ind] - y_test[ind])**2)/test_length
    ind+=1
rmse=np.sqrt(rmse) 
print(f"Kernel Smoother => RMSE is {rmse} when h is {bin_width}")


# In[ ]:
