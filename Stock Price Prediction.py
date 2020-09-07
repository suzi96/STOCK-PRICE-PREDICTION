#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#importing the dataset
dataset=pd.read_csv('NSE-TATAGLOBAL11.csv')


# In[3]:


dataset.head()


# In[4]:


x=dataset.iloc[:, 1:7].values
y=dataset.iloc[:, 7].values


# In[5]:


print(x)
print(y)


# In[6]:


#Splitting the dataset into Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2, random_state=0)
print(x_train.shape)


# In[7]:


#Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train, y_train)


# In[8]:


print("The intercept is: ", regressor.intercept_)


# In[9]:


print('The slope is: ', regressor.coef_)


# In[10]:


#Predicting the Test set results
y_pred=regressor.predict(x_test)


# In[11]:


#visualizing the Training set Result
print(y_pred)


# In[12]:


df=pd.DataFrame({'Actual' :y_test, 'Predicted': y_pred})
df


# In[15]:


from sklearn import metrics
from sklearn.metrics import mean_squared_error,r2_score
print("Mean_Squared_Error :", mean_squared_error(y_test,y_pred))
print("r_2 statistic: %.2f" %r2_score(y_test,y_pred))
print("Root_Mean_Sqared_Error :", np.sqrt(mean_squared_error(y_test,y_pred)))


# In[ ]:




