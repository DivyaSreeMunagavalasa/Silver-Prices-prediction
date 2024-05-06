#!/usr/bin/env python
# coding: utf-8

# # SILVER PRICE PREDICTION IN INDIA

# # IMPORTING LIBRARIES

# In[41]:


#import the required libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
#to enable the inline plotting, where the plots/graphs 
#will be displayed just below the cell where your plotting commands are written.


# In[2]:


#load the data file
sil=pd.read_csv(r"C:\Users\Divya sree\Downloads\silver.csv")


# In[3]:


#Load the top 5 records of data
sil.head()


# In[4]:


#Load the last 5 records of data
sil.tail()


# In[5]:


#dataset size
sil.shape


# In[6]:


#displaying column names
sil.columns


# In[7]:


#displaying values
sil.values


# In[8]:


#check the descriptive statistics of numeric values
sil.describe()


# In[9]:


#displaying columns datatypes
sil.dtypes


# # DATA PROCESSING

# In[10]:


#Checking for null values
sil.isnull().sum()


# In[11]:


#There are no null values


# In[12]:


#Checking for duplicate values
sil.duplicated().sum()


# In[13]:


#No duplicated values


# In[14]:


#Volume should be float amount, so convert that column to float datatype
sil['Volume']=sil['Volume'].astype(float)


# In[15]:


#Info about the dataframe
sil.info()


# In[16]:


#Checking NA values
sil.isna().sum()


# In[17]:


#There are no NA values


# In[18]:


sil


# # DATA VISUALIZATION

# In[19]:


#Plotting between open price and close price
plt.scatter(sil['Open'],sil['Close'])
plt.xlabel('open')
plt.ylabel('close')


# In[20]:


#Plotting between high and close
plt.scatter(sil['High'],sil['Close'])
plt.xlabel('high')
plt.ylabel('close')


# In[21]:


plt.plot(sil['High'],sil['Close'])
plt.xlabel('high')
plt.ylabel('close')


# In[22]:


plt.hist(sil['Close'])


# In[23]:


plt.boxplot(sil['Close'])


# In[24]:


plt.boxplot(sil['Volume'])


# In[25]:


#Finding correlation among different columns
sil.corr()


# # MODEL BULIDING

# In[26]:


from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score


# In[27]:


# Defining Independent and Dependent variables


# In[28]:


x=sil.iloc[ :,[1,2,3,5]]
x


# In[29]:


y=sil.iloc[:,4]
y


# In[30]:


# Splitting the data into training and testing data


# In[31]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y)


# In[32]:


x_train.shape


# In[33]:


x_test.shape


# In[34]:


y_train.shape


# In[35]:


y_test.shape


# # USING DECISION TREE

# In[36]:


from sklearn import tree
model=tree.DecisionTreeRegressor()
model_dt=model.fit(x_train,y_train)


# In[37]:


y_pred= model_dt.predict(x_test)
y_pred


# In[38]:


y_pred.shape


# In[39]:


from sklearn.metrics import r2_score
r2_score(y_pred,y_test)*100


# In[40]:


l=[]
for i in range(0,4):
    print("Enter "+x.columns[i],"value:")
    z=float(input())
    l.append(z)
l=np.array(l).reshape(1,-1)
solution=model_dt.predict(l)
solution


# # Using Random Forest Regressor

# In[71]:


from sklearn.ensemble import RandomForestRegressor
model=RandomForestRegressor()
modelr=model.fit(x_train,y_train)


# In[72]:


y_pred=modelr.predict(x_test)
y_pred


# In[73]:


from sklearn.metrics import r2_score
r2_score(y_pred,y_test)*100


# In[74]:


l=[]
for i in range(0,4):
    print("Enter "+x.columns[i],"value:")
    z=float(input())
    l.append(z)
l=np.array(l).reshape(1,-1)
solution=modelr.predict(l)
solution


# In[ ]:


modelr.predict([[5.420,5.420,5.320,27560.0]])


# In[ ]:


modelr.predict([[17.672,18.163,17.625,0.0]])


# In[ ]:


modelr.predict([[19.558,19.605,19.457,0	]])


# In[ ]:


modelr.predict([[19.637,19.690,19.555,0]])


# In[ ]:


modelr.predict([[18.355,18.670,18.355,50]])	


# # Using Multiple Linear Regression

# In[75]:


from sklearn.linear_model import LinearRegression
model=LinearRegression()
modelm=model.fit(x_train,y_train)


# In[76]:


pred_y=modelm.predict(x_test)


# In[79]:


from sklearn.metrics import r2_score
r2_score(pred_y,y_test)*100


# In[80]:


l=[]
for i in range(0,4):
    print("Enter "+x.columns[i],"value:")
    z=float(input())
    l.append(z)
l=np.array(l).reshape(1,-1)
solution=modelm.predict(l)
solution


# In[ ]:


modelm.predict([[5.420,5.420,5.320,27560.0]])


# In[ ]:


modelm.predict([[17.672,18.163,17.625,0.0]])


# In[ ]:


modelm.predict([[19.637,19.690,19.555,0]])


# In[ ]:


modelm.predict([[18.355,18.670,18.355,50]])


# In[ ]:




