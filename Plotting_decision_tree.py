#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[2]:


from sklearn.datasets import load_iris
data = load_iris()
df = pd.DataFrame(data.data,columns=data.feature_names)
df['target'] = pd.DataFrame(data.target)


# In[3]:


x = df.drop(['target'],axis=1)
y = df['target']


# In[4]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=143)


# In[5]:


from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(x_train,y_train)


# In[6]:


from sklearn import tree
plt.figure(figsize=(15,10))
tree.plot_tree(classifier,filled=True)


# In[7]:


from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
params = {'max_depth':range(1,50),
         'criterion':['gini','entropy'],
         'max_leaf_nodes':range(1,5)}
model = GridSearchCV(classifier,param_grid=params,cv=5,verbose=1)
model.fit(x_train,y_train)


# In[8]:


model.best_params_


# In[9]:


model.best_score_


# In[10]:


y_pred = model.predict(x_test)


# In[11]:


from sklearn.metrics import confusion_matrix,accuracy_score
confusion_matrix(y_pred,y_test)


# In[12]:


confusion_matrix(model.predict(x_train),y_train)


# In[13]:


accuracy_score(y_test,y_pred)


# In[14]:


accuracy_score(y_train,model.predict(x_train))


# In[15]:


from sklearn.linear_model import LogisticRegression


# In[16]:


lr = LogisticRegression()
lr.fit(x_train,y_train)


# In[17]:


accuracy_score(y_test,lr.predict(x_test))


# In[18]:


accuracy_score(y_train,lr.predict(x_train))


# ##### Defining another data

# In[19]:


from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
data1 = pd.DataFrame(data.data,columns=data.feature_names)
data1['Target'] = pd.DataFrame(data.target)


# In[20]:


data1.head()


# In[21]:


data1['Target'].value_counts()


# In[22]:


x = data1.drop(['Target'],axis=1)
y = data1['Target']


# In[23]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=125)


# In[24]:


dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)


# In[25]:


plt.figure(figsize=(15,10))
tree.plot_tree(dt,filled=1)


# In[26]:


params = {'criterion':['gini','entropy'],
         'max_depth':range(1,10),
         'max_leaf_nodes':range(1,10)}
model = GridSearchCV(dt,param_grid=params,cv=5,verbose=1)
model.fit(x_train,y_train)


# In[27]:


model.best_params_


# In[28]:


model.best_score_


# In[29]:


accuracy_score(y_test,model.predict(x_test))


# In[30]:


accuracy_score(y_train,model.predict(x_train))

