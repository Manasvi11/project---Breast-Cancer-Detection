#!/usr/bin/env python
# coding: utf-8

# # Breast Cancer Detection
# 

# In[5]:


# import libraries
import pandas as pd # for data manupulation or analysis
import numpy as np # for numeric calculation
import matplotlib.pyplot as plt # for data visualization
import seaborn as sns # for data visualization
import sklearn


# In[6]:


df=pd.read_csv(r'C:\Users\manas\Downloads\data.csv')
df.head()


# In[7]:


df.shape


# In[8]:


df.info()


# In[9]:


df.diagnosis.value_counts()


# In[6]:


cancer_dataset['target']
cancer_dataset['target_names']


# In[10]:


sns.countplot(df.diagnosis,label="count")
plt.show()


# In[11]:


from sklearn.preprocessing import LabelEncoder
labelEncoder_Y=LabelEncoder()
df.iloc[:,1]=labelEncoder_Y.fit_transform(df.iloc[:,1].values)
sns.pairplot(df.iloc[:,1:6],hue="diagnosis")
plt.show()


# In[12]:


df.iloc[:,1:12].corr()


# In[13]:


x=df.iloc[:,2:31].values
y=df.iloc[:,1].values


# In[14]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.25,random_state=0)


# In[15]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)


# In[16]:


#create a function for the models
def models(x_train,y_train):
  #Logistic Regression Model
  from sklearn.linear_model import LogisticRegression
  log=LogisticRegression(random_state=0)
  log.fit(x_train,y_train)
  
  #Decision Tree
  from sklearn.tree import DecisionTreeClassifier
  tree=DecisionTreeClassifier(criterion='entropy',random_state=0)
  tree.fit(x_train,y_train)
  
  #Random Forest Classifier
  from sklearn.ensemble import RandomForestClassifier
  forest = RandomForestClassifier(n_estimators=10,criterion="entropy",random_state=0)
  forest.fit(x_train,y_train)

  #Print the models accuracy on the training data
  print("[0]Logistic Regression Training Accuracy:",log.score(x_train,y_train))
  print("[1]Decision Tree Classifier Training Accuracy:",tree.score(x_train,y_train))
  print("[2]Random Forest Classifier Training Accuracy:",forest.score(x_train,y_train))
  
  return log,tree,forest


# In[17]:


model = models(x_train,y_train)


# In[18]:


#test model accuracy on confusion matrix
from sklearn.metrics import confusion_matrix

for i in range(len(model)):
  print("Model ", i)
  cm =confusion_matrix(y_test,model[i].predict(x_test))

  TP=cm[0][0]
  TN=cm[1][1]
  FN=cm[1][0]
  FP=cm[0][1]

  print(cm)
  print("Testing Accuracy = ", (TP+TN) / (TP+TN+FN+FP))
  print()


# In[19]:


#print the prediction of random forest classifier model
pred=model[2].predict(x_test)
print(pred)
print()
print(y_test)


# In[ ]:




