#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic('config', "InclinedBackend.figure_format ='retina'")


# In[2]:


train_data = pd.read_csv('loan_train.csv')
test_data = pd.read_csv('loan_test.csv')


# In[ ]:





# In[3]:


train_data.head()


# In[4]:


train_data.shape


# In[5]:


test_data.head()


# In[6]:


test_data.shape


# In[7]:


test_data['Loan_Status'] = 0
test_data.columns


# In[8]:


test_data.shape


# In[9]:


comb_data = train_data.append(test_data)


# In[10]:


print(comb_data.shape)


# In[11]:


comb_data.head()


# In[12]:


comb_data['Gender'].value_counts()


# In[13]:


comb_data['Gender'].isna().sum()


# In[14]:


comb_data.describe(include='object')


# In[15]:


comb_data['Gender'].value_counts(dropna=False)


# In[16]:


comb_data['Gender'].fillna('Male',inplace=True)
comb_data['Gender'].value_counts(dropna=False)


# In[17]:


comb_data['Married'].value_counts(dropna=False)


# In[18]:


comb_data['Married'].fillna('Yes',inplace=True)


# In[19]:


comb_data['Married'].value_counts(dropna=False)


# In[20]:


comb_data['Dependents'].value_counts(dropna=False)


# In[21]:


comb_data['Dependents'].fillna('0',inplace=True)
comb_data['Dependents'].value_counts(dropna=False)


# In[22]:


comb_data['Self_Employed'].value_counts(dropna=False)
comb_data['Self_Employed'].fillna('No',inplace=True)
comb_data['Self_Employed'].value_counts(dropna=False)


# In[23]:


comb_data['LoanAmount'].fillna(comb_data['LoanAmount'].mean(),inplace=True)
comb_data.info()


# In[24]:


comb_data['Loan_Amount_Term'].fillna(comb_data['Loan_Amount_Term'].median(),inplace=True)
comb_data.info()


# In[25]:


comb_data['Credit_History'].value_counts(dropna=False)
comb_data['Credit_History'].fillna(2.0,inplace=True)
comb_data['Credit_History'].value_counts(dropna=False)


# In[26]:


comb_data.info()


# In[27]:


comb_data.isnull().sum()


# In[28]:


comb_data.describe(include='object')


# In[29]:


comb_data['Gender']=comb_data['Gender'].map({'Male':1 , 'Female':0})


# In[30]:


comb_data['Married']=comb_data['Married'].map({'Yes':1, 'No':0})


# In[31]:


comb_data['Education'] = comb_data['Education'].map({'Graduate':1,'Not Graduate':0})


# In[32]:


comb_data['Self_Employed']=comb_data['Self_Employed'].map({'No':0 , 'Yes':1})


# In[33]:


comb_data.head()


# In[34]:


comb_data.describe()


# In[35]:


comb_data.describe(include='object')


# In[36]:


comb_data['Total Income']=comb_data['ApplicantIncome']+comb_data['CoapplicantIncome']
comb_data.drop(['ApplicantIncome','CoapplicantIncome'],axis=1,inplace=True)


# In[37]:


pd.crosstab(comb_data['Loan_Amount_Term'], comb_data['Loan_Status'], margins=True)


# In[38]:


#imp features : education,property_area,creditHistory
plt.scatter(comb_data['Total Income'],comb_data['LoanAmount'])


# In[39]:


comb_data['debt_ratio']=comb_data['Total Income']/comb_data['LoanAmount']


# In[40]:


comb_data.head()


# In[41]:


comb_data['LoanAmount']=np.log10(comb_data['LoanAmount'])
comb_data['Total Income']=np.log10(comb_data['Total Income'])
comb_data['debt_ratio']=np.log10(comb_data['debt_ratio'])
comb_data.head()


# In[42]:


#finding corrrelation matrix
col=['LoanAmount','Total Income','debt_ratio']
corr_matrix= comb_data[col].corr()
corr_matrix


# In[43]:


#scailing the feature
for i in range(len(col)):
    comb_data[col[i]]=(comb_data[col[i]]-comb_data[col[i]].min())/(comb_data[col[i]].max()-comb_data[col[i]].min())
comb_data.head()


# In[44]:


train_data = comb_data[comb_data['Loan_Status']!=0]
test_data =comb_data[comb_data['Loan_Status']==0]


# In[45]:


train_data['Loan_Status']= train_data['Loan_Status'].map({'Y':1,'N':0})


# In[46]:


train_data['Loan_Status'].value_counts()


# In[47]:


#checking distplot of scailing variables
import seaborn as sns
sns.distplot(train_data['debt_ratio'])


# In[48]:


#feature_selection
train_features=train_data.drop(['Loan_Status','Loan_ID','Dependents','Property_Area'],axis=1)
train_labels=train_data['Loan_Status']
test_features=test_data.drop(['Loan_Status','Loan_ID','Dependents','Property_Area'],axis=1)
test_labels=test_data['Loan_Status']


# In[49]:


corr_matrix=train_features[train_features.columns].corr()
sns.heatmap(corr_matrix)


# In[50]:


train_features.columns


# In[ ]:





# In[51]:


from sklearn.feature_selection import SelectKBest
x = SelectKBest(k=6)
best_features = x.fit(train_features,train_labels)
features = best_features.transform(train_features)
y = pd.DataFrame({'Features':list(train_features.columns),'scores':best_features.scores_})
y.sort_values(by='scores')


# In[52]:


train_features=['Credit_History','Married','Education','debt_ratio','LoanAmount']
train_features= train_data[train_features]
train_features.head()


# In[60]:


from sklearn import tree
from sklearn.naive_bayes import GaussianNB
#clf=tree.DecisionTreeClassifier()
clf = GaussianNB()
clf.fit(train_features,train_labels)
pred=clf.predict(train_features)
from sklearn.metrics import accuracy_score
acc =  accuracy_score(train_labels,pred)
print(acc)


# In[54]:


#svm = 0.68
#GaussianNB = 0.788
#decisontreeclassifier=1


# In[55]:


clf


# In[56]:


from sklearn import tree
clf=tree.DecisionTreeClassifier(max_depth=10,min_samples_split=20,)
clf.fit(train_features,train_labels)
pred=clf.predict(train_features)
from sklearn.metrics import accuracy_score
acc =  accuracy_score(train_labels,pred)
print(acc)


# In[57]:


from sklearn import tree
clf=tree.DecisionTreeClassifier(max_depth=10,min_samples_split=20,criterion='entropy')
clf.fit(train_features,train_labels)
pred=clf.predict(train_features)
from sklearn.metrics import accuracy_score
acc =  accuracy_score(train_labels,pred)
print(acc)


# In[58]:


from sklearn import tree
clf=tree.DecisionTreeClassifier(max_depth=10,min_samples_split=18,criterion='entropy')
clf.fit(train_features,train_labels)
pred=clf.predict(train_features)
from sklearn.metrics import accuracy_score
acc =  accuracy_score(train_labels,pred)
print(acc)


# In[62]:





# In[ ]:




