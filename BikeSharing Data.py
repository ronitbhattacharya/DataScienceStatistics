#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
df1=pd.read_csv('C:/Users/personal/Desktop/MBA/day.csv')
df2=pd.read_csv('C:/Users/personal/Desktop/MBA/hour.csv')
#Merging the two dataFrames
df2=pd.concat([df1,df2],axis=0)
df2.reset_index(inplace=True)
df2.drop('index',inplace=True,axis=1)
df2.drop('instant',inplace=True,axis=1)
##Replacing the Nan Values#########
df2.fillna(0,inplace=True)
#Adding column of 1s to do matrix calculations######
df2 = pd.concat([pd.Series(1, index=df2.index, name='00'), df2], axis=1)
df2.head()
df2.fillna(1,inplace=True)
X=df2.iloc[:,2:12] #Independent Variables
X = pd.concat([pd.Series(1, index=df2.index, name='00'), X], axis=1)
X.fillna(1,inplace=True)
Y=df2['cnt'] #Dependent Varaibles
X.rename(columns = {'00':'one'}, inplace = True)
X=X.values  #Converting DataFrame to 2d array
Y=Y.values

##Multivariate Linear Regression#####

def LinReg_with_gradient_descent(X,y,alpha,epoch):
    m=X.shape[0] #no of samples
    n=X.shape[1] #no of thetas 
    Theta = np.ones(n)  #Initializing theta variable
    h = np.dot(X,Theta)  
    cost=np.ones(epoch)
    for i in range(0,epoch):
        Theta[0]= Theta[0]-(alpha/X.shape[0])*(np.sum(h-y))
        for j in range(1,n):
            Theta[j]=Theta[j]-(alpha/X.shape[0])*np.sum((h-y)*X[:,j])
        h = np.dot(X,Theta)
        cost[i]=1/(2*m)*np.sum((np.square(h-y))) #Cost Function                      
    return  cost,Theta    


# In[2]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.5,random_state=12345)


# In[3]:


cost, Theta= LinReg_with_gradient_descent(X_train,Y_train,0.001,1000)


# In[4]:


Theta


# In[5]:


###Cost Function is decreasing with no of iterations######


import matplotlib.pyplot as plt
plt.plot(cost)
plt.xlabel("Epoch No of Iterations")
plt.ylabel("Cost Function")
print("Lowest Cost =", + (np.min(cost)))
print("Cost after 1000 iterations=" +str(cost[-1]))
plt.show()


# In[6]:


####Checking performance of Model#####
Y_pred = np.dot(X_test,Theta)


# In[7]:


####Putting it in a dataframe to compare values######
finalpredicteddf=pd.DataFrame(Y_test)
finalpredicteddf['1']=pd.DataFrame(Y_pred)
finalpredicteddf.columns=['Y_test','Y_pred']


# In[8]:


finalpredicteddf


# In[9]:


from sklearn.metrics import r2_score
r2_score(Y_test,Y_pred)


# In[10]:


from sklearn.metrics import mean_squared_error
a=mean_squared_error(Y_test,Y_pred)
rmse=np.sqrt(a)
rmse


# ######Hence Linear Regression model is not a good fit for this data. Although the cost function is decreasing and more no of iterations and learning rate can be changed but still model will not predict values accurately since there is no linear relationship between any of the variables####
# 

# In[11]:


##Going for no linear technique####
##Demo Random Forest Model####


 
import numpy as np
import seaborn
import numpy as np
import matplotlib.pyplot as matplotlib
 
from sklearn.ensemble import RandomForestRegressor
 
from matplotlib.lines import Line2D
from scipy.stats import pearsonr
 
# set seed to make results reproducible
RF_SEED = 30


# In[37]:


df2.describe()


# In[12]:


X=df2.iloc[:,2:12] #Independent Variables
X = pd.concat([pd.Series(1, index=df2.index, name='00'), X], axis=1)
X.fillna(1,inplace=True)
Y=df2['cnt'] #Dependent Varaibles
X.rename(columns = {'00':'one'}, inplace = True)


# In[13]:


Y=df2['cnt']


# In[28]:


df2


# In[31]:


X['cnt']=Y


# In[32]:


X


# In[34]:


import seaborn as sns
# Create the default pairplot
sns.pairplot(X,vars = ['temp', 'atemp', 'hum'])


# In[15]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.5,random_state=12345)


# In[16]:


feature_names=X_train.columns


# In[17]:


feature_names


# In[18]:


regressor = RandomForestRegressor(n_estimators=1000, random_state=RF_SEED)
regressor.fit(X_train, Y_train)


# In[19]:


predictions = regressor.predict(X_test)


# In[20]:


####Putting it in a dataframe to compare values######
finalpredicteddf=pd.DataFrame(Y_test)
finalpredicteddf['1']=pd.DataFrame(Y_pred)
finalpredicteddf.columns=['Y_test','predictions']


# In[21]:


# find the correlation between real answer and prediction
finalpredicteddf.corr()


# In[22]:


plt.scatter(finalpredicteddf['Y_test'],finalpredicteddf['predictions'])
plt.xlabel='Actual Y'
plt.ylabel='Predicted Y'


# In[23]:


Y_test.shape


# In[24]:


#####Checking performance of Model######
errors = abs(predictions - Y_test)
mape = 100 * (errors / Y_test)
meanmape=np.sum(mape)/Y_test.shape[0]
a=np.sqrt(meanmape)
accuracy = 100 - a
print('Accuracy:', round(accuracy, 2), '%.')


# In[25]:


####Feature Importance#######
features_importance = regressor.feature_importances_
featurenames=[]
importance=[]
print("Feature ranking:")
for i, data_class in enumerate(feature_names):
    featurenames.append(data_class)
    importance.append(features_importance[i])
    print("{}. {} ({})".format(i + 1, data_class, features_importance[i]))


# In[26]:


#####Making final dataframe of Feature Importance#####
featureimportancedf=pd.DataFrame(featurenames)
featureimportancedf['1']=pd.DataFrame(importance)
featureimportancedf.columns=['FeatureLabels','Importance']
featureimportancedf


# In[27]:


featureimportancedf.plot.bar()

