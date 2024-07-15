#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# ## Reading the Files

# In[2]:


data_new = pd.read_csv('INNHotelsGroup_newdata.csv')
data_past = pd.read_csv('INNHotelsGroup_pastdata.csv')


# In[3]:


data_past.shape


# In[4]:


data_new.shape


# In[5]:


data_new.info()


# In[6]:


data_past.info()


# ### Descriptive Statistics

# In[7]:


data_past.describe().T


# #### Inference
# * we can infer from the data that we can suspect outliers on the upper tail of lead time,avg price per room
#   and no.of.week nights.

# In[8]:


# let us look at the correlation plot as well
sns.heatmap(data_past.corr(numeric_only=True),vmax=1,vmin=-1,annot=True,cmap='RdBu')
plt.show()


# #### Inference
# 
# * There is some positive correlation between average price per room and no.of.adults.
# * Weak Positive Correlation between no.of.special request and no.of.adults.
# * Weak Positive Correlation between no.of.week nights and weekend nights.
# * Weak Positive Correlation between average price per room and special request.

# ### Data Visualization and Data Preprocessing

# In[9]:


# Can you tell me what % of cancelled booking were rebooked?
rebooked = data_past[data_past['booking_status'] == 'Canceled']['rebooked']
rebooked


# In[10]:


data_past['booking_status'].value_counts().plot(kind = 'pie',autopct='%.2f%%')
plt.show()


# In[11]:


rebooked.value_counts().plot(kind = 'pie',autopct='%.2f%%')
plt.show()


# #### Inference
# * There were 33% bookings which were cancelled at the last minute.Out of that 80% of the
#   cancelled bookings were not rebooked.That is the main reason which company is incurring losses.

# In[12]:


# Now we do not need the rebooked column in our analysis,hence we can drop it
data_past.drop(columns=['rebooked'],inplace=True)


# In[13]:


# past data has to be used as training set and new data as test set
# for now we will store y_test in separate variable for doing validation Later on.
# We also need to combine train and test data over here in order to preprocess.


# In[14]:


y_test = data_new['booking_status']
data_new.drop(columns=['booking_status'],inplace = True)


# In[15]:


data = pd.concat([data_past,data_new],axis=0)


# In[16]:


data.shape


# In[17]:


data.isnull().sum()


# #### Drop the duplicates

# In[18]:


data[data.duplicated()]


# In[19]:


data.info()


# In[20]:


data['arrival_date']=pd.to_datetime(data['arrival_date'],format='%Y-%m-%d')


# In[21]:


data['arrival_day'] = data['arrival_date'].dt.day
data['arrival_month'] = data['arrival_date'].dt.month
data['arrival_weekday'] = data['arrival_date'].dt.weekday


# In[22]:


data.drop(columns=['arrival_date','booking_id'],inplace=True)


# In[23]:


data.head(3)


# ## Visualization

# In[24]:


data.columns


# In[25]:


num_cols = ['lead_time','avg_price_per_room','arrival_day']
cat_cols = data.drop(columns=num_cols).columns


# In[26]:


num_cols


# In[27]:


cat_cols


# ## Univariate Analysis

# #### Numerical Columns

# In[28]:


for i in num_cols:
    sns.distplot(data[i])
    plt.show()


# ## Inference
# * Lead time is highly right skewed.
# * Average is highly right skewed with 0 price in some entries.
# * Although arrival day is almost uniform,but there is no data in 15th Day.
# 

# In[29]:


for i in num_cols:
    sns.boxplot(data[i],orient='h')
    plt.show()


# ### Inference
# * There are extreme outliers in lead time and average price per room.

# #### Categorical Columns

# In[30]:


for i in cat_cols:
    sns.countplot(x=data[i])
    plt.show()


# ## Inference
# * Most of the bookings are online.
# * Either there are no requests or 1 requests.
# * Most of the bookings have been made by couples.
# * Most of the customers have 0,1 or 2 weekend nights in their stay.
# * Very few customers requested for car parking space.
# * Very few customers are having more than 5 week nights in their stay.
# * Maximum arrivals are in month March and April and arrival day is Friday.

# ## Bivariate Analysis

# #### Numerical Columns vs Categorical Columns

# In[31]:


for i in num_cols:
    sns.displot(data=data,x=i,hue=data['booking_status'],kind='kde')
    plt.show()


# ## Inference
# * In lead time and average price for more extreme values the booking is canceled.

# #### Categorical vs Categorical columns

# In[32]:


for i in cat_cols:
    if i != 'booking_status':
        pd.crosstab(data[i],data['booking_status']).plot(kind='bar')
        plt.show()


# ## Missing value Treatment

# In[33]:


data.isnull().sum()


# ## Outlier Treatment

# In[34]:


# We will treat outliers from test data only,As it might lead to data Leakage.


# ## Encoding

# In[35]:


data.head()


# In[36]:


# Online = 1,Offline = 0
data['market_segment_type'] = data['market_segment_type'].map({'Online' : 1,'Offline' : 0})


# In[37]:


data['market_segment_type'].value_counts()


# In[38]:


# Canceled = 1,Not Canceled = 0
data['booking_status'] = data['booking_status'].map({'Canceled' : 1,'Not Canceled' : 0})


# In[39]:


data['booking_status'].value_counts()


# In[40]:


y_test = y_test.map({'Canceled' : 1,'Not Canceled' : 0})


# In[41]:


y_test.value_counts()


# ## Train Test Split

# In[42]:


x_test = data[data['booking_status'].isnull()]


# In[43]:


x_test.drop(columns=['booking_status'],inplace=True)


# In[44]:


x_test.shape


# In[45]:


train = data[data['booking_status'].notnull()]


# In[46]:


train.shape


# In[47]:


# Lets drop duplicates from train
train.drop_duplicates(inplace=True)


# In[48]:


train.shape


# In[49]:


# Cap the extreme outliers

for i in ['lead_time','avg_price_per_room']:
    q3,q1 = np.quantile(train[i],[0.75,0.25])
    IQR = q3-q1
    ul,ll = q3+2.5*IQR,q1-2.5*IQR
    train[i] = np.where((train[i]>ul),ul,train[i])
    train[i] = np.where((train[i]<ll),ll,train[i])


# In[50]:


for i in ['lead_time','avg_price_per_room']:
    sns.boxplot(train[i],orient = 'h')
    plt.show()


# In[51]:


x_train = train.drop(columns='booking_status')
y_train = train['booking_status']


# In[52]:


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# ## Predictive Modelling

# In[53]:


from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier,StackingClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,cohen_kappa_score
from sklearn.metrics import classification_report,roc_auc_score,roc_curve

from sklearn.model_selection import GridSearchCV


# In[54]:


mod = []
acc = []
pre = []
rec = []
f1 = []
ck = []

def model_validation(model,xtrain,ytrain,xtest,ytest):
    m = model
    m.fit(xtrain,ytrain)
    hard = m.predict(xtest)
    soft = m.predict_proba(xtest)[:,1]
    
    print('classification report\n',classification_report(ytest,hard))
    fpr,tpr,th = roc_curve(ytest,soft)
    plt.title(f'ROC AUC: {round(roc_auc_score(ytest,soft),3)}')
    plt.plot(fpr,tpr)
    plt.plot([0,1],[0,1],ls='--',color='red')
    plt.show()
    
    inp = input('Do you want to save the model in score card? Y/N : ')
    if inp.lower() == 'y':
        global scorecard
        mod.append(str(model))
        acc.append(accuracy_score(ytest,hard))
        pre.append(precision_score(ytest,hard))
        rec.append(recall_score(ytest,hard))
        f1.append(f1_score(ytest,hard))
        ck.append(cohen_kappa_score(ytest,hard))
        scorecard = pd.DataFrame({'Model' : mod,
                                  'Accuracy': acc,
                                  'Precision' : pre,
                                  'Recall' : rec,
                                  'F1 Score' : f1,
                                  'Cohen Kappa' : ck})
    else:
        return


# ### Logistic Regression

# In[55]:


model_validation(LogisticRegression(),x_train,y_train,x_test,y_test)


# In[56]:


scorecard


# ### Decision Tree

# In[57]:


gscv = GridSearchCV(estimator=DecisionTreeClassifier(),param_grid={'max_depth' : [4,5,6,7,8]},
                   cv=5,scoring='f1')


# In[58]:


gscv.fit(x_train,y_train)


# In[59]:


gscv.best_score_


# In[60]:


gscv.best_params_


# In[61]:


model_validation(DecisionTreeClassifier(**gscv.best_params_),x_train,y_train,x_test,y_test)


# In[62]:


scorecard


# ### Random Forest

# In[63]:


model_validation(RandomForestClassifier(max_depth=8,n_estimators = 10),x_train,y_train,x_test,y_test)


# In[64]:


scorecard


# ### Ada Boost

# In[65]:


model_validation(AdaBoostClassifier(n_estimators=120,learning_rate=0.5),x_train,y_train,x_test,y_test)


# In[66]:


scorecard


# ### Gradient Boost

# In[67]:


model_validation(GradientBoostingClassifier(max_depth=6,n_estimators=150),x_train,y_train,x_test,y_test)


# In[68]:


scorecard


# ### XGBoost

# In[69]:


model_validation(XGBClassifier(n_estimators=120,max_depth=4),x_train,y_train,x_test,y_test)


# In[70]:


scorecard


# ### Voting

# In[71]:


base_learners = [('DT_4',DecisionTreeClassifier(max_depth=4)),
                 ('DT_6',DecisionTreeClassifier(max_depth=6)),
                 ('RF',RandomForestClassifier(n_estimators=120,max_depth=5)),
                 ('XGB',XGBClassifier(max_depth=4,n_estimators=100))]


# In[72]:


model_validation(VotingClassifier(estimators=base_learners,voting='soft'),x_train,y_train,x_test,y_test)


# In[73]:


scorecard


# ### Stacking

# In[74]:


model_validation(StackingClassifier(estimators=base_learners),
                x_train,y_train,x_test,y_test)


# #### Final Scorecard

# In[75]:


scorecard


# Lets Go with GBM and tune it for final prediction

# In[76]:


param = {'n_estimators' : [70,100,120,150],
         'learning_rate' : [1,0.5,0.1],
         'max_depth' : [3,4,5,6,7,8]}


# In[77]:


gscv = GridSearchCV(estimator=XGBClassifier(),param_grid=param,scoring='f1',verbose=1)


# In[78]:


gscv.fit(x_train,y_train)


# In[79]:


gscv.best_params_


# In[80]:


model_validation(XGBClassifier(**gscv.best_params_),x_train,y_train,x_test,y_test)


# In[81]:


scorecard


# ### Final Model

# In[82]:


final_model= XGBClassifier(**gscv.best_params_)


# In[83]:


final_model.fit(x_train,y_train)


# In[84]:


# Prediction


# In[85]:


x_test.head(2)


# In[86]:


final_model.predict_proba([[10,1,0,170,2,2,1,1,12,4,4]])[:,1][0]


# In[5]:


import pickle
import gradio as gr    # pip install gradio
import numpy as np
import pandas as pd


# In[6]:


with open('final_model.pkl','rb') as file:
    model = pickle.load(file)


# In[7]:


def prediction(lt,mark,spcl,price,noa,wends,parking,wnights,a_day,a_month,a_wday):
    
    input = [[lt,mark,spcl,price,noa,wends,parking,wnights,a_day,a_month,a_wday]]
    prediction = model.predict_proba(input)[:,1][0]
    
    return round(prediction,3)


# In[8]:


prediction(20,1,1,120,1,2,0,1,1,1,4)


# # INTERFACE

# In[9]:


iface = gr.Interface(fn=prediction,
                     inputs=[gr.Number(label='How many days prior booking was made?'),
                             gr.Dropdown([('Online',1),('Offline',0)],label = 'Booking was Online/Offline?'),
                             gr.Dropdown([0,1,2,3,4,5],label = 'How many Special Requests?'),
                             gr.Number(label='What is the Room Price?'),
                             gr.Dropdown([1,2,3,4,5],label='Count of Adults?'),
                             gr.Number(label='How many weekends in the stay?'),
                             gr.Dropdown([('yes',1),('No',0)],label='Does customer require parking?'),
                             gr.Number(label='How many weeknights in the stay?'),
                             gr.Slider(minimum=1,maximum=31,step=1,label='Day of arrival?'),
                             gr.Slider(minimum=1,maximum=12,step=1,label='Month of arrival?'),
                             gr.Dropdown([('Mon',0),('Tue',1),('Wed',2),('Thus',3),('Fri',4),('Sat',5),('Sun',6)],
                                         label = 'Weekday of arrival')],
                     
                     outputs = gr.Textbox(label = 'Chances of getting this booking canceled'),
                     title = 'INN Hotel Bookings',
                     description = 'This app will predict the chances of cancellation',
                     allow_flagging = 'never')
                     


# In[10]:


iface.launch()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




