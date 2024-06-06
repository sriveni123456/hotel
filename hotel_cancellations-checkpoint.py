#!/usr/bin/env python
# coding: utf-8

# # Predicting Hotel Cancellations

# ## Problem Statement: <br>
# As a data scientist, you are supporting a hotel with a project aimed at increasing revenue from their room bookings. They believe that they can use data science to help them reduce the number of cancellations. You are to use any appropriate methodology to identify what contributes to whether a booking will be fulfilled or canceled. The results of your work will be used to reduce the chance someone cancels their booking.

# ## Goal: <br>
#     1.To develop/build a system/web app that predicts whether a hotel booking will be fulfilled or canceled.
#     2.To determine the factors with high importance in predicting whether a hotel booking will be fulfilled or canceled.

# ### Importing libraries

# In[1]:


get_ipython().system('pip install xgboost')


# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import xgboost
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report, ConfusionMatrixDisplay

from sklearn.model_selection import RandomizedSearchCV


# ### Loading the data

# In[3]:


df = pd.read_csv(r"C:\Users\Lenovo\Downloads\hotel_bookings.csv")
df.head()


# In[4]:


df.shape


# In[5]:


df.info()


# ### Data cleaning and validation

# In[6]:


df.isna().sum()


# In[7]:


df = df.dropna(axis=0)


# In[8]:


df.shape


# In[9]:


df.duplicated().sum()


# In[10]:


df.describe()


# In[11]:


df['arrival_year'].value_counts()


# the dataset only cover two years - 2017 and 2018

# ### Exploratory Data Analysis

# #### Univariate Analysis

# In[12]:


sns.kdeplot(df['avg_price_per_room'], fill=True)
plt.xlabel('Average price per room')
plt.title('Distribution of the average price per day of the reservation')
plt.show()


# the distribution is approximately right skewed as most of the prices are kind of betwwen 50-200. This is kind of okay since the variable is calculated per day.

# In[13]:


sns.countplot(x='room_type_reserved',data=df, color='steelblue')
plt.xticks(rotation=90)
plt.ylabel('Number of bookings')
plt.xlabel('Room type')
plt.title('Number of bookings for each room type')
plt.show()


# the *room_type 1* has the highest number of bookings with over 20000 bookings.

# In[14]:


sns.countplot(x='type_of_meal_plan', data=df, color='steelblue')
plt.xlabel('Meal plan')
plt.ylabel('Frequency')
plt.title('Meal plan selected during bookings')
plt.show()


# the *meal plan 1* was mostly selected during bookings. <br>
# N.B.; the *not selected* was not filtered out because it's possible to not select a meal plan when making a booking.

# In[15]:


sns.countplot(x='market_segment_type', data=df, color='steelblue')
plt.xlabel('Market segment type')
plt.ylabel('Number of bookings')
plt.title('How the booking was made')
plt.show()


# Most bookings were made online, least is avaiation medium.

# In[16]:


car_space_yes = df['required_car_parking_space'].sum()
required_car_space = ['Yes', 'No']
data = [car_space_yes, (len(df)-car_space_yes)]
data


# In[17]:


plt.pie(data, labels=required_car_space, autopct='%1.1f%%', colors=['Lightblue', 'Steelblue'])
plt.show()


# 96.9% indicated no need for car parking space, while 3.1% indicated need for car parking space.

# In[18]:


sns.countplot(x='arrival_month', data=df, color='Steelblue')
plt.xlabel('Month')
plt.ylabel('Number of bookings')
plt.title('Number of bookings made per month')
x = range(0, 12)
labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
plt.xticks(x, labels, rotation=90)
plt.show()


# October has the highest number of bookings with over 4000 bookings made, while the January has the least amount number of bookings.

# #### Bivariate Analysis

# In[19]:


sns.countplot(x='room_type_reserved', hue='booking_status', data=df)
plt.xticks(rotation=90)
plt.ylabel('Number of bookings')
plt.xlabel('Room type')
plt.title('Relationship between the type of room reserved and booking status')
plt.show()


# In[20]:


sns.boxplot(x='booking_status', y='avg_price_per_room', data=df, sym='')
plt.xlabel('Booking status')
plt.ylabel('Average price per room')
plt.title('Relationship between the booking status and average price per room')
plt.show()


# there is no much observable relationship between the average price per day for a room and the booking status

# In[21]:


sns.boxplot(x='booking_status', y='lead_time', data=df, sym='')
plt.xlabel('Booking status')
plt.ylabel('Lead time')
plt.title('Relationship between the booking status and lead time')
plt.show()


# there is an observable difference as canceled bookings have more lead time than bookings that were not canceled.

# In[22]:


sns.countplot(x='repeated_guest',hue='booking_status', data=df)
plt.ylabel('Number of bookings')
plt.xlabel('Repeated guest')
plt.title('Relationship between the repeated guest and booking status')
x = range(2)
labels = ['Not repeated', 'Repeated']
plt.xticks(x, labels)
plt.show()


# In[23]:


df.groupby(['repeated_guest','booking_status'])[['booking_status']].count()


# In[24]:


df['no_of_previous_cancellations'].value_counts()


# ### Feature preprocessing

# _make a copy of the dataset_

# In[25]:


model_data = df.copy()


# In[26]:


model_data.columns


# #### Feature engineering

# In[27]:


model_data['no_of_individuals'] = model_data['no_of_adults'] + model_data['no_of_children']


# In[28]:


model_data['no_of_days_booked'] = model_data['no_of_weekend_nights'] + model_data['no_of_week_nights']


# engineered new columns to the data

# In[29]:


model_data.head()


# #### Feature encoding

# In[30]:


cat_features = ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type']


# In[31]:


lab = LabelEncoder()
for col in cat_features:
    model_data[col] = lab.fit_transform(model_data[col])


# In[32]:


model_data.head()


# In[33]:


model_data['market_segment_type'].value_counts()


# In[34]:


df['arrival_year'].value_counts()


# In[35]:


model_data['booking_status'] = model_data['booking_status'].replace({'Canceled':0, 'Not_Canceled':1})


# In[36]:


model_data['booking_status'].value_counts()


# In[37]:


sns.countplot(x='booking_status', data=df)
plt.show()


# #### Feature correlation

# In[38]:


features = model_data.drop(['Booking_ID', 'booking_status'], axis=1)
features.head()


# In[39]:


plt.figure(figsize=(20,10))
sns.heatmap(features.corr(), annot=True, cmap='GnBu_r')
plt.title('Heatmap showing correlation between features')
plt.show()


# ### Train-test split validation

# In[40]:


X = features
y = model_data['booking_status']


# In[41]:


print(X.shape, y.shape)


# In[42]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[43]:


print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# ### Feature scaling

# In[44]:


scaler = StandardScaler()
scaledX_train = scaler.fit_transform(X_train)
scaledX_test = scaler.transform(X_test)


# ### Modelling

# #### Logistic Regression

# In[45]:


logit = LogisticRegression()
logit.fit(scaledX_train, y_train)


# In[46]:


pred_logit = logit.predict(scaledX_test)
pred_logit


# In[47]:


print(f'The accuracy score of the logistic regression model is {accuracy_score(y_test, pred_logit)}')


# In[48]:


print(f'The f1_score of the logistic regression model is {f1_score(y_test, pred_logit)}')


# In[49]:


print(classification_report(y_test, pred_logit))


# In[50]:


confusion_matrix_log = confusion_matrix(y_test, pred_logit)
cm_display_log = ConfusionMatrixDisplay(confusion_matrix = confusion_matrix_log, display_labels = ['Canceled', 'Not canceled'])
cm_display_log.plot(cmap=plt.cm.Blues)

plt.savefig('logcm.png', bbox_inches='tight', dpi=300)
plt.show()


# #### Random forest

# In[51]:


forest = RandomForestClassifier(random_state = 1)
forest.fit(scaledX_train, y_train)


# In[52]:


pred_forest = forest.predict(scaledX_test)
pred_forest


# In[53]:


print(f'The accuracy score of the random forest model is {accuracy_score(y_test, pred_forest)}')


# In[54]:


print(f'The f1_score of the random forest model is {f1_score(y_test, pred_forest)}')


# In[55]:


print(classification_report(y_test, pred_forest))


# In[56]:


confusion_matrix_forest = confusion_matrix(y_test, pred_forest)
cm_display_forest = ConfusionMatrixDisplay(confusion_matrix = confusion_matrix_forest, display_labels = ['Canceled', 'Not canceled'])
cm_display_forest.plot(cmap=plt.cm.Blues)

plt.savefig('forestcm.png', bbox_inches='tight', dpi=300)
plt.show()


# #### Xgboost

# In[57]:


boost = XGBClassifier(random_state=23)
boost.fit(scaledX_train, y_train)


# In[58]:


pred_boost = boost.predict(scaledX_test)
pred_boost


# In[59]:


print(f'The f1_score of the xgboost model is {f1_score(y_test, pred_boost)}')


# In[60]:


print(classification_report(y_test, pred_boost))


# In[61]:


confusion_matrix_boost = confusion_matrix(y_test, pred_boost)
cm_display_boost = ConfusionMatrixDisplay(confusion_matrix = confusion_matrix_boost, display_labels = ['Canceled', 'Not canceled'])
cm_display_boost.plot(cmap=plt.cm.Blues)

plt.savefig('boostcm.png', bbox_inches='tight', dpi=300)
plt.show()


# ### Hyperparameter Tuning

# In[62]:


param_grid = {'n_estimators':[100, 300, 500], 'max_depth':[5, 10, 15], 'max_features':['sqrt', 'log2', None]}


# In[63]:


tuned_forest = RandomForestClassifier(max_depth=15, max_features=None, n_estimators=500)
tuned_forest.fit(scaledX_train, y_train)


# In[64]:


pred_tunedforest = tuned_forest.predict(scaledX_test)


# In[65]:


print(f'The f1_score of the random forest model is {f1_score(y_test, pred_tunedforest)}')


# In[66]:


print(f'The accuracy score of the random forest model is {accuracy_score(y_test, pred_tunedforest)}')


# In[67]:


print(classification_report(y_test, pred_tunedforest))


# ### Feature Importance

# In[68]:


importance = list(forest.feature_importances_)
importance


# In[69]:


feature = list(X_train.columns)
feature_importance = list(zip(feature, importance))
feature_importance_df = pd.DataFrame(feature_importance, columns=['Feature', 'Importance']).sort_values(ascending = False, by= 'Importance')
feature_importance_df


# In[70]:


sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='GnBu_r')
plt.title('Barplot showing the importance of each feature for the prediction system')
plt.show()


# ### Model Deployment

# In[71]:


import pickle


# Specify the file path where you want to save the model
save_path = "model.pkl"

# Save the model as a pickle file
with open(save_path, 'wb') as file:
    pickle.dump(forest, file)

print("Model saved as pickle file.")


# In[72]:




# Specify the file path where you want to save the scaler
save_path = "scaler.pkl"

# Save the model as a pickle file
with open(save_path, 'wb') as file:
    pickle.dump(scaler, file)

print("Scaler saved as pickle file.")


# In[76]:


import os
os.getcwd()


# In[77]:


pip install streamlit


# In[78]:


import streamlit as st
import numpy as np
import pickle


# In[79]:


import pickle


# Specify the file path where you want to save the model
save_path = "model.pkl"

# Save the model as a pickle file
with open(save_path, 'wb') as file:
    pickle.dump(forest, file)

print("Model saved as pickle file.")


# Specify the file path where you want to save the scaler
save_path = "scaler.pkl"

# Save the scaler as a pickle file
with open(save_path, 'wb') as file:
    pickle.dump(scaler, file)

print("Scaler saved as pickle file.")


# In[80]:


model = pickle.load(open("model.pkl", 'rb'))
scaler = pickle.load(open("scaler.pkl", 'rb'))


# In[81]:


# Define the main function to run your Streamlit app
def main():

    # Add a title and description for your app
    st.title("Predicting Hotel Cancellations using Machine Learning")

    no_of_adults = st.number_input('Number of adults',step=1, min_value=0)
    no_of_children = st.number_input('Number of children',step=1, min_value=0)
    no_of_weekend_nights = st.number_input('Number of weekend nights',step=1, min_value=0)
    no_of_week_nights = st.number_input('Number of week nights', step=1, min_value=0)
    type_of_meal_plan = st.slider('Type of meal plan [0 for meal plan 1, 1 for meal plan 2, 2 for meal plan 3, and 3 if meal plan was not selected]',step=1, min_value=0, max_value=3)
    required_car_parking_space = st.number_input('Whether a car parking space is required [0 for No, 1 for Yes]',step=1,min_value=0, max_value=1)
    room_type_reserved = st.slider('Type of room reserved [0 for room type 1, 1 for room_type 2, 2 fo room type 3, 3 for room type 4, 4 for room type 5, 5 for room type 6, 6 for room type 7]',step=1, min_value=0, max_value=6)
    lead_time = st.number_input('Number of days before the arrival date the booking was made.',step=1, min_value=0)
    arrival_year = st.number_input('Year of arrival', step=1, min_value=2017)
    arrival_month = st.slider('Month of arrival.',step=1, min_value=1, max_value=12)
    arrival_date = st.slider('Date of the month for arrival',step=1, min_value=1, max_value=31)
    market_segment_type = st.slider('How the booking was made [0 for Aviation, 1 for Complementary, 2 for Corporate, 3 for Offline, 4 for Online]',step=1, min_value=0, max_value=4)
    repeated_guest = st.number_input('Whether the guest has previously stayed at the hotel [0 for No, 1 for Yes]',step=1, min_value=0, max_value=1)
    no_of_previous_cancellations = st.number_input('Number of previous cancellations.',step=1, min_value=0)
    no_of_previous_bookings_not_cancelled = st.number_input('Number of previous bookings that were canceled.',step=1, min_value=0)
    avg_price_per_room = st.number_input('Average price per day of the booking.')
    no_of_special_requests = st.number_input('Count of special requests made as part of the booking.',step=1, min_value=0)
    no_of_individuals = st.number_input('The total number of individuals (adults and children)',step=1, min_value=0)
    no_of_days_booked = st.number_input('The total number of nights booked (weekend included)',step=1, min_value=0)




    user_input = [no_of_adults, no_of_children, no_of_weekend_nights, no_of_week_nights, type_of_meal_plan, required_car_parking_space, room_type_reserved,
                  lead_time, arrival_year, arrival_month, arrival_date, market_segment_type, repeated_guest, no_of_previous_cancellations,
                  no_of_previous_bookings_not_cancelled, avg_price_per_room, no_of_special_requests, no_of_individuals, no_of_days_booked]

    scaled_data = scaler.transform(np.array([user_input]))  #scaling the input

    if st.button('Predict'):
      prediction = model.predict(scaled_data)
      output = prediction[0]
      if output == 1:
        st.success("The booking will not be canceled")
      else:
         st.success("The booking will be canceled")


# In[82]:


if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:




