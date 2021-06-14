# -*- coding: utf-8 -*-
"""Machine Learning Case Study 02 - Job_Placement - Emerson Ham
This covers using various methods to predict the efficacy of employees at a firm given their resume info
Questions of machine learning ethics are addressed
"""

import pandas as pd
import matplotlib
import numpy as np
import sklearn as sk
from sklearn.model_selection import train_test_split
import datetime
from datetime import date
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
import tensorflow.keras as keras
from keras import metrics
from tensorflow.keras.layers import Dense, Dropout, Activation

# Load CSV excel sheet for processing
uploaded = files.upload()
df = pd.read_csv("employees.csv")
df.tail()

df.describe(include="all")
print("The columns of data are:")
list(df.columns)

# Split Myers Briggs personality types into parts
#df['MBTI_EI'] = df['MBTI_EI'][0]
df['MBTI_EI'] = [0] * len(df['Myers Briggs Type'])
df['MBTI_SN'] = df['MBTI_EI']
df['MBTI_TF'] = df['MBTI_EI']
df['MBTI_JP'] = df['MBTI_EI']

for i in range(len(df['MBTI_EI'])):
    df['MBTI_EI'][i] = df['Myers Briggs Type'][i][0]
    df['MBTI_SN'][i] = df['Myers Briggs Type'][i][1]
    df['MBTI_TF'][i] = df['Myers Briggs Type'][i][2]
    df['MBTI_JP'][i] = df['Myers Briggs Type'][i][3]

# List of categorical columns to convert into binary dummy variables
categorical_columns = [
   'Gender', #this one?
   'Race / Ethnicity', #this one?
   'English Fluency', #this one?
   'Spanish Fluency', #this one?
   'Education', #this one?
   'MBTI_EI',
   'MBTI_SN',
   'MBTI_TF',
   'MBTI_JP',
   'Requires Sponsorship',
   'Fired']
list(df.columns)

# Calculate the dummy variables and add them to the dataframe
for column in categorical_columns:
    df = pd.concat([df, pd.get_dummies(df[column])], axis=1)

print("The current columns are:")
list(df.columns)

# Drop old categorical data which has now been replaced
df.drop(categorical_columns, axis=1,inplace=True)
df.drop('Myers Briggs Type', axis=1,inplace=True)

print("The current columns are:")
list(df.columns)

def calculate_age(born):
    # Calculates age based on date of birth using https://stackoverflow.com/a/9754466/818687
    # Inputs:
    #   born (datetime): The date of birth
    # Returns:
    #   int: The age based on date of birth

    # We will use this date as "today" to make sure we get consistent results if an issue arises
    today = datetime.datetime.strptime("2019-10-20", "%Y-%m-%d") 
    # today = date.today() # This is what you'd normally do
    return (today.year - born.year - ((today.month, today.day) < (born.month, born.day)))

# Add the "Age" column to the dataframe
df['Age'] = [0] * len(df['Date of Birth'])
for i in range(len(df['Date of Birth'])):
    born = datetime.datetime.strptime(df['Date of Birth'][i], "%Y-%m-%d")
    df.loc[i,'Age'] = calculate_age(born)

"""## Modelling

Based on your understanding of the data, select the features that you want to use to predict:

1. Customer Satisfaction
1. Sales Performance
1. Fired
"""

# Save the columns we are trying to predict to targets
targets = ['Customer Satisfaction Rating','Sales Rating','Current Employee']

print("The available columns are:")
list(df)

# These are the features selected purely for performance reasons (ethics considered in later model)
rank_features = [ 'High School GPA',
 'College GPA',
 'Years of Experience',
 'Years of Volunteering',
 'Twitter followers',
 'Instagram Followers',
 'Female',
 'Male',
 'Black',
 'Caucasian',
 'Hispanic',
 'Basic',
 'Fluent',
 'Proficient',
 'Basic',
 'Fluent',
 'Proficient',
 'Associates',
 'Graduate',
 'High School',
 'None',
 'Undergraduate',
 'E',
 'I',
 'N',
 'S',
 'F',
 'T',
 'J',
 'P',
 False,
 True,
 'Age']

# I removed all specific data like address to prevent overfitting.
# Removing discriminating features comes later in this exercises, although I'd argue that that still applies at this part of the exercise

# Perform a train and test split on the data with the variable names:
df.dropna(inplace=True)
sc = MinMaxScaler()
rank_x_train, rank_x_test, rank_y_train, rank_y_test = train_test_split(sc.fit_transform(df[rank_features]), df[targets], test_size=0.2, random_state=0)

# Create a FFNN for ranking the employees
layers = [ Dense(len(rank_x_train[0]), activation='relu', input_shape=(len(rank_x_train[0]),)),
    Dense(10, activation='relu'),
    Dense(10, activation='relu'),
    Dense(3, activation='linear') ]
model = keras.Sequential(layers)
model.compile(optimizer='adam', loss='mae', metrics=['mae', "mse"])

# Fit the data
print("\n-----[ Fitting: ]-----\n")
model.fit(rank_x_train, rank_y_train, epochs=50)

# Evaluate the performance
print("\n-----[ Evaluating: ]-----\n")
model.evaluate(rank_x_test, rank_y_test)
results = model.predict(rank_x_test)
print("Compare the left side(original) with the right(predictions):")
print(np.concatenate([rank_y_test.values, results], axis=1))

# Removing discrimination based features
print("The available columns are:")
list(df)

# These are the features for the model which also include ethics concerns
selection_features = [ 'High School GPA',
 'College GPA',
 'Years of Experience',
 'Years of Volunteering',
 'Twitter followers',
 'Instagram Followers',
 'Basic',
 'Fluent',
 'Proficient',
 'Basic',
 'Fluent',
 'Proficient',
 'Associates',
 'Graduate',
 'High School',
 'None',
 'Undergraduate',
 'E',
 'I',
 'N',
 'S',
 'F',
 'T',
 'J',
 'P',
 False,
 True]

# Create the train and test data with the updated restrictions
selection_x_train, selection_x_test, selection_y_train, selection_y_test = train_test_split(sc.fit_transform(df[rank_features]), df[targets], test_size=0.2, random_state=0)

# Create a duplicate model to the original
layers = [ Dense(len(selection_x_train[0]), activation='relu', input_shape=(len(selection_x_train[0]),)),
    Dense(10, activation='relu'),
    Dense(10, activation='relu'),
    Dense(3, activation='linear') ]
model = keras.Sequential(layers)
model.compile(optimizer='adam', loss='mae', metrics=['mae', "mse"])

# Fit the updated data
print("\n-----[ Fitting: ]-----\n")
model.fit(selection_x_train, selection_y_train, epochs=50)

# Evaluate its effectiveness
print("\n-----[ Evaluating: ]-----\n")
model.evaluate(selection_x_test, selection_y_test)
results = model.predict(selection_x_test)
print("Compare the left side(original) with the right(predictions):")
print(np.concatenate([selection_y_test.values, results], axis=1))

# The unlimited model had a mean absolute error of 0.0677, while the limited legal model had a mean absolute error of 0.0439.
# This means the extra illegal data actually hurt the performance, likely because it led to extra data to overfit with.
# This goes to show that machine learning ethics does not have to get in the way of performance