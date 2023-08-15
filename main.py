# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 10:49:24 2023

@author: kgavahi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error



def RMSPE(y, yhat):
    
    n = len(y)
    
    return np.sqrt(1/n * np.sum(((y-yhat)/y)**2))

df = pd.read_csv('data/train.csv')
store_df = pd.read_csv('data/store.csv')
df_test = pd.read_csv('data/test.csv')
df_submit = pd.read_csv('data/sample_submission.csv')

#df = df.merge(store_df, on='Store', how='inner')

# Remove rows where 'open' column is zero
df = df.loc[df['Open'] != 0]

# Remove rows where 'Sales' column is zero
df = df.loc[df['Sales'] != 0]

df['Date'] = pd.to_datetime(df['Date'])  # Convert the 'date' column to datetime format
df_test['Date'] = pd.to_datetime(df_test['Date'])  # Convert the 'date' column to datetime format

# Add a new column 'day_of_year' which represents the day of the year
df['DayOfYear'] = df['Date'].dt.dayofyear
df_test['DayOfYear'] = df_test['Date'].dt.dayofyear

# Extract month and season
df['Month'] = df['Date'].dt.month
df_test['Month'] = df_test['Date'].dt.month

unique_values = df_test['Store'].unique()



for target_store_number in unique_values:
    # Store number to filter
    #target_store_number = 1
    
    # Filtering the DataFrame by store
    target_store_df = df.loc[df['Store'] == target_store_number]
    
    
    # Filtering the DataFrame by Sales=0
    no_sale = df.loc[df['Sales'] < 10]
    
    
    '''
    # Calculate the mean of 'sales' for rows where 'promo' column is 1
    mean_sales_with_promo = target_store_df.loc[target_store_df['Promo'] == 1, 'Sales'].mean()
    print('target store with promo: ', mean_sales_with_promo)
    
    mean_sales_without_promo = target_store_df.loc[target_store_df['Promo'] == 0, 'Sales'].mean()
    print('target store without promo: ', mean_sales_without_promo)
    
    
    cond = (df['Promo'] == 0) & (df['Promo2'] == 1)
    # Calculate the mean of 'sales' for rows where 'promo' column is 1
    mean_sales_with_promo = df.loc[cond, 'Sales'].mean()
    print('all stores with cond: ', mean_sales_with_promo)'''
    
    
    
    
    '''
    # Plot histogram
    plt.hist(target_dow_df['Sales'], bins=100, edgecolor='black', alpha=0.7)
    
    
    # Create a boxplot using Seaborn
    plt.figure(figsize=(10, 6))
    plt.title('Boxplots of Sales by DayOfWeek')
    sns.violinplot(x='DayOfWeek', y='Sales', data=target_store_df)
    plt.show()
    
    
    # Get unique values from the PromoInterval column
    unique_values = df['StateHoliday'].unique()'''
    
    
    
    train_data = target_store_df.drop(['Store', 'Date', 'StateHoliday', 'Open', 'Customers'], axis=1)
    
    
    features = train_data.drop('Sales', axis=1)
    target = train_data['Sales']
    
    features_mean = features.mean()
    features_std = features.std()
    
    target_mean = target.mean()
    target_std = target.std()
    
    features = (features - features_mean) / features_std
    target = (target - target_mean) / target_std
    
    
    
    
    
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    
    # Initialize the Random Forest Regressor model
    random_forest_model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    
    # Train the model on the training data
    random_forest_model.fit(X_train, y_train)
    
    # Make predictions on the test data
    predictions = random_forest_model.predict(X_test)
    
    predictions = (predictions * target_std) + target_mean
    y_test = (y_test * target_std) + target_mean
    
    print(target_store_number, RMSPE(y_test, predictions))
    
    #################################################################################
    
    # Filtering the DataFrame by store
    target_store_df_test = df_test.loc[df_test['Store'] == target_store_number]
    
    
    test_data = target_store_df_test.drop(['Store', 'Date', 'StateHoliday', 'Open', 'Id'], axis=1)
    test_data_sn = (test_data - features_mean) / features_std
    
    # Make predictions on the test data
    predictions_test = random_forest_model.predict(test_data_sn)
    
    predictions_test = (predictions_test * target_std) + target_mean
    
    target_store_df_test['predict'] = predictions_test
    
    target_store_df_test.loc[target_store_df_test['Open'] == 0, 'predict'] = 0
    
    df_submit.loc[target_store_df_test['Id']-1, 'Sales'] = target_store_df_test['predict'].values



df_submit.to_csv('sample_submission.csv', index=False)










