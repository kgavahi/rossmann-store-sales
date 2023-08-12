# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 10:49:24 2023

@author: kgavahi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('data/train.csv')
store_df = pd.read_csv('data/store.csv')

df = df.merge(store_df, on='Store', how='inner')

# Remove rows where 'open' column is zero
df = df.loc[df['Open'] != 0]

df['Date'] = pd.to_datetime(df['Date'])  # Convert the 'date' column to datetime format

# Add a new column 'day_of_year' which represents the day of the year
df['DayOfYear'] = df['Date'].dt.dayofyear

# Extract month and season
df['Month'] = df['Date'].dt.month



# Store number to filter
target_store_number = 2

# Filtering the DataFrame by store
target_store_df = df.loc[df['Store'] == target_store_number]

# Filtering the DataFrame by DayOfWeek
target_dow_df = df.loc[df['DayOfWeek'] == 7]


# Calculate the mean of 'sales' for rows where 'promo' column is 1
mean_sales_with_promo = target_store_df.loc[target_store_df['Promo'] == 1, 'Sales'].mean()
print('target store with promo: ', mean_sales_with_promo)

mean_sales_without_promo = target_store_df.loc[target_store_df['Promo'] == 0, 'Sales'].mean()
print('target store without promo: ', mean_sales_without_promo)


# Calculate the mean of 'sales' for rows where 'promo' column is 1
mean_sales_with_promo = df.loc[df['Promo2'] == 1, 'Sales'].mean()
print('all stores with promo: ', mean_sales_with_promo)

mean_sales_without_promo = df.loc[df['Promo2'] == 0, 'Sales'].mean()
print('all stores without promo: ', mean_sales_without_promo)



# Plot histogram
plt.hist(target_dow_df['Sales'], bins=100, edgecolor='black', alpha=0.7)


# Create a boxplot using Seaborn
plt.figure(figsize=(10, 6))
plt.title('Boxplots of Sales by DayOfWeek')
sns.boxplot(x='Promo2', y='Sales', data=df)
plt.show()


# Get unique values from the PromoInterval column
unique_values = df['PromoInterval'].unique()




