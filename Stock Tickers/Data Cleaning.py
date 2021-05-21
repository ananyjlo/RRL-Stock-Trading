# -*- coding: utf-8 -*-
"""
Created on Mon May 17 17:31:20 2021

@author: night
"""

# Import Libraries
## CHANGE THIS 

dir = "C:/Users/night/RRL-Stock-Trading/Stock Tickers/AAPL"
path = f'{dir}/*.csv'
name = 'AAPL'

import glob
files = glob.glob(path)
for f in files:
  print(f)

import pandas as pd

df = pd.DataFrame()
for yr in files:
  df = pd.concat([pd.read_csv(yr, parse_dates=['time'], index_col=1), df])
df.sort_index(inplace=True)
df.drop(['symbol'], axis=1, inplace=True)
df.head()
    
# Verify Data has loaded 
print(f"No. of data points: {len(df)}")
print(f"Start of raw data: {df.index[0]}")
print(f"End of raw data: {df.index[-1]}")



"""## Data Preprocessing"""
import matplotlib.pyplot as plt

start = '2017-03' # Start Period
end = '2021-03' # End Period

df = df.loc[start:end].copy()

print("After filtering:")
print(f"No. of data points: {len(df)}")
print(f"Start of raw data: {df.index[0]}")
print(f"End of raw data: {df.index[-1]}")

df['prev_high'] = df['high'].shift(1)
df['fod_high'] = df['high'] - df['high'].shift(1)
df['spread'] = (df['high'] - df['low'])
df['return'] = df['close'] / df['open'] - 1
df.drop(['close', 'low', 'open'], axis=1, inplace=True)
# Because of first-order difference, we have to discard the very first entry.
df = df.iloc[1:].copy()

def dfplot(input_df, size=(20,10)):
  # plot each column
  plt.figure(figsize=size)
  for cnt, feature in enumerate(input_df.columns):
    plt.subplot(len(input_df.columns), 1, cnt+1)
    plt.plot(input_df[feature])
    plt.title(feature, y=0.5, loc='right')
  plt.show()

df.head()

df.tail()

# plot each column
dfplot(df)

"""## Preprocessing Part 2"""

# reorder columns
col_names = ['fod_high', 'volume', 'spread', 'return']
df1 = df[col_names]

# feat_names = col_names[1:]
# tgt_name = col_names[0]

# Print the first few rows
df1.head()

# Print the last few rows
df1.tail()

# We will use df instead of df1 since we only need the "high" and "volume" data

ticker = df
ticker.head() # create a copy|

ticker.drop(['prev_high', 'fod_high', 'spread', 'return'], axis=1, inplace=True)

# Double check on data
ticker.head()

# Save Ticker File as CSV and TXT
from google.colab import files
ticker.to_csv(f"{name}.txt", sep='\t', index=False) # TXT
ticker.to_csv(f"{name}.csv", index=False) # CSV

files.download(f"{name}.csv")
files.download(f"{name}.txt")

# Print total number of samples to split later
print(f"No. of data points in {name} dataset: {len(ticker)}")

