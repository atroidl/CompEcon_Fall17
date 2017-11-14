# import pandas-datareader and other packages
import pandas_datareader.data as web
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from pandas_datareader import wb

# 1980 Analysis
ind = "NY.GDP.PCAP.CD"
c = ["CHN", "IND", "USA", "IDN", "BRA", "NGA", "WLD"]
year = 1980
newdat = wb.download(indicator=ind, country=c, start= year, end=year)
newdat.columns= [ "GDP"]
df = newdat.reset_index(level='year', drop=True)
df = df.sort_values(by='GDP', ascending=True)

#Create Bar Chart
fig, ax = plt.subplots()
df['GDP'].plot(ax=ax, kind='barh', alpha=0.5)
ax.set_title('World Distribution of GDP in 1980', loc='left', fontsize=14)
ax.set_xlabel('Thousands of US Dollars')
ax.set_ylabel('')
fig.savefig('gdp1980.png', transparent=False, dpi=80, bbox_inches="tight")

#1990 Analysis
ind = "NY.GDP.PCAP.CD"
c = ["CHN", "IND", "USA", "IDN", "BRA", "NGA", "WLD"]
year = 1990
newdat = wb.download(indicator=ind, country=c, start= year, end=year)
newdat.columns= [ "GDP"]
df = newdat.reset_index(level='year', drop=True)
df = df.sort_values(by='GDP', ascending=True)

#Create Bar Chart
fig, ax = plt.subplots()
df['GDP'].plot(ax=ax, kind='barh', alpha=0.5)
ax.set_title('World Distribution of GDP in 1990', loc='left', fontsize=14)
ax.set_xlabel('Thousands of US Dollars')
ax.set_ylabel('')
fig.savefig('gdp1990.png', transparent=False, dpi=80, bbox_inches="tight")

# 2000 Analysis
ind = "NY.GDP.PCAP.CD"
c = ["CHN", "IND", "USA", "IDN", "BRA", "NGA", "WLD"]
year = 2000
newdat = wb.download(indicator=ind, country=c, start= year, end=year)
newdat.columns= [ "GDP"]
df = newdat.reset_index(level='year', drop=True)
df = df.sort_values(by='GDP', ascending=True)

#Create Bar Chart
fig, ax = plt.subplots()
df['GDP'].plot(ax=ax, kind='barh', alpha=0.5)
ax.set_title('World Distribution of GDP in 2000', loc='left', fontsize=14)
ax.set_xlabel('Thousands of US Dollars')
ax.set_ylabel('')
fig.savefig('gdp2000.png', transparent=False, dpi=80, bbox_inches="tight")

# 2010 Analysis
ind = "NY.GDP.PCAP.CD"
c = ["CHN", "IND", "USA", "IDN", "BRA", "NGA", "WLD"]
year = 2010
newdat = wb.download(indicator=ind, country=c, start= year, end=year)
newdat.columns= [ "GDP"]
df = newdat.reset_index(level='year', drop=True)
df = df.sort_values(by='GDP', ascending=True)

#Create Bar Chart
fig, ax = plt.subplots()
df['GDP'].plot(ax=ax, kind='barh', alpha=0.5)
ax.set_title('World Distribution of GDP in 2010', loc='left', fontsize=14)
ax.set_xlabel('Thousands of US Dollars')
ax.set_ylabel('')
fig.savefig('gdp2010.png', transparent=False, dpi=80, bbox_inches="tight")

# 2016 Analysis
ind = "NY.GDP.PCAP.CD"
c = ["CHN", "IND", "USA", "IDN", "BRA", "NGA", "WLD"]
year = 2016
newdat = wb.download(indicator=ind, country=c, start= year, end=year)
newdat.columns= [ "GDP"]
df = newdat.reset_index(level='year', drop=True)
df = df.sort_values(by='GDP', ascending=True)

#Create Bar Chart
fig, ax = plt.subplots()
df['GDP'].plot(ax=ax, kind='barh', alpha=0.5)
ax.set_title('World Distribution of GDP in 2016', loc='left', fontsize=14)
ax.set_xlabel('Thousands of US Dollars')
ax.set_ylabel('')
fig.savefig('gdp2016.png', transparent=False, dpi=80, bbox_inches="tight")
