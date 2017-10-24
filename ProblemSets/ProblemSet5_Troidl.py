import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Import dataset
gdp = pd.read_excel('/Users/alexandradinu/Desktop/CompEcon_Fall17/ProblemSets/GDPcapita.xls')
gdp_new = gdp[(gdp['Country Name']=="Romania")|(gdp['Country Name']== "Croatia")]
gdp_new.drop(gdp_new.columns[[ 1, 2, 3]], axis=1)

plt.style.use('ggplot') # select a style (theme) for plot
fig, ax = plt.subplots() # make figure and axes separate objects
x = [1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016]
y1 = np.array(gdp_new.iloc[0, 39:61])
y2 = np.array(gdp_new.iloc[1, 39:61])
plt.plot(x, y1, axes=ax, label="Croatia")
plt.plot(x, y2, axes=ax, label="Romania")
ax.set_xlim([1995, 2016]) # set axis range
ax.set(title='GDP per Capita', xlabel='Years',
       ylabel="GDP") # plot title, axis labels
ax.axvline(x=2007, color='k', linestyle='--') #insert vertical line at year 2007
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.) #include a legend

# save figure
fig.savefig('gdppcap.png', transparent=False, dpi=80, bbox_inches="tight")

#Import new dataset
unemp = pd.read_excel('/Users/alexandradinu/Desktop/CompEcon_Fall17/ProblemSets/UnempY.xls', skiprows=3)
unemp_1= unemp[(unemp['Country Name']=="Romania")|(unemp['Country Name']== "Croatia")]

plt.style.use('ggplot') # select a style (theme) for plot
fig, ax = plt.subplots() # make figure and axes separate objects
x = [1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016]
y1 = np.array(unemp_1.iloc[0, 39:61])
y2 = np.array(unemp_1.iloc[1, 39:61])
plt.plot(x, y1, axes=ax, label="Croatia")
plt.plot(x, y2, axes=ax, label="Romania")
ax.set_xlim([1995, 2016]) # set axis range
ax.set(title='Youth Unemployment Rate %', xlabel='Years',
       ylabel="Unemployment Rate") # plot title, axis labels
ax.axvline(x=2007, color='k', linestyle='--') #insert vertical line at year 2007
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.) #include a legend

#save figure
fig.savefig('unemp.png', transparent=False, dpi=80, bbox_inches="tight")

#Import final dataset
bd = pd.read_excel('/Users/alexandradinu/Desktop/CompEcon_Fall17/ProblemSets/bd.xlsx')
bd2 = bd[['Country','1995_High', '2000_High', '2005_High', '2010_High']]


plt.style.use('ggplot') # select a style (theme) for plot
fig, ax = plt.subplots() # make figure and axes separate objects
x = [1995, 2000, 2005, 2010]
y1 = np.array(bd2.iloc[0, 1:5])
y2 = np.array(bd2.iloc[30, 1:5])
plt.plot(x, y1, axes=ax, label="Croatia")
plt.plot(x, y2, axes=ax, label="Romania")
ax.set_xlim([1995, 2010]) # set axis range
ax.set(title='Highly Educated Migration', xlabel='Years',
       ylabel="Migraton Rate") # plot title, axis labels
ax.axvline(x=2007, color='k', linestyle='--') #insert vertical line at year 2007
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.) #include a legend
# save figure
fig.savefig('migrate.png', transparent=False, dpi=80, bbox_inches="tight")
