# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 13:56:52 2021

@author: TeenieTiny
"""

import pandas as pd
from scipy.stats import chi2_contingency
from scipy.stats import chi2
from scipy import stats as st

drown = pd.read_csv('DrowningData.csv')

#%%
##Initial tables and chisq test, cc month vs untampered drowning results
table = pd.crosstab(drown['CC_Month'], drown['Drowning_results'])

table_pct = pd.crosstab(drown['CC_Month'], drown['Drowning_results'], 
                        normalize='index')

c, p, dof, expected = chi2_contingency(table)
p

#%%
## adding margins to previous tables just to look
table2 = pd.crosstab(drown['CC_Month'], drown['Drowning_results'], margins=True)

table_pct2 = pd.crosstab(drown['CC_Month'], drown['Drowning_results'], 
                        normalize='index', margins=True)

##chisq doesn't work with margins, readable tables
print(table2)
print(table_pct2)

#%%

##Same as previous cells but combining missing and dead into one category
##missing is so small and it seems reasonable to assume missing people are,
##more often than not, dead with no body found
drown2 = drown.replace(to_replace=["Dead", "Missing"], value="Dead_or_Missing")

table3 = pd.crosstab(drown2['CC_Month'], drown2['Drowning_results'])
table4 = pd.crosstab(drown2['CC_Month'], drown2['Drowning_results'], margins=True)

table_pct3 = pd.crosstab(drown2['CC_Month'], drown2['Drowning_results'], 
                         margins=True, normalize='index')

table_pct4 = pd.crosstab(drown2['CC_Month'], drown2['Drowning_results'], 
                         margins=True, normalize='index')

#more readable tables
print(table4)
print(table_pct4)


c2, p2, dof2, expected2 = chi2_contingency(table3)
print(p2)

##combining does not drastically change our p-value so we'll stick with this result


#%%

##seeing if ghost month count is significantly different from 6th month

expected = (713+652)/2

chisq = ((713 - expected)**2 + (652 - expected)**2)/expected

pchi = chi2.sf(chisq, 1)

print(pchi)


###p-val about .0987 for chisq with 1df

#%%

##curious if the two most fatal months are statistically different from eachother

drown3 = drown2.replace(to_replace=["Dead_or_Missing", "Rescued"], value= [1, 0])

month2 = drown3.loc[drown2["CC_Month"] == 1, "Drowning_results"].to_numpy()
month9 = drown3.loc[drown2["CC_Month"] == 8, "Drowning_results"].to_numpy()

t = st.ttest_ind(a = month9, b = month2, equal_var=False)

print(t)

#%%