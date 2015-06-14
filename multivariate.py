import pandas as pd
import statsmodels.api as sm
import numpy as np


# visit https://www.lendingclub.com/info/download-data.action
# select years 2013-2014 or loanStats3c.csv

# retrieve data
loanStats = pd.read_csv('\Users\zecharmo\Thinkful\loanStats3c.csv', low_memory=False)

# convert int_rate from a percent to a floating point number
loanStats['int_rate'] = loanStats['int_rate'].map(lambda x: round(float(x.rstrip('%')) / 100, 4))

# home_ownership is a categorical variable, can have the value 'OWN', 'RENT', or 'MORTGAGE'
# create dummy variables to separate home_ownership into three columns based on type
ownership_type = pd.get_dummies(loanStats['home_ownership'])

# merge new columns into original DataFrame
loanStats = pd.merge(loanStats, ownership_type, left_index=True, right_index=True)



# use annual income (annual_inc) to model interest rate (int_rate)
# linear model will have the equation int_rate = intercept + coefficient* (annual_inc)

# extract the columns that will be used in the linear model
rate = loanStats['int_rate']
income = loanStats['annual_inc']

# reshape the data
y = np.matrix(rate).transpose()
x = np.matrix(income).transpose()

# create the linear model
X = sm.add_constant(x)
model = sm.OLS(y,X)
f = model.fit()
f.summary()
# Out: const = 0.1246, x1 = -6.539e-08

# linear model --- int_rate = 0.1246 + -6.539e-08 * (annual_inc)
# R-squared value is 0.007 



# add home ownership type to the model of annual income (annual_inc) and interest rate (int_rate)
# new model will have the equation int_rate = intercept + coefficient * (annual_inc) + coefficient * (RENT) + coefficient * (MORTGAGE) + coefficient * (OWN)

# extract the columns that will be used in the linear model
rent = loanStats['RENT']
mortgage = loanStats['MORTGAGE']
own = loanStats['OWN']

# reshape the data
y = np.matrix(rate).transpose()
x1 = np.matrix(income).transpose()
x2 = np.matrix(rent).transpose()
x3 = np.matrix(mortgage).transpose()
x4 = np.matrix(own).transpose()

# put the columns together to create an input matrix
x = np.column_stack([x1,x2,x3,x4])

# create the linear model
X = sm.add_constant(x)
model = sm.OLS(y,X)
f = model.fit()
f.summary()
# Out: const = 0.1189, x1 = -6.175e-08, x2 = 0.0247, x3 = 0.0223, x4 = 0.0243

# new model --- int_rate = 0.1189 + -6.175e-08 * (annual_inc) + 0.0247 * (RENT) + 0.0223 * (MORTGAGE) + 0.0243 * (OWN)
# R-squared value is 0.008



# add the interaction of home ownership type and income
loanStats['income_RENT'] = loanStats.apply(lambda x: x['annual_inc'] * x['RENT'], axis = 1)
loanStats['income_MORTGAGE'] = loanStats.apply(lambda x: x['annual_inc'] * x['MORTGAGE'], axis = 1)
loanStats['income_OWN'] = loanStats.apply(lambda x: x['annual_inc'] * x['OWN'], axis = 1)

# extract the columns that will be used in the linear model
inc_rent = loanStats['income_RENT']
inc_mortgage = loanStats['income_MORTGAGE']
inc_own = loanStats['income_OWN']

# reshape the data
y = np.matrix(rate).transpose()
x1 = np.matrix(inc_rent).transpose()
x2 = np.matrix(inc_mortgage).transpose()
x3 = np.matrix(inc_own).transpose()

# put the columns together to create an input matrix
x = np.column_stack([x1,x2,x3])

# create the linear model
X = sm.add_constant(x)
model = sm.OLS(y,X)
f = model.fit()
f.summary()
# Out: const = 0.1424, x1 = -5.251e-08, x2 = -6.958e-08, x3 = -5.729e-08

# new model --- int_rate = 0.1424 + -5.251e-08 * (income_RENT) + -6.958e-08 * (income_MORTGAGE) + -5.729e-08 * (income_OWN)
# R-squared value is 0.007