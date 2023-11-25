import pyblp
import pandas as pd
import re
import numpy as np 
import statsmodels.formula.api as smf

# Load data
product_data = pd.read_csv(pyblp.data.NEVO_PRODUCTS_LOCATION)
product_data.head()
product_data.columns


product_data.loc[product_data.market_ids=='C01Q1',
        [i for i in product_data.columns if re.search('^((?!demand).)*$', i)]
        ]
# --> market_ids account for city and quarter
product_data.loc[product_data.market_ids=='C01Q1', ['city_ids', 'quarter', 'shares']]
product_data.loc[product_data.market_ids=='C01Q1', 'shares'].sum()

product_data.shape

# Set up the problem
logit_formulation = pyblp.Formulation('prices', absorb='C(product_ids)')
logit_formulation = pyblp.Formulation('prices + sugar + mushy')
logit_formulation = pyblp.Formulation('prices')
print(logit_formulation)

problem = pyblp.Problem(logit_formulation, product_data)
print(problem)

# Solving the problem
logit_results = problem.solve()
print(logit_results)



print(1.085386E+01)
print(+8.359402E-01)
print(-3.004710E+01)
print(+1.008589E+00)


# =============
# OLS
product_data.columns
small_df = product_data[
        ['market_ids', 'city_ids', 'quarter',
         'shares', 'prices', 'sugar', 'mushy']].copy()

# Derive "outside option's share"; recall that market_ids includes city and qtr
small_df.groupby(['market_ids'])['shares'].sum()
small_df['s0'] = 1 - small_df.groupby(['market_ids'])['shares'].transform('sum')


# Regression
reg = smf.ols('np.log(shares/s0) ~ prices + sugar + mushy', data = small_df)
res = reg.fit()
print(res.summary())
