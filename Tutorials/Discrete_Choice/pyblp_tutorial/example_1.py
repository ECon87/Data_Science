import pyblp
import pandas as pd
import re

# Load data
product_data = pd.read_csv(pyblp.data.NEVO_PRODUCTS_LOCATION)
product_data.head()
product_data.columns

product_data[
        [i for i in product_data.columns if re.search('^((?!demand).)*$', i)]
        ].head()


product_data.loc[product_data.market_ids=='C01Q1',
        [i for i in product_data.columns if re.search('^((?!demand).)*$', i)]
        ].head()

product_data.shape

# Set up the problem
logit_formulation = pyblp.Formulation('prices', absorb='C(product_ids)')
logit_formulation

problem = pyblp.Problem(logit_formulation, product_data)
problem

# Solving the problem
logit_results = problem.solve()
logit_results
