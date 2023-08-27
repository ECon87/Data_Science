# Description: Peform preliminary data exploration exercises
# Author: Evangelos Constantinou
#
# Variables to use in ML model -
# - Class_1, Class_2, Class_3
# - Embarked_NS
# - Female
# - ParCh_dm
# - Siblings
# - Log_Age
# - Log_Fare
# - Interactions:
#   - Class and Embarked_NS
#   - Class and Female (but lift is same direction for all groups - ignore)
#   - Class and Parents
#   - Class and Siblings
#   - Female and Parents
#   - Female and Siblings
#
#   TODO: Create family groups and number of family members saved?

from os import listdir
from pprint import pprint
import numpy as np
import pandas as pd
import re
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.impute import SimpleImputer
from statsmodels.formula.api import ols
from gen_features import gen_features


def summarize_category(df: pd.DataFrame, *args):
    """
    Get survival summary statistics by the args categorical variables.
    """
    print(round(df.groupby(by=list(args))["Survived"].describe(), 2))

def count_missing(df: pd.DataFrame, *args):
    """
    Return missing values for *args
    """
    return data[list(args)].isnull().sum()


# Directory
DIR_MASTER = "/home/econ87/Documents/Learning/Data_Science/Kaggle/Competitions/Titanic/"
DIR_DATA = f"{DIR_MASTER}/Data/"

# Load train data
data = pd.read_csv(f"{DIR_DATA}/train.csv", engine="pyarrow")

# Check missing values
count_missing(data, 'Age', 'Fare', 'Embarked', 'Pclass', 'Parch', 'SibSp')

# Features
data = gen_features(data)
data['Age_imputed'] = SimpleImputer(strategy = 'mean').fit_transform(data.Age.values.reshape(-1,1))

# Split into train and test sets

pprint(data.columns)
pprint(data.describe())
pprint(data.info())


# ==========================================================================
# Exploration of simple correlations.
# Variables examined -
# - Class
# - Embarked
# - Sex
# - Parch
# - SibSp
# - Name: Should I split into train and test based on family name; look into 
# - Cabin: Ignore too many missing values
# ==========================================================================

# Class (Pclass) is strongly correlated with the probability of surviving;
summarize_category(data, "Pclass")

# Embarked is correlated with Survived;
# Not strong correlation between Class and Embarked, but among those embarked
# at C most where 1st class, at Q most were 3rd class, and S had the most second
# class. However, most people were embarked at S.
summarize_category(data, "Embarked")

# Class `1st-Q` and `2nd-Q` don't have enough observations, while `3rd-Q` mean
# Survival rates for the `3rd-Q` class are the same as `3rd-C'class.
pprint(pd.crosstab(data.Pclass, data.Embarked))
pprint(round(pd.crosstab(data.Pclass, data.Embarked, normalize="columns"), 2))

# Relationship between Class and Embarked - Correlation between Survived and Class-Embarked class.
summarize_category(data, "Pclass", "Embarked")
summarize_category(data, "Pclass", "Embarked_NS")

# Sex is strongly correlated with Survival, but the difference between male and female
# does not appear to be affected by class
summarize_category(data, "Sex")
summarize_category(data, "Pclass", "Sex")


# Parents - number of parents
summarize_category(data, "Parch")
summarize_category(data, "ParCh_dm")
summarize_category(data, "Pclass", "ParCh_dm")
summarize_category(data, "Sex", "ParCh_dm")

# Number of siblings
summarize_category(data, "SibSp")
summarize_category(data, "Siblings")
summarize_category(data, "Pclass", "Siblings")
summarize_category(data, "Sex", "Siblings")


# Correlation between Fare and Pclass
pprint(round(data.groupby(by=["Pclass"])["Fare"].describe(), 2))


# Correlation between Fare and Age: low correlation
data[['Fare', 'Age_imputed']].corr()


# Use regressions to understand the correlation between Age, Fare and Survival
# The specification with just Log_Fare and Log_Age had the highest Adjusted_R2
ols("Survived ~ Log_Fare + Log_Fare2 + Log_Age + Log_Age2", data=data).fit(
    cov_type="HC0"
).summary()

ols("Survived ~ Log_Fare + Log_Age", data=data).fit(
    cov_type="HC0"
).summary()

ols("Survived ~ Fare_std + Age_std", data=data).fit(
    cov_type="HC0"
).summary()

ols("Survived ~ Fare + Age", data=data).fit(
    cov_type="HC0"
).summary()


# Cabin number has too many missing values so IGNORE
data["Cabin"].describe()


# Perhaps we should group by Name to create family groups
data.Name
data.Name.info()
