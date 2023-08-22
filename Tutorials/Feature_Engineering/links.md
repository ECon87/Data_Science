# Links
1. [Comprehensive quide to feature selection by Kaggle](https://www.kaggle.com/code/prashant111/comprehensive-guide-on-feature-selection)
2. [On using the full set for feature engineering](https://datascience.stackexchange.com/questions/80770/i-do-feature-engineering-on-the-full-dataset-is-this-wrong)
    - Not necessarily wrong to use the whole dataset for feature engineering as long as only use 'row' information
3. [Medium article of feature selection and engineering](https://towardsdatascience.com/the-art-of-finding-the-best-features-for-machine-learning-a9074e2ca60d)
    - Feature Selection Algorithms:
        - Manual feature selection; e.g., correlation plots and feature importance after training a model
        - Automated feature selection
            - Variance threshold: drop if feature's variance is below a threshold
            - Univariate feature selection: ANOVA, `SelectKBest` method htat selects the K best features
            - Recursive feature elimination: perform model training on a gradually smaller and smaller set of features, features with the lowest scores are removed.
    - Feature Engineering:
        - Manual feature engineering
        - Automated feature engineering - see `featuretools`
4. [Machine learning mastery article on RCE and PCA]( https://machinelearningmastery.com/feature-selection-machine-learning-python/)
# Questions
1. Can we use cross-validation in feature engineering and feature selection?
