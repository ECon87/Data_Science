# Description: Create feeatures to feed into ML models
# AUthor: Evangelos Constantinou
# Date: Aug 20, 2023

import re
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# =========================================================================
# Feature engineering


def gen_features(data: pd.DataFrame):
    # Flag if not Embarked on S
    data["Embarked_NS"] = np.where(data["Embarked"] != "S", 1, 0)

    # Indicator dummies for each Class
    data = pd.concat(
        [data, pd.get_dummies(data["Pclass"], prefix="Class", dtype="float")], axis=1
    )

    # Female indicator
    data["Female"] = (data["Sex"] == "female").astype(int)

    # Any parents/children on board
    data["ParCh_dm"] = (data["Parch"] > 0).astype(int)
    data["Female_ParCh"] = data.Female * data.ParCh_dm

    # Any siblings on board
    data["Siblings"] = (data["SibSp"] > 0).astype(int)
    data['Female_Siblings'] = data.Female * data.Siblings

    # Interactions between Class and Embarked_NS
    for i in [j for j in data.columns if re.search("Class", j)]:
        data[f"{i}_Embarked_NS"] = data[i] * data["Embarked_NS"]
        data[f"{i}_Female"] = data[i] * data["Female"]
        data[f"{i}_ParCh"] = data[i] * data["ParCh_dm"]
        data[f"{i}_Siblings"] = data[i] * data["Siblings"]


    # Standarized Fare
    data["Fare_std"] = StandardScaler().fit_transform(
        data.Fare.values.reshape(-1, 1)
    )

    # Impute age
    data["Age_imputed"] = SimpleImputer(strategy="mean").fit_transform(
        data.Age.values.reshape(-1, 1)
    )

    data["Age_std"] = StandardScaler().fit_transform(
        data.Age_imputed.values.reshape(-1, 1)
    )

    # Log Age and Fare
    data["Log_Age"] = np.log(1 + data["Age_imputed"])
    data["Log_Fare"] = np.log(1 + data["Fare"])
    data["Log_Age2"] = np.log(1 + data.Age_imputed) ** 2
    data["Log_Fare2"] = data["Log_Fare"] ** 2

    return data
