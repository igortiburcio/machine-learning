import pandas as pd
import pickle as pkl
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split


#  0   age             32561 non-null  int64
#  1   workclass       32561 non-null  object
#  2   final-weight    32561 non-null  int64
#  3   education       32561 non-null  object
#  4   education-num   32561 non-null  int64
#  5   marital-status  32561 non-null  object
#  6   occupation      32561 non-null  object
#  7   relationship    32561 non-null  object
#  8   race            32561 non-null  object
#  9   sex             32561 non-null  object
#  10  capital-gain    32561 non-null  int64
#  11  capital-loos    32561 non-null  int64
#  12  hour-per-week   32561 non-null  int64
#  13  native-country  32561 non-null  object
#  14  income          32561 non-null  object


def process_database() -> None:
    base_census = pd.read_csv("data/census.csv")

    features = base_census.values[:, 0:14]
    target = base_census.values[:, 14]

    one_hot_encoder_census = ColumnTransformer(
        transformers=[
            ("OneHot", OneHotEncoder(sparse_output=False), [1, 3, 5, 6, 7, 8, 9, 13])
        ],
        remainder="passthrough",
    )
    features = one_hot_encoder_census.fit_transform(features)
    features = StandardScaler().fit_transform(features)

    features_train, target_train, features_test, target_test = train_test_split(
        features, target, test_size=0.15, random_state=0
    )

    with open("census.pkl", "wb") as f:
        pkl.dump([features_train, target_train, features_test, target_test], f)


def create_credit_risk_dataset() -> None:
    base_credit_risk = pd.read_csv("data/risco_credito.csv")
    x_credit_risk = base_credit_risk.iloc[:, 0:4].values
    y_credit_risk = base_credit_risk.iloc[:, 4].values

    label_encoder_history = LabelEncoder()
    label_encoder_debt = LabelEncoder()
    label_encoder_collateral = LabelEncoder()
    label_encoder_income = LabelEncoder()

    x_credit_risk[:, 0] = label_encoder_history.fit_transform(x_credit_risk[:, 0])
    x_credit_risk[:, 1] = label_encoder_debt.fit_transform(x_credit_risk[:, 1])
    x_credit_risk[:, 2] = label_encoder_collateral.fit_transform(x_credit_risk[:, 2])
    x_credit_risk[:, 3] = label_encoder_income.fit_transform(x_credit_risk[:, 3])

    with open("data/credit_risk.pkl", "wb") as f:
        pkl.dump([x_credit_risk, y_credit_risk], f)

    print(x_credit_risk)
