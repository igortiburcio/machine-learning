from sklearn.naive_bayes import BernoulliNB
import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

## historia boa (0), divida alta(0), garantia nenhuma (1), renda > 35 (2)
## historia ruim (2), divida alta (0), garantia adequada (0), renda < 15 (0)


def credit_risk_naive() -> None:
    x_credit_risk, y_credit_risk = pickle.load(open("data/credit_risk.pkl", "rb"))
    naive_credit_risk = GaussianNB()

    naive_credit_risk.fit(x_credit_risk, y_credit_risk)

    prediction = naive_credit_risk.predict([[0, 0, 1, 2], [2, 0, 0, 0]])

    print(prediction)


def credit_naive() -> None:
    X_credit_training, y_credit_training, X_credit_test, y_credit_test = pickle.load(
        open("data/credit.pkl", "rb")
    )

    naive_credit_data = GaussianNB()
    naive_credit_data.fit(X_credit_training, y_credit_training)

    prediction = naive_credit_data.predict(X_credit_test)

    print(accuracy_score(y_credit_test, prediction))
    print(classification_report(y_credit_test, prediction))


def census_naive() -> None:
    features_train, target_train, features_test, target_test = pickle.load(
        open("data/census.pkl", "rb")
    )

    naive_census = BernoulliNB()
    naive_census.fit(features_train, target_train)

    prediction = naive_census.predict(features_test)

    print(prediction)
    print(target_test)

    print(accuracy_score(target_test, prediction))
    print(classification_report(target_test, prediction))
