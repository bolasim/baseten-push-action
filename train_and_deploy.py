from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import datasets

import baseten

import sys

def make_model():
    iris = datasets.load_iris()

    x = iris.data
    y = iris.target

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.5)

    classifier = tree.DecisionTreeClassifier()

    classifier.fit(x_train, y_train)

    return classifier


def deploy_model(model, model_name, baseten_token):
    baseten.login(baseten_token)

    baseten.deploy(model, model_name)


if __name__ == '__main__':
    _, model_name, token = sys.argv

    deploy_model(make_model(), model_name, token)    

