import itertools
import numpy as np
import pandas as pd
import evaluation as e
import read_dataset as rd
from sklearn.pipeline import Pipeline
from sklearn.neural_network import BernoulliRBM
from sklearn.ensemble import RandomForestClassifier

def evaluate_parameters():
    X,y = get_train_data(limit=25)

    scores = []
    scores_std = []

    print('Start learning...')
    forests = [70]
    rbm_components = [1100]
    rbm_learning_rate = [0.06]
    rbm_n_iter = [20]

    it = itertools.product(forests,rbm_components,rbm_learning_rate,rbm_n_iter)

    for (trees,components,learning_rate,n_iter) in it:
        classifier = get_classifier(trees,components,learning_rate,n_iter)
        name = "plots_pipeline/pipeline_{}.png".format(trees)
        e.evaluate_classifier(classifier,X,y, name=name)

def submission(trees=70,components=1100,learning_rate=0.06,n_iter=20):
    X,y,test_X = get_train_and_test_data()

    print("Defining classifiers")
    classifier = get_classifier(trees,components,learning_rate,n_iter)
    print("Training classifier")
    classifier.fit(X,y)
    predictions = classifier.predict(test_X)

    #Most submitions are cute with a CSV. Might as well learn how to do it.
    pd.DataFrame({"ImageId": range(1,len(predictions)+1), "Label": predictions}).to_csv('submit_rbm.csv', index=False, header=True)

def get_classifier(trees,components,learning_rate,n_iter):
    rbm = BernoulliRBM(verbose=True,n_components=components,
                        n_iter=n_iter,learning_rate=learning_rate)
    random_forest = RandomForestClassifier(trees)
    return Pipeline(steps=[('rbm',rbm), ('forest',random_forest)])

def scale(X):
    return (X - np.min(X, 0)) / (np.max(X, 0) + 0.0001)  # 0-1 scaling

def get_train_data(limit=-1):
    print('Loading train data')
    X,y = rd.read_train(limit=limit)
    print('Augmenting data set')
    X,y = rd.nudge_dataset(X,y)
    print('Scaling data')
    X = scale(X)
    return X,y

def get_train_and_test_data(train_limit=-1,test_limit=-1):
    X,y = get_train_data(train_limit)
    print('Loading test data')
    test_X = rd.read_test(limit=test_limit)
    test_X = scale(test_X)
    return X,y,test_X

#evaluate_parameters()
submission()
