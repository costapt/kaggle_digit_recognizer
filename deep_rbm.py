import numpy as np
import pandas as pd
import read_dataset as rd
from sklearn.pipeline import Pipeline
from sklearn.neural_network import BernoulliRBM

class DeepRbmMnistClassifier:

    def __init__(self):
        self.n_components_first = 500
        self.n_components_second = 500
        self.n_components_third = 2000
        self.n_iter_first = 20
        self.n_iter_second = 20
        self.n_iter_third = 20
        self.learning_rate_first = 0.06
        self.learning_rate_second = 0.06
        self.learning_rate_third = 0.06
        self.verbose = True

    def label_to_feature(self,y):
        feature = [0]*10
        feature[y] = 1
        return feature

    def fit(self,X,y):
        self.rbm_1 = BernoulliRBM(verbose=self.verbose,
                            n_components=self.n_components_first,
                            n_iter=self.n_iter_first,
                            learning_rate=self.learning_rate_first)
        self.rbm_2 = BernoulliRBM(verbose=self.verbose,
                            n_components=self.n_components_second,
                            n_iter=self.n_iter_second,
                            learning_rate=self.learning_rate_second)
        self.first_pipeline = Pipeline(steps=[('rbm_1',self.rbm_1), ('rbm_2',self.rbm_2)])
        self.first_pipeline.fit(X,y)

        # TODO improve. Look at how it is done in classify
        new_features = []
        for example,label in zip(X,y):
            transformed = self.first_pipeline.transform(example)[0]
            new_features.append(np.concatenate((transformed,self.label_to_feature(label))))

        self.rbm_3 = BernoulliRBM(verbose=self.verbose,
                            n_components=self.n_components_third,
                            n_iter=self.n_iter_third,
                            learning_rate=self.learning_rate_third)
        self.rbm_3.fit(new_features,y)

    def classify(self,X):
        transformed = self.first_pipeline.transform(X)
        transformed = np.concatenate((transformed,[[0]*10]*len(transformed)),axis=1)

        # The inverse of rbm_3 to go from hidden layer to visible layer
        rbm_aux = BernoulliRBM()
        rbm_aux.intercept_hidden_ = self.rbm_3.intercept_visible_
        rbm_aux.intercept_visible_ = self.rbm_3.intercept_hidden_
        rbm_aux.components_ = np.transpose(self.rbm_3.components_)
        results = rbm_aux.transform(self.rbm_3.transform(transformed))
        results = results[:,-10:]
        return np.argmax(results,axis=1)


def submit():
    X,y = rd.get_train_data(augment=False)

    classifier = DeepRbmMnistClassifier()
    print('Fitting classifier')
    classifier.fit(X,y)

    testX = rd.get_test_data()
    print('Predicting')
    predictions = classifier.classify(testX)

    #Most submitions are cute with a CSV. Might as well learn how to do it.
    pd.DataFrame({"ImageId": range(1,len(predictions)+1), "Label": predictions}).to_csv('submit_deep_rbm.csv', index=False, header=True)

if __name__ == '__main__':
    submit()
