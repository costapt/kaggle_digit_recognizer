import numpy as np
import read_dataset as rd
import evaluation as e
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline

# loading training data
print('Loading training data')
X,y = rd.read_train(limit=500)
X,y = rd.nudge_dataset(X,y)
X = (X - np.min(X, 0)) / (np.max(X, 0) + 0.0001)  # 0-1 scaling

scores = []
scores_std = []

#just so we know it didn't blow up or something
print('Start learning...')
#The last few might be excessive.
forests = [70]
rbm_components = [1100]
rbm_learning_rate = [0.06]
rbm_n_iter = [20]

for tree in forests:
    for components in rbm_components:
        for learning_rate in rbm_learning_rate:
            for iterations in rbm_n_iter:
                rbm = BernoulliRBM(verbose=True,n_components=components,n_iter=iterations,learning_rate=learning_rate)
                random_forest = RandomForestClassifier(tree)
                classifier = Pipeline(steps=[('rbm',rbm), ('forest',random_forest)])
                name = "plots_pipeline/pipeline_{}.png".format(tree)
                e.evaluate_classifier(classifier,X,y, name=name)
