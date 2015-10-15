#Let's test out what number of trees is best on a forest!
import numpy as np
import read_dataset as rd
import evaluation as e
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation

# loading training data
print('Loading training data')
X,y = rd.read_train()
X,y = rd.nudge_dataset(X,y)

scores = []
scores_std = []

#just so we know it didn't blow up or something
print('Start learning...')
#The last few might be excessive.
forests = [10, 15, 20, 25, 30, 40, 50, 70, 100, 125, 150, 175, 200, 250]

for tree in forests:
    print("This forest has {} trees!".format(tree))
    classifier = RandomForestClassifier(tree)
    #score = cross_validation.cross_val_score(classifier, X, y)
    #scores.append(np.mean(score))
    #scores_std.append(np.std(score))
    name = "plots_extended/RandomForest_{}_trees.png".format(tree)
    e.evaluate_classifier(classifier,X,y, name=name)

#print('Score: ', np.array(scores))
#print('Std  : ', np.array(scores_std))
