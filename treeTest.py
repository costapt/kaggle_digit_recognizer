#Let's test out what number of trees is best on a forest!
import numpy as np
import read_dataset
from sklearn.ensemble import RandomForestClassifier

# loading training data
print('Loading training data')
features,label = read_train()

score = list()
scores_std = list()

#just so we know it didn't blow up or something
print('Start learning...')
#The last few might be excessive.
forests = [10, 15, 20, 25, 30, 40, 50, 70, 100, 125, 150, 175, 200, 250]

for tree in forest:
    print("This forest has {} trees!".format(tree))
    classifier = RandomForestClassifier(tree)
    score = cross_val_score(classifier, features, label)
    scores.append(np.mean(score))
    scores_std.append(np.std(score))

print('Score: ', np.array(scores))
print('Std  : ', np.array(scores_std))
