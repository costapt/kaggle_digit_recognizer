import numpy as np
import read_dataset
from sklearn.ensemble import RandomForestClassifier

def classifyRF(estimators=50):
    #So, let's classifiy this thing. Reading the Features and then the test.
    features,label = read_train()
    test_data = read_test()

    #Creating the classifier. It has a ton of parameters, but since this a hard and fast one, here you go.
    rfc = RandomForestClassifier(n_estimators = estimators)
    #Train with the data we have. Cry a little inside.
    rfc.fit(features, label)
    #TODO some sort of validation would be great. Perhaps splitting the training data beforehand.
    #TODO Expand training data set?
    #How wrong can we be?
    prediction = rfc.predict(test_data)

    #Most submitions are cute with a CSV. Might as well learn how to do it.
    pd.DataFrame({"ImageId": range(1,len(prediction)+1), "Label": prediction}).to_csv('submit.csv', index=False, header=True)
