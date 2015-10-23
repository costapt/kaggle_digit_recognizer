import read_dataset as rd
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def classifyRF(train_file="train.csv", test_file ="test.csv", trees=70):
    #So, let's classifiy this thing. Reading the Features and then the test.
    print("Reading train data")
    X,y = rd.read_train(file_name=train_file)
    print("Augmenting dataset")
    X,y = rd.nudge_dataset(X,y)
    print("Reading test data")
    test_data = rd.read_test(file_name=test_file)

    #Creating the classifier. It has a ton of parameters, but since this a hard and fast one, here you go.
    rfc = RandomForestClassifier(trees)
    #Train with the data we have. Cry a little inside.
    print("Training classifier")
    rfc.fit(X, y)
    predictions = rfc.predict(test_data)
    print(len(predictions))

    #Most submitions are cute with a CSV. Might as well learn how to do it.
    pd.DataFrame({"ImageId": range(1,len(predictions)+1), "Label": predictions}).to_csv('submit.csv', index=False, header=True)

if __name__ == '__main__':
    classifyRF(trees=70)
