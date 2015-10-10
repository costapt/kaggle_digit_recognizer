import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def classifyRF(train_file="train.csv", test_file ="test.csv",estimators=50):
    # Read competition data files:
    train = pd.read_csv(train_file)
    test  = pd.read_csv(test_file)

    # Extract the relevant values. The first column is the prediction/label, so we keep it separate.
    # Test data has no such label, so we can just read it.
    x_train = train.values[:,1:]
    y_train = train.ix[:,0]
    test = test.values

    rfc = RandomForestClassifier(n_estimators= estimators)
    rfc.fit(x_train, y_train)
    prediction = rfc.predict(test)

    pd.DataFrame({"ImageId": range(1,len(prediction)+1), "Label": prediction}).to_csv('submit.csv', index=False, header=True)
