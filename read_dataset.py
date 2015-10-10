import csv
import random
import numpy as np

def split(X,y,n):
    if n > 0:
        rows_to_include = random.sample(range(len(X)), n)
        X = [X[row] for row in rows_to_include]
        y = [y[row] for row in rows_to_include]

    return X,y

def read_train(file_name='train.csv', limit=-1):
    X = []
    # Train data has a label
    y = []
    with open(file_name,'rt') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        # Skip header
        next(reader)
        # row = 'label pixel0 pixel1 pixel2 pixel3 ... pixel783'
        # label in [0...9]
        for row in reader:
            r = [int(elem) for elem in row]
            y.append(r[0])
            X.append(np.array(r[1:len(r)]))

    return split(X,y,limit)

def read_test(file_name='test.csv', limit=-1):
    X = []
    #But Test data has no label.
    with open(file_name,'rt') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        # Skip header
        next(reader)
        # row = 'label pixel0 pixel1 pixel2 pixel3 ... pixel783'
        # label in [0...9]
        for row in reader:
            r = [int(elem) for elem in row]
            X.append(r)

    y = [i for i in range(len(X))]
    X,_ = split(X,y,limit)
    return X

if __name__ == '__main__':

    X,y = read_train(limit=-1)
    print(len(X))
    print(len(y))

    X = read_test(limit=-1)
    print(len(X))
