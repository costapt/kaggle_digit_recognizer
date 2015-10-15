import csv
import random
import numpy as np
from scipy.ndimage import convolve

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

def nudge_dataset(X, y):
    """
    This produces a dataset 5 times bigger than the original one,
    by moving the 8x8 images in X around by 1px to left, right, down, up
    """
    direction_vectors = [
        [[0, 1, 0],
         [0, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [1, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 1],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 0],
         [0, 1, 0]]]

    shift = lambda x, w: convolve(x.reshape((28, 28)), mode='constant',
                                  weights=w).ravel()
    X = np.concatenate([X] +
                       [np.apply_along_axis(shift, 1, X, vector)
                        for vector in direction_vectors])
    y = np.concatenate([y for _ in range(5)], axis=0)
    return X, y

if __name__ == '__main__':

    X,y = read_train(limit=-1)
    print(len(X))
    print(len(y))
    X,y = nudge_dataset(X,y)
    print(len(X),len(y))
