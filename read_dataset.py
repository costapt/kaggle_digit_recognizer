import csv
import random
import numpy as np

#TODO limit can't be -1, or else we have "sample size larger than population errors"... at least on my machine.
def read_train(file_name='train.csv', limit=-1):
    X = []
    # Train data has a label
    y = []
    #TODO I don't understand this line, but limit can't be -1
    rows_to_include = random.sample(range(784), limit)
    i = 0
    with open(file_name,'rt') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        # Skip header
        next(reader)
        # row = 'label pixel0 pixel1 pixel2 pixel3 ... pixel783'
        # label in [0...9]
        for row in reader:
            if i in rows_to_include:
                r = [int(elem) for elem in row]
                y.append(r[0])
                X.append(np.array(r[1:len(r)]))

            i = i + 1
    return X,y


    #TODO limit can't be -1, or else we have "sample size larger than population errors"... at least on my machine.
def read_test(file_name='test.csv', limit=-1):
    X = []
    #But Test data has no label.
    #TODO I don't understand this line, but limit can't be -1
    rows_to_include = random.sample(range(784), limit)
    i = 0
    with open(file_name,'rt') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        # Skip header
        next(reader)
        # row = 'label pixel0 pixel1 pixel2 pixel3 ... pixel783'
        # label in [0...9]
        for row in reader:
            if i in rows_to_include:
                r = [int(elem) for elem in row]
                X.append(r)

            i = i + 1
    return X
