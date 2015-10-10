import csv
import random
import numpy as np

def read_dataset(file_name, limit):
    X = []
    y = []
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

def read_train(file_name='train.csv', limit=-1):
	return read_dataset(file_name,limit)

def read_test(file_name='test.csv', limit=-1):
	return read_dataset(file_name,limit)
