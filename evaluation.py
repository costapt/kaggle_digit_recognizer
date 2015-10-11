from sklearn import cross_validation
import matplotlib.pyplot as plt

def evaluate_classifier(classifier, X, y, name="plot.png"):

    cv_sizes = [0.9, 0.75, 0.5, 0.3]

    train_scores = []
    cv_scores = []

    for size in cv_sizes:
        print("Testing with cross validation size of {}.".format(size))
        X_train, X_cv, y_train, y_cv = cross_validation.train_test_split(X,y,test_size=size)

        print("Fitting")
        classifier.fit(X_train,y_train)

        print("Classifying test")
        train_score = classifier.score(X_train,y_train)
        print("Classifying cv")
        cv_score = classifier.score(X_cv,y_cv)

        print(train_score,cv_score)
        train_scores.append(train_score)
        cv_scores.append(cv_score)

    line1, = plt.plot(cv_sizes,train_scores,label="Train")
    line2, = plt.plot(cv_sizes,cv_scores,label="Cross Validation")
    plt.legend(handles=[line1,line2], loc=1)
    plt.ylabel("Accuracy")
    plt.xlabel("Cross Validation set size")
    plt.savefig(name)
    plt.close()
