# Load libraries
import sys
import pandas
import argparse
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


def startupLoad():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', help="Load dataset (currently matrix dataset) from web site", default="https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", type=str, dest="webSite")
    parser.add_argument('-lr', help="Run logistic regression", action="store_true")
    parser.add_argument('-lda', help='Run Linear Discriminant Analysis', action='store_true')
    parser.add_argument('-knn', help='Run K Neighbors Classifier', action="store_true")
    parser.add_argument('-cart', help="Run Decision Tree Classifier", action="store_true")
    parser.add_argument('-nb', help="Run Gaussian NB", action="store_true")
    parser.add_argument('-svc', help="Run SVC", action='store_true')
    args = parser.parse_args()
    algList = []
    if args.lr:
        algList += ['LR']
    if args.lda:
        algList += ['LDA']
    if args.knn:
        algList += ['KNN']
    if args.cart:
        algList += ['CART']
    if args.nb:
        algList += ['NB']
    if args.svc:
        algList += ['SVC']
    if len(algList) == 0:
        print "At least one algorithm is required"
        sys.exit()
    return args.webSite, algList
    
if __name__ == "__main__":
    
    
    
    cmdArgs = startupLoad()
    print (cmdArgs)
    url = cmdArgs[0]
    # Load dataset
    #url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    dataset = pandas.read_csv(url, names=names)
    
    
    # shape
    print "Dataset shape"
    print(dataset.shape)
    
    # head
    print "First 20 colums of data"
    print(dataset.head(20))
    
    # descriptions
    print "Statistical measures of data"
    print(dataset.describe())
    
    # class distribution
    print "Number of instances of each class"
    print(dataset.groupby('class').size())
    
    # box and whisker plots
    dataset.plot(kind='box', title="Univariate plots", subplots=True, layout=(2,2), sharex=False, sharey=False)
    plt.show()
    
    # histograms
    dataset.hist()
    plt.suptitle("Histogram")
    plt.show()
    
    # scatter plot matrix
    scatter_matrix(dataset)
    plt.suptitle("Multivariate Plots")
    plt.show()
    
    # Split-out validation dataset
    array = dataset.values
    X = array[:,0:4]
    Y = array[:,4]
    validation_size = 0.20
    seed = 7
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
    
    '''
    Test Harness
    Test options and evaluation metric
    We are using the metric of accuracy to evaluate models. This is a ratio of the number of correctly predicted instances in divided by 
    the total number of instances in the dataset multiplied by 100 to give a percentage (e.g. 95 accurate). We will be using the scoring 
    variable when we run build and evaluate each model next.
    '''
    seed = 7
    scoring = 'accuracy'
    '''
    Build Models
    We dont know which algorithms would be good on this problem or what configurations to use. 
    We get an idea from the plots that some of the classes are partially linearly separable in
    some dimensions, so we are expecting generally good results.
    
    Lets evaluate 6 different algorithms:
    
        Logistic Regression (LR)
        Linear Discriminant Analysis (LDA)
        K-Nearest Neighbors (KNN).
        Classification and Regression Trees (CART).
        Gaussian Naive Bayes (NB).
        Support Vector Machines (SVM).
    
    This is a good mixture of simple linear (LR and LDA), nonlinear (KNN, CART, NB and SVM) algorithms. 
    We reset the random number seed before each run to ensure that the evaluation of each algorithm is performed 
    using exactly the same data splits. It ensures the results are directly comparable.
    Spot Check Algorithms
    '''
    models = []
    if 'LR' in cmdArgs[1]:
        models.append(('LR', LogisticRegression()))
    if 'LDA' in cmdArgs[1]:
        models.append(('LDA', LinearDiscriminantAnalysis()))
    if 'KNN' in cmdArgs[1]:
        models.append(('KNN', KNeighborsClassifier()))
    if 'CART' in cmdArgs[1]:
        models.append(('CART', DecisionTreeClassifier()))
    if 'NB' in cmdArgs[1]:
        models.append(('NB', GaussianNB()))
    if 'SVN' in cmdArgs[1]:
        models.append(('SVM', SVC()))
            
    # evaluate each model in turn
    results = []
    names = []
    for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
      
    # Compare Algorithms
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()
    
    for name, model in models:
        alg = model
        alg.fit(X_train, Y_train)
        predictions = alg.predict(X_validation)
        print("Name: %s" % name)
        print("Accuracy report: %.2f" % accuracy_score(Y_validation, predictions))
        print("Confusion matrix: ")
        print(confusion_matrix(Y_validation, predictions))
        print("Classification report: ")
        print(classification_report(Y_validation, predictions))
        
    # Make predictions on validation dataset
    knn = KNeighborsClassifier()
    knn.fit(X_train, Y_train)
    predictions = knn.predict(X_validation)
    print("---KNNeighbors is the best fit for iris data---")
    print("Accuracy report: %.2f" % accuracy_score(Y_validation, predictions))
    print("Confusion matrix: ")
    print(confusion_matrix(Y_validation, predictions))
    print("Classification report: ")
    print(classification_report(Y_validation, predictions))