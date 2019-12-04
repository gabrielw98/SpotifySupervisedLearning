# General packages
import pandas, numpy
from collections import Counter

# NN packages
from sklearn.neural_network import MLPRegressor

# KNR packages
from sklearn.neighbors import KNeighborsRegressor

# KRR packages
from sklearn.metrics.pairwise import polynomial_kernel
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics.pairwise import laplacian_kernel
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.model_selection import train_test_split
from sklearn.kernel_ridge import KernelRidge

#Classifiers

# NN classification packages
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler  # doctest: +SKIP



def load_data():
    print("Loading data...")
    df = pandas.read_csv("songs.csv")
    is_pop = df["genre"] == "Pop"
    df = df[is_pop]

    X = df.drop(columns=["danceability", "genre", "artist_name", "track_name", "key", "mode", "time_signature"])
    y = df["danceability"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

def load_data_classification():
    print("Loading data...")
    df = pandas.read_csv("songs.csv")

    X = df.drop(columns=["genre", "artist_name", "track_name", "key", "mode", "time_signature"])
    y = df["genre"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    print("Scaled the data")

    return X_train, X_test, y_train, y_test

class KernelRidgeRegressor():
    '''
    Kernel Ridge Regression Class
    '''

    def __init__(self):
        print("Initialized - Kernel Ridge Regressor")

    def predict(self, X_train, X_test, y_train, y_test):
        print("Predicting")
        poly_kernel = KernelRidge(kernel='rbf', alpha=1.0)

        print(y_train)
        print("before fit")
        #poly_kernel.fit(X_train, y_train_values)
        #print(poly_kernel.predict(X_test))
        print("finished predicting")

        for kernel in ["rbf", "poly", "laplacian"]:
            print(kernel)
            K = pairwise_kernels(X_train, X_train, metric=kernel)
            krr = KernelRidge(kernel=kernel)
            pred = krr.fit(X_train, y_train).predict(X_train)
            #pred2 = KernelRidge(kernel="precomputed").fit(K, y_train).predict(K)
            # print(assert_array_almost_equal(pred, pred2))
            print(pred)
            print(krr.score(X_test, y_test))
            print("\n")

class KNR():
    '''
    K Neighbors Regression Class
    '''

    def __init__(self):
        '''
        Initializes data members.
        '''

    def predict(self, k, prediction_count, X_train, X_test, y_train, y_test):
        knn = KNeighborsRegressor(n_neighbors=k, weights="distance", p=1)

        # Fit the classifier to the data
        knn.fit(X_train, y_train)
        print(knn.predict(X_test)[0:prediction_count])
        print(knn.score(X_test, y_test))



class NeuralNetwork():
    '''
    Neural Network Class
    '''

    def __init__(self):
        print("Initialized - Neural Network")

    def predict(self, X_train, X_test, y_train, y_test):
        print("\nNN  - Predicting danceability")
        mlp = MLPRegressor(hidden_layer_sizes=(200, 100, 50, 10), activation="relu", solver="adam",
                           max_iter=15000)
        mlp.fit(X_train, y_train)
        predict_train = mlp.predict(X_train)
        predict_test = mlp.predict(X_test)
        print(predict_train)
        print(predict_test)
        print(mlp.score(X_train, y_train))

    def choose_hyperparameters(self):
        activation_types = ["identity", "logistic", "tanh", "relu"]
        solver_types = ["adam", "sgd"]
        print("Comparing parameters")
        '''for activation in activation_types:
            for solver in solver_types:
                print("Activation: ", activation, "\nSolver: ", solver)
                mlp = MLPRegressor(hidden_layer_sizes=(100, 50, 25, 10), activation=activation, solver=solver,
                                   max_iter=5000)
                mlp.fit(X_train, y_train)
                print("\n")
                predict_train = mlp.predict(X_train)
                print(predict_train)
                predict_test = mlp.predict(X_test)
                print(predict_test)
                print(mlp.score(X_train, y_train))'''


def main():
    X_train, X_test, y_train, y_test = load_data()

    # KNR
    knr = KNR()
    k = 20
    prediction_count = 10
    knr.predict(k, prediction_count, X_train, X_test, y_train, y_test)

    '''
    # KRR
    krr = KernelRidgeRegressor()
    krr.predict(X_train, X_test, y_train, y_test)


    # NN
    nn = NeuralNetwork()
    nn.predict(X_train, X_test, y_train, y_test)'''

if __name__ == '__main__':
    main()

    '''
    X_train, X_test, y_train, y_test = load_data_classification()

    clf = MLPClassifier(solver='lbfgs', activation='relu',
                        hidden_layer_sizes=(30, 20, 20, 20), random_state=100)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)
    print(prediction)
    print(len(prediction))
    print(len(Counter(prediction).keys()))
    print(clf.score(X_test, y_test))'''
