# General packages
import pandas, numpy
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import time


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
from sklearn.model_selection import GridSearchCV

#Classifiers

# NN classification packages
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler  # doctest: +SKIP

# TODO
# Create the features for the genre and the keysgit s


def is_trivial(X_train, y_train):
    reg = LinearRegression().fit(X_train, y_train)
    score = reg.score(X_train, y_train)
    print("Linear Seperability Score:", score)
    return score > 0.5

def load_data():
    print("Loading data...")
    df = pandas.read_csv("songs.csv")
    genres = ["Pop", "Dance", "Hip-Hop", "Country"]

    is_popular = df["popularity"] > 60
    df = df[is_popular]
    df = df[df["genre"].isin(genres)]

    X = df.drop(columns=["danceability", "genre", "artist_name", "track_name", "key", "mode", "time_signature"])
    y = df["danceability"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    #determine_important_features(X_train, X_test, y_train, y_test)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

def load_data_classification():
    print("Loading data...")
    df = pandas.read_csv("songs.csv")

    X = df.drop(columns=["genre", "artist_name", "track_name", "key", "mode", "time_signature", "duration_ms"])
    y = df["genre"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    print(X_train.keys())

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    print("Scaled the data")

    return X_train, X_test, y_train, y_test

def determine_important_features(X_train, X_test, y_train, y_test):
    # Determine the important features
    print(X_train.keys())
    for feature in X_train.keys():
        x_train_d = X_train[[feature]]
        y_train_d = y_train
        x_test_d = X_test[[feature]]

        krr = KernelRidge(kernel="laplacian", alpha=0.7, gamma=0.04)
        krr.fit(x_train_d, y_train_d)
        krr.predict(x_test_d)
        score = krr.score(x_train_d, y_train)

        '''knn = KNeighborsRegressor(n_neighbors=20, weights="distance", p=1)

        # Fit the classifier to the data
        knn.fit(x_train_d, y_train_d)'''
        score = krr.score(x_test_d, y_test)

        print("feature:", feature, "score:", score)
        plt.title(feature + " vs. danceability")
        plt.ylabel("danceability")
        plt.xlabel(feature)
        prediction = krr.predict(x_test_d)
        plt.plot(x_test_d, y_test, "ro")
        plt.plot(x_test_d, prediction, "bo")
        plt.show()




class KernelRidgeRegressor():
    '''
    Kernel Ridge Regression Class
    '''

    def __init__(self):
        print("Initialized - Kernel Ridge Regressor")

    def tune_hyper_params(self, X_train, X_test, y_train, y_test):
        krr = KernelRidge(kernel="laplacian")
        params = {
            "alpha": numpy.arange(0, 2.5, 0.1),
            "gamma": numpy.arange(0.01, 0.2, 0.01)
        }
        krr = GridSearchCV(krr, params, n_jobs=-1)
        krr.fit(X_train, y_train)
        print(krr.best_params_)
        return krr.best_params_

    def graph_kernel(self, X_train, X_test, y_train, y_test):
        kernels = ("rbf", "poly", "laplacian")
        y_pos = numpy.arange(len(kernels))
        performance = []
        max_value = 0.0
        max_kernel = ""

        for kernel in kernels:
            krr = KernelRidge(kernel=kernel)
            krr.fit(X_train, y_train)
            krr.predict(X_test)
            score = krr.score(X_train, y_train)
            #print(score)
            if score > max_value:
                max_value = score
                max_kernel = kernel
            performance.append(score)

        plt.bar(y_pos, performance, align='center', alpha=0.5)
        plt.xticks(y_pos, kernels)
        plt.ylabel('Score')
        plt.xlabel('Kernel')
        plt.title('Tuning Kernel Hyperparameter')
        plt.show()
        return max_kernel

    def graph_alpha(self, X_train, X_test, y_train, y_test):
        alphas = numpy.arange(0, 2.5, 0.1)
        performance = []
        max_value = 0.0
        max_alpha = -1

        for alpha in alphas:
            krr = KernelRidge(kernel="laplacian", alpha=alpha)
            krr.fit(X_train, y_train)
            krr.predict(X_test)
            score = krr.score(X_test, y_test)
            #print(score)
            if score > max_value:
                max_value = score
                max_alpha = alpha
            performance.append(score)

        plt.plot(alphas, performance, "ro")
        plt.ylabel('Score')
        plt.xlabel('Alpha')
        plt.title('Tuning Alpha Hyperparameter')
        plt.show()
        return max_alpha

    def graph_gamma(self, X_train, X_test, y_train, y_test):
        gamma_list = numpy.arange(0.01, 0.2, 0.01)
        y_pos = numpy.arange(len(gamma_list))
        performance = []
        max_value = 0.0
        max_gamma = -1

        for gamma in gamma_list:
            krr = KernelRidge(kernel="laplacian", gamma=gamma)
            krr.fit(X_train, y_train)
            krr.predict(X_test)
            score = krr.score(X_test, y_test)
            if score > max_value:
                max_value = score
                max_gamma = gamma
            performance.append(score)

        plt.plot(gamma_list, performance, "bo")
        plt.ylabel('Score')
        plt.xlabel('Gamma')
        plt.title('Tuning Gamma Hyperparameter')
        plt.show()
        return max_gamma

    # Hyper parameters: kernel, gamma, alpha
    def predict(self, params, X_train, X_test, y_train, y_test):
        krr = KernelRidge(kernel="laplacian", gamma=params["gamma"], alpha=params["alpha"])
        krr.fit(X_train, y_train)
        print(krr.score(X_test, y_test))
        #cross_val_scores = cross_val_score(krr, X_train, y_train, cv=20)
        #print(cross_val_scores)

class KNR():
    '''
    K Neighbors Regression Class
    '''

    def __init__(self):
        '''
        Initializes data members.
        '''

    def tune_hyper_params(self, X_train, X_test, y_train, y_test):
        knn = KNeighborsRegressor()
        params = {
            "n_neighbors": range(1, 30, 1),
            "weights": ["distance", "uniform"],
            "p": [1, 2]
        }
        knn = GridSearchCV(knn, params, n_jobs=-1)
        knn.fit(X_train, y_train)
        return knn.best_params_

    def graph_weights(self, X_train, X_test, y_train, y_test):
        weights_list = ("distance", "uniform")

        y_pos = numpy.arange(len(weights_list))
        performance = []
        max_value = 0.0
        best_weight = ""

        for weight in weights_list:
            knn = KNeighborsRegressor(n_neighbors=24, weights=weight, p=1)
            knn.fit(X_train, y_train)
            knn.predict(X_test)
            score = knn.score(X_test, y_test)
            #print(score)
            performance.append(score)
        plt.bar(y_pos, performance, align='center', alpha=0.5)
        plt.xticks(y_pos, weights_list)
        plt.ylabel('Score')
        plt.xlabel('Weight Type')
        plt.title('Weight Hyperparameter')
        plt.show()

    def graph_neighbors(self, X_train, X_test, y_train, y_test):
        k_values = []
        scores_array = []
        for i in range(5, 50):
            k_values.append(i)
            knn = KNeighborsRegressor(n_neighbors=i, weights="distance", p=1)
            # Fit the classifier to the data
            knn.fit(X_train, y_train)
            #print(knn.predict(X_test)[0:prediction_count])
            scores = cross_val_score(knn, X_test, y_test, cv=10)
            scores_array.append(knn.score(X_test, y_test))
            #score = knn.score(X_test, y_test)
            #scores_array.append(scores.mean())
        plt.title("Performance vs # Neighbors")
        plt.ylabel("Scores")
        plt.xlabel("# Neighbors")
        plt.plot(k_values, scores_array, "bo")
        plt.show()

    def predict(self, params, X_train, X_test, y_train, y_test):
        knn = KNeighborsRegressor(n_neighbors=params["n_neighbors"], weights=params["weights"], p=params["p"])
        knn.fit(X_train, y_train)
        scores = cross_val_score(knn, X_test, y_test, cv=10)
        print(scores.mean())
        print(numpy.std(scores))

        print("Scores:", knn.score(X_test, y_test))



class NeuralNetwork():
    '''
    Neural Network Class
    '''

    def __init__(self):
        print("Initialized - Neural Network")

    def predict(self, params,  X_train, X_test, y_train, y_test):
        print("\nNN - Predicting danceability")
        mlp = MLPRegressor(hidden_layer_sizes=params["hidden_layer_sizes"], activation=params["activation"],
                           solver=params["solver"], alpha=params["alpha"])
        mlp.fit(X_train, y_train)

        print("Score:", mlp.score(X_test, y_test))

    def tune_hyper_params(self, X_train, X_test, y_train, y_test):
        mlp = MLPRegressor()
        params = {
            "activation": ["tanh", "identity", "logistic", "relu"],
            "solver": ["lbfgs", "sgd", "adam"],
            "hidden_layer_sizes" : [[10,10], [20,20], [30,30],
                                    [10,10,10], [20,20,20], [30,30,30]],
            "alpha" : [0.0001, 1e-5, 0.01, 0.001]
        }
        mlp = GridSearchCV(mlp, params, n_jobs=-1)
        mlp.fit(X_train, y_train)
        print(mlp.best_params_)
        return mlp.best_params_

    def graph_hidden_layers(self, X_train, X_test, y_train, y_test):
        layers = numpy.arange(1, 100, 1)
        performance = []
        max_value = 0.0
        max_layer = -1

        for layer in layers:
            nn = MLPRegressor(activation="relu", solver="lbfgs", hidden_layer_sizes=(layer, ))
            nn.fit(X_train, y_train)
            nn.predict(X_test)
            score = nn.score(X_test, y_test)
            #print(score)
            if score > max_value:
                max_value = score
                max_layer = layer
            performance.append(score)

        plt.plot(layers, performance, "ro")
        plt.ylabel('Score')
        plt.xlabel('Layer')
        plt.title('Tuning Layer Hyperparameter')
        plt.show()
        return max_layer

    def graph_activation(self, X_train, X_test, y_train, y_test):
        activation_types = ("identity", "logistic", "tanh", "relu")
        y_pos = numpy.arange(len(activation_types))
        performance = []
        max_value = 0.0
        max_kernel = ""

        for activation in activation_types:
            nn = MLPRegressor(activation=activation, solver="lbfgs", hidden_layer_sizes=(10,))
            nn.fit(X_train, y_train)
            nn.predict(X_test)
            score = nn.score(X_train, y_train)
            #print(score)
            if score > max_value:
                max_value = score
                max_kernel = activation
            performance.append(score)

        plt.bar(y_pos, performance, align='center', alpha=0.5)
        plt.xticks(y_pos, activation_types)
        plt.ylabel('Score')
        plt.xlabel('Kernel')
        plt.title('Tuning Activation Hyperparameter')
        plt.show()
        return max_kernel

    def graph_solver(self, X_train, X_test, y_train, y_test):
        solver_types = ("lbfgs", "sgd", "adam")
        y_pos = numpy.arange(len(solver_types))
        performance = []
        max_value = 0.0
        max_kernel = ""

        for solver in solver_types:
            nn = MLPRegressor(activation="relu", solver=solver, hidden_layer_sizes=(10,))
            nn.fit(X_train, y_train)
            nn.predict(X_test)
            score = nn.score(X_train, y_train)
            #print(score)
            if score > max_value:
                max_value = score
                max_kernel = solver
            performance.append(score)

        plt.bar(y_pos, performance, align='center', alpha=0.5)
        plt.xticks(y_pos, solver_types)
        plt.ylabel('Score')
        plt.xlabel('Solver')
        plt.title('Tuning Solver Hyperparameter')
        plt.show()
        return max_kernel


def main(algorithm):
    X_train, X_test, y_train, y_test = load_data()

    # Check if the data set is trivial - 0.253
    #is_trivial(X_train, y_train)

    if algorithm == "KNR":
        # KNR - 0.379
        knr = KNR()

        # Graphs
        knr.graph_weights(X_train, X_test, y_train, y_test)
        knr.graph_neighbors(X_train, X_test, y_train, y_test)

        # Fitting to best params
        start_time = time.time()
        best_params = knr.tune_hyper_params(X_train, X_test, y_train, y_test)
        print("params:", best_params)
        knr.predict(best_params, X_train, X_test, y_train, y_test)
        print("--- %s seconds ---" % (time.time() - start_time))
    elif algorithm == "KRR":
        # KRR - 0.580
        krr = KernelRidgeRegressor()

        # Graph individual hyper parameters
        kernel = krr.graph_kernel(X_train, X_test, y_train, y_test)
        alpha = krr.graph_alpha(X_train, X_test, y_train, y_test)
        gamma = krr.graph_gamma(X_train, X_test, y_train, y_test)
        print("alpha:", alpha, "\ngamma:", gamma, "\nkernel:", kernel)


        start_time = time.time()
        params = krr.tune_hyper_params(X_train, X_test, y_train, y_test)
        print("params:", params)
        krr.predict(params, X_train, X_test, y_train, y_test)
        print("--- %s seconds ---" % (time.time() - start_time))

    elif algorithm == "NN":
        # NN 0.505
        nn = NeuralNetwork()
        start_time = time.time()
        print(nn.graph_solver(X_train, X_test, y_train, y_test))
        print(nn.graph_activation(X_train, X_test, y_train, y_test))
        print(nn.graph_hidden_layers(X_train, X_test, y_train, y_test))

        params = nn.tune_hyper_params(X_train, X_test, y_train, y_test)
        print("params:", params)
        nn.predict(params, X_train, X_test, y_train, y_test)
        print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    print("--Welcome to the Spotify Song Danceability Predictor--")
    print("To recreate graphs and make danceability predictions, choose any of the following algorithms")
    print("KRR, KNR, NN")
    user_input = input()
    main(user_input)
