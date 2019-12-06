# General packages
import pandas, numpy
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

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

        krr = KernelRidge(kernel="laplacian", alpha=0.75, gamma=0.05)
        krr.fit(x_train_d, y_train_d)
        krr.predict(x_test_d)
        score = krr.score(x_train_d, y_train)

        '''knn = KNeighborsRegressor(n_neighbors=20, weights="distance", p=1)

        # Fit the classifier to the data
        knn.fit(x_train_d, y_train_d)'''
        score = krr.score(x_test_d, y_test)

        print("feature:", feature, "score:", score)
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
        krr = KernelRidge(kernel="rbf")
        params = {
            #"kernel": ["rbf", "poly", "laplacian"],
            "alpha": numpy.arange(0, 2.5, 0.1),
            "gamma": numpy.arange(0.01, 0.2, 0.01)
        }
        krr = GridSearchCV(krr, params, n_jobs=-1)
        krr.fit(X_train, y_train)
        print(krr.best_params_)
        return krr.best_params_

    def tune_kernel(self, X_train, X_test, y_train, y_test):
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

    def tune_alpha(self, X_train, X_test, y_train, y_test):
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

    def tune_gamma(self, X_train, X_test, y_train, y_test):
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

    def grid_search(self, X_train, X_test, y_train, y_test):
        krr = GridSearchCV(KernelRidge(kernel="laplacian"),
                           param_grid={"alpha": [1, 0.1, 0.01, 0.0001], "gamma": [0.01, 0.05, 0.1, 0.15, .2, 0.25, 0.3,
                                                                                  0.35, 0.4]})
        krr.fit(X_train, y_train)
        krr.predict(X_test)
        print(krr.score(X_train, y_train))

    # Hyper parameters: kernel
    def predict(self, kernel, alpha, gamma, X_train, X_test, y_train, y_test):
        krr = KernelRidge(kernel=kernel, gamma=gamma, alpha=alpha)
        krr.fit(X_train, y_train)
        krr.predict(X_test)
        print(krr.score(X_test, y_test))
        cross_val_scores = cross_val_score(krr, X_train, y_train, cv=20)
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
        knn = KNeighborsRegressor(n_neighbors=25, weights="distance", p=1)
        params = {
            "n_neighbors": range(1, 25, 1),
            "weights": ["distance", "uniform"],
            "p": [1, 2]
        }
        knn = GridSearchCV(knn, params, n_jobs=-1)
        knn.fit(X_train, y_train)
        print(knn.best_params_)
        return knn.best_params_

    def tune_weights(self, X_train, X_test, y_train, y_test):
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

    def predict(self, params, X_train, X_test, y_train, y_test):
        k_values = []
        scores_array = []

        '''knn = KNeighborsRegressor(n_neighbors= 25, weights="distance", p=1)
        params = {
            "n_neighbors": range(1,25, 1),
            "weights" : ["distance", "uniform"],
            "p": [1,2]
        }
        knn = GridSearchCV(knn, params, n_jobs=-1)
        knn.fit(X_train, y_train)
        print(knn.best_params_)'''

        '''knn = KNeighborsRegressor(n_neighbors=params["n_neighbors"], weights=params["weights"], p=params["p"])
        knn.fit(X_train, y_train)
        scores = cross_val_score(knn, X_test, y_test, cv=10)
        print(scores.mean())
        print(knn.score(X_test, y_test))'''

        for i in range(5, 50):
            k_values.append(i)
            knn = KNeighborsRegressor(n_neighbors=i, weights="distance", p=1)
            # Fit the classifier to the data
            knn.fit(X_train, y_train)
            #print(knn.predict(X_test)[0:prediction_count])
            scores = cross_val_score(knn, X_test, y_test, cv=10)
            scores_array.append(knn.score(X_test, y_test))
            #score = knn.score(X_test, y_test)
            print(scores.mean())
            #scores_array.append(scores.mean())
        plt.title("Performance vs # Neighbors")
        plt.ylabel("Scores")
        plt.xlabel("# Neighbors")
        plt.plot(k_values, scores_array, "bo")
        plt.show()


class NeuralNetwork():
    '''
    Neural Network Class
    '''

    def __init__(self):
        print("Initialized - Neural Network")

    def predict(self, X_train, X_test, y_train, y_test):
        print("\nNN - Predicting danceability")
        mlp = MLPRegressor(hidden_layer_sizes=(68, 34, 17), activation="tanh", solver="adam",
                           max_iter=10000)
        mlp.fit(X_train, y_train)
        print(mlp.score(X_test, y_test))

    def choose_hyperparameters(self, X_train, X_test, y_train, y_test):
        activation_types = ["identity", "logistic", "tanh", "relu"]
        solver_types = ["adam", "sgd"]
        print("Comparing parameters")
        for activation in activation_types:
            for solver in solver_types:
                print("Activation: ", activation, "\nSolver: ", solver)
                mlp = MLPRegressor(hidden_layer_sizes=(100, 50, 25, 10), activation=activation, solver=solver,
                                   max_iter=10000)
                mlp.fit(X_train, y_train)
                print("\n")
                predict_train = mlp.predict(X_train)
                #print(predict_train)
                predict_test = mlp.predict(X_test)
                #print(predict_test)
                print(mlp.score(X_test, y_test))


def main():
    X_train, X_test, y_train, y_test = load_data()

    # Check if the data set is trivial - 0.253
    is_trivial(X_train, y_train)


    # KNR - 0.379
    knr = KNR()
    #knr.tune_weights(X_train, X_test, y_train, y_test)
    #knr.tune_hyper_params(X_train, X_test, y_train, y_test)
    #best_params = knr.tune_hyper_params(X_train, X_test, y_train, y_test)
    #knr.predict(best_params, X_train, X_test, y_train, y_test)


    # KRR - 0.580
    krr = KernelRidgeRegressor()
    krr.tune_hyper_params(X_train, X_test, y_train, y_test)
    #kernel = krr.tune_kernel(X_train, X_test, y_train, y_test)
    #alpha = krr.tune_alpha(X_train, X_test, y_train, y_test)
    #gamma = krr.tune_gamma(X_train, X_test, y_train, y_test)
    #print(alpha, gamma, kernel)
    #krr.predict(kernel, alpha, gamma, X_train, X_test, y_train, y_test)

    #krr.grid_search(X_train, X_test, y_train, y_test)
    #krr.predict(X_train, X_test, y_train, y_test)'''

    '''
    # NN 0.505
    nn = NeuralNetwork()
    nn.predict(X_train, X_test, y_train, y_test)
    #nn.choose_hyperparameters(X_train, X_test, y_train, y_test)'''

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
