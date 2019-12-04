import pandas
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split


def load_data():
    print("Loading data...")
    df = pandas.read_csv("songs.csv")
    X = df.drop(columns=["danceability", "genre", "artist_name", "track_name", "key", "mode", "time_signature"])
    y = df["danceability"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    return X_train, X_test, y_train, y_test

class KernelRidgeRegression():
    '''
    Kernel Ridge Regression Class
    '''

    def __init__(self):
        print("Create KernelRidgeRegression")


class KNR():
    '''
    K Neighbors Regression Class
    '''

    def __init__(self):
        '''
        Initializes data members.
        '''
        self.clusters = None
        self.print_every = 1000

    def cluster(self, k, data):
        X = [[0], [1], [2], [3]]
        y = [0, 0, 1, 1]

        """ All of the features (minus danceability)
        data["genre"], data["artist_name"], data["track_name"], data["popularity"], data["acousticness"],
        data["duration_ms"], data["energy"], data["instrumentalness"],
        data["key"], data["liveness"], data["loudness"], data["mode"], data["speechiness"], data["tempo"], 
        data["time_signature"], data["valence"]]
        

        y_data = data["danceability"]
        X_data = [data["popularity"], data["acousticness"], data["duration_ms"], data["energy"], data["instrumentalness"],
                data["liveness"], data["loudness"], data["speechiness"], data["tempo"], data["valence"]]
        y_data = data["danceability"]
        neigh = KNeighborsRegressor(n_neighbors=k)
        print("y shape", y_data.shape, "X shape:", X_data[0].shape)
        for i in range(9):
            print(i, len(X_data[i]))
        neigh.fit(X_data, y_data)
        """
        neigh = KNeighborsRegressor(n_neighbors=k)
        neigh.fit(X, y)

        return neigh

    def predict(self, neighbors):
        print("made it here")
        print(neighbors.predict([[1.5]]))



class NeuralNetwork():
    '''
    Neural Network Class
    '''

    def __init__(self):
        print("Create Neural Network")



def main():
    X_train, X_test, y_train, y_test = load_data()

    knn = KNeighborsRegressor(n_neighbors=1000)

    # Fit the classifier to the data
    knn.fit(X_train, y_train)
    print(X_test)
    print(knn.predict(X_test)[0:5])
    print(knn.score(X_test, y_test))

if __name__ == '__main__':
    main()



