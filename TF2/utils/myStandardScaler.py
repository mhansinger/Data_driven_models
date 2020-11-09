class myStandardScaler():
    def __init__(self):
        self.mean=None
        self.var = None

    def fit_transform(self,data,label=True):
        try:
            assert type(data) is np.ndarray
        except AssertionError:
            print('Only numpy arrays!')

        if label is True:
            self.mean = data.mean()
            self.std = data.std()
        else:
            self.mean = data.mean(axis=1).reshape(-1, 1)
            self.std = data.std(axis=1).reshape(-1, 1)

        transformed = (data - self.mean)/self.std

        return transformed

    def rescale(self,data):
        try:
            assert type(data) is np.ndarray
        except AssertionError:
            print('Only numpy arrays!')

        rescaled = data * self.std + self.mean

        return rescaled