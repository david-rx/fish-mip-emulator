# from pca import pca
from sklearn.decomposition import PCA

class PcaSingleton:
    def __init__(self) -> None:
        self.pca = PCA(n_components=40)
        self.is_fit = False

    def fit(self, data):
        self.is_fit = True
        return self.pca.fit_transform(data,)
    
    def run(self, data):
        if self.is_fit:
            return self.pca.transform(data)
        else:
            return self.fit(data)
