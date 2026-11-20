
from sklearn.base import BaseEstimator, TransformerMixin

SERVICOS = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies']

class DerivatedAtributes(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['serviceQuantity'] = (X[SERVICOS] == 'Yes').sum(axis=1)

        X['fibra_sem_suporte'] = ((X['InternetService'] == 'Fiber optic')
                                  & (X['TechSupport'] == 'No')).astype(int)
        return X
