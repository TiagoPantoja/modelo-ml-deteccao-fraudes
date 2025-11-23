from sklearn.base import BaseEstimator, TransformerMixin


def binary_function_x_plus_1(x):
    return x + 1


class CustomFeatureTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        _ = X
        _ = y
        return self

    def transform(self, X):
        X_copy = X.copy()

        if 'Amount' in X_copy.columns:
            X_copy['Amount_Plus_One'] = binary_function_x_plus_1(X_copy['Amount'])

        return X_copy