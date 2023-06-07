import pandas as pd
from sklearn.model_selection import train_test_split


def preprocess_data(X: pd.DataFrame, y: pd.Series):
    """
    preprocess data (train & test)
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem
    y : Series of shape (n_samples,)
        Response vector corresponding given samples
    Returns
    -------
    Post-processed design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    data = pd.concat([X, y], axis=1)  # for consistency of the X and Y rows, at the end of the func we split it back
"""    data = data.dropna().drop_duplicates()  # remove missing values & remove identical rows (leave one from each)
    print("hi")"""


if __name__ == '__main__':
    # read the data and split it into design matrix and response vector
    df = pd.read_csv(
        r"C:\Users\97252\OneDrive - Yezreel Valley College\Desktop\שנה ב\IML\האקתון\Hackaton\agoda_cancellation_train.csv")
    train, test = train_test_split(df, test_size=0.2)
    X_train, y_train = train.loc[:, train.columns != "cancellation_datetime"], train["cancellation_datetime"]
    X_test, y_test = test.loc[:, test.columns != "cancellation_datetime"], test["cancellation_datetime"]
    preprocess_data(X_train, y_train)
