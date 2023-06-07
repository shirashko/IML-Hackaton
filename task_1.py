from turtle import pd
from typing import Optional
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

"""
def preprocess_data(X: pd.DataFrame, y: Optional[pd.Series] = None):
    """
"""    preprocess data
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem
    y : array-like of shape (n_samples, )
        Response vector corresponding given samples
    Returns
    -------
    Post-processed design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """"""
    global saved_data
    if y is not None:  # we are in train
        data = pd.concat([X, y], axis=1)  # for consistency of the X and Y rows, at the end of the func we split it back
        data = data.dropna().drop_duplicates()  # remove missing values & remove identical rows (leave one from each)
        data = data.drop(["id", "date", "lat", "long"], axis=1)  # drop unwanted distracting features

        features_with_positive_values = ["zipcode", "sqft_living", "sqft_lot", "yr_built", "floors", "sqft_living15",
                                         "sqft_lot15", "price"]
        features_with_non_negative_values = ["sqft_basement", "yr_renovated", "sqft_above", "bathrooms", "bedrooms"]

        # cut data which doesn't makes sense.
        for feature in features_with_positive_values:
            data = data[data[feature] > 0]
        for feature in features_with_non_negative_values:
            data = data[data[feature] >= 0]

        # keeping only examples with valid range of values (correct range according to sources I found in the internet)
        data = data[data["view"].isin(range(5)) & data["waterfront"].isin([0, 1]) & data["grade"].isin(range(1, 14)) &
                    data["condition"].isin(range(1, 6))]

    else:  # we are in test
        data = X
        # For every missing value (null), substitute it with the mean value of the feature in the training set.
        # iterate only on features that are in the test data and that we don't drop
        for feature in data.drop(["id", "date", "lat", "long", "zipcode"], axis=1):
            data[feature].fillna(value=saved_data["means"][feature], inplace=True)

    # process needed to be done both in the train set and in the test set

    # creates new binary columns for each unique value in the 'zipcode' column of the data frame. Each of these new
    # columns is given a name that starts with the prefix 'zipcode_feature'. The resulting dataframe will have one
    # binary column for each unique value in the 'zipcode' column, and the value of each binary column will be 1 if
    # the corresponding value in the 'zipcode' column matches the unique value of that binary column, and 0 otherwise.
    data["zipcode"] = data["zipcode"].astype(int)
    data = pd.get_dummies(data, prefix='zipcode_feature', columns=['zipcode'])

    # add informative column about the size of the house compare to other houses in the area
    data["sqft_living_relative_to_neighbors"] = data["sqft_living"] / data["sqft_living15"]
    data["suitable_for_many_people"] = ((data["bedrooms"] >= 4) & (data["bathrooms"] >= 2)).astype(int)

    if y is None:
        data = data.reindex(columns=saved_data["means"].index, fill_value=0)  # make the zipcode columns in test to be
        # suitable to the train data, and also drop columns which is in test and not in train (id, date...)
        return data
    else:
        price = data.price
        data = data.drop("price", axis=1)
        saved_data["means"] = data.mean()
        return data, price"""



def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem
    y : array-like of shape (n_samples, )
        Response vector to evaluate against
    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    # The covariance is a measure of how much the two vectors vary together. we divide by the std to normalize the
    # covariance and make it a unitless measure that is independent of the scales of the two vectors. This
    # normalization ensures that the resulting correlation coefficient will be between -1 and 1, where a value of -1
    # indicates a perfect negative linear relationship, a value of 1 indicates a perfect positive linear relationship,
    # and a value of 0 indicates no linear relationship.
    std_y = np.std(y)
    for feature in X.columns:
        pearson_correlation = np.cov(X[feature], y)[0, 1] / (X[feature].std() * std_y)  # np.cov returns a 2X2 matrix
        # which it's [0,1] entrance is cov(X,Y). Cov(X,Y) = Σ [ (Xi - μx) * (Yi - μy) ] / (n - 1)
        fig = go.Figure(go.Scatter(x=X[feature], y=y, mode="markers", showlegend=False))
        fig.update_layout(xaxis_title=f"{feature}", yaxis_title=f"price", title=f"correlation between {feature} and y -"
                                                                                f"\nPearson Correlation = {pearson_correlation}")
        pio.write_image(fig, rf"{output_path}\{feature}.png", format="png", engine='orca')  # the default engine doesn't
        # work on my computer, so I loaded a suitable engine: conda install -c plotly plotly-orca


if __name__ == '__main__':
    # read the data and split it into design matrix and response vector
    df = pd.read_csv(
        r"C:\Users\97252\OneDrive - Yezreel Valley College\Desktop\שנה ב\IML\האקתון\Hackaton\agoda_cancellation_train.csv")
    # The train in the file is with 58,659 records with 38 features determined upon booking and the unknown variable
    # of interest, the “cancellation_datetime”, which features the date of cancellation when it is such
    train, test = train_test_split(df, test_size=0.2)  # todo: 0.2 is hyperparameter we need to decide
    X_train, y_train = train.loc[:, train.columns != "cancellation_datetime"].values, train["cancellation_datetime"].values
    X_test, y_test = test.loc[:, test.columns != "cancellation_datetime"].values, test["cancellation_datetime"].values

    # Preprocessing of housing prices dataset
    # train_data, train_labels = preprocess_data(train_x, train_y)
    # todo: maybe we should do feature evaluation with respect to response to see which feature are important
    # feature_evaluation(train_data, train_labels)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)

    # first preprocess test_data by removing examples with invalid price (null or not positive), and process the data of
    # the test in a similar way to the way we preprocess the train (add the same features, drop the same features...)
    test_labels = test_labels[test_labels.values > 0]  # also drops all the examples with a price = null
    test_data = test_data.loc[test_labels.index]  # keep consistency between the data and the corresponding labels
    test_data = preprocess_data(test_data)  # without sending the test_labels to indicate it's a test

    lr = LinearRegression()
    loss_mean, loss_std = [], []
    p_values = list(range(10, 101))
    for p in p_values:
        # sample p% of the train set using pandas.DataFrame.sample function
        losses_of_p = []
        for i in range(10):
            x = train_data.sample(frac=p / 100)
            y = train_labels.loc[x.index]  # take only the labels that fit the chosen rows in the train data
            lr.fit(x, y)
            loss = lr.loss(test_data.values, test_labels)
            losses_of_p.append(loss)
        loss_mean.append(np.mean(losses_of_p)), loss_std.append(np.std(losses_of_p))
    loss_mean, loss_std = np.array(loss_mean), np.array(loss_std)
    fig = go.Figure(
        [go.Scatter(x=p_values, y=loss_mean - 2 * loss_std, fill=None, mode="lines", line=dict(color="lightgrey")),
         go.Scatter(x=p_values, y=loss_mean + 2 * loss_std, fill='tonexty', mode="lines", line=dict(color="lightgrey")),
         go.Scatter(x=p_values, y=loss_mean, mode="markers+lines", marker=dict(color="black"))],
        layout=go.Layout(title="mean square error over test data as function of train size",
                         xaxis=dict(title="percentage of train data"),
                         yaxis=dict(title="MSE over test data"), showlegend=False))
    fig.show()

"""
