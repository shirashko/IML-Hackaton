from typing import NoReturn

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import re


def cancel_code_to_numeric(cancel_code, staying_time):
    """
    Converts a cancellation code to a numeric value representing the customer-friendliness of the cancellation policy.

    Args:
        cancel_code (str): The cancellation code to be converted.
        staying_time (int): The number of nights the customer has booked to stay.

    Returns:
        float: The numeric value representing the customer-friendliness of the cancellation policy.

    """
    segments = str(cancel_code).split('_')
    total_value = 0
    for seg in range(len(segments)):
        if seg == len(segments) - 1:
            days_next = 0
        else:
            if 'D' in segments[seg + 1]:
                days_next = segments[seg + 1][:segments[seg + 1].find('D')]
            else:
                days_next = 0
        total_value += cancelation_segment_to_numeric(segments[seg], staying_time, days_next)
    return total_value


def cancelation_segment_to_numeric(segment, staying_time, days_next):
    """
    Converts a cancellation policy segment to a numeric value based on the staying time and days before check-in.

    Args:
        segment (str): The cancellation policy segment to be converted.
        staying_time (int): The number of nights the customer has booked to stay.
        days_next (int): The number of days before check-in for the next segment.

    Returns:
        float: The numeric value representing the customer-friendliness of the cancellation policy segment.

    """
    D_index = segment.find('D')
    days_before_checkin = int(segment[:D_index])
    days_current = days_before_checkin - days_next
    if 'P' in segment:
        P_index = segment.find('P')
        sum_precentage = int(segment[D_index + 1: P_index]) / 100
    else:
        N_index = segment.find('N')
        sum_precentage = int(segment[D_index + 1: N_index]) / staying_time
    return 1 / (days_current * sum_precentage)


def check_cancellation_policy(code):
    """
    Checks if a given cancellation code matches the defined cancellation policy format.

    Args:
        code (str): The cancellation code to be checked.

    Returns:
        bool: True if the cancellation code matches the defined format, False otherwise.

    """
    pattern = r'^(\d+D[\d+P\d+N]+)(\d+D[\d+P\d+N]+)*(\d+D\d+P)?(_\d+P)?$'
    return re.match(pattern, str(code)) is not None


def calculate_date_difference(vector1, vector2):
    """
        Calculate the absolute difference in days between two date vectors.

        Parameters
        ----------
        vector1 : pandas.Series or array-like
            First vector of date strings.
        vector2 : pandas.Series or array-like
            Second vector of date strings.

        Returns
        -------
        pandas.Series
            Series containing the absolute difference in days between the dates
            in vector1 and vector2.

        Notes
        -----
        The date strings in vector1 and vector2 should be in the format "%d/%m/%Y %H:%M:%S".
        The function uses pandas.to_datetime() to convert the date strings to datetime objects,
        and then calculates the absolute difference in days using vectorized operations.
        """
    format_str = "%Y-%m-%d %H:%M:%S"
    datetime1 = pd.to_datetime(vector1, format=format_str)
    datetime2 = pd.to_datetime(vector2, format=format_str)
    difference = (datetime2 - datetime1).dt.days
    return abs(difference)

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
        pearson_correlation = np.cov(X[feature], y)[0, 1] / (
                    X[feature].std() * std_y)  # np.cov returns a 2X2 matrix
        # which it's [0,1] entrance is cov(X,Y). Cov(X,Y) = Σ [ (Xi - μx) * (Yi - μy) ] / (n - 1)
        fig = go.Figure(go.Scatter(x=X[feature], y=y, mode="markers", showlegend=False))
        fig.update_layout(xaxis_title=f"{feature}", yaxis_title=f"cancellation_datetime",
                          title=f"correlation between {feature} and y -"
                                f"\nPearson Correlation = {pearson_correlation}")
        fig.show()
        #pio.write_image(fig, rf"{output_path}\{feature}.png", format="png",
        #                engine='orca')  # the default engine doesn't
        # work on my computer, so I loaded a suitable engine: conda install -c plotly plotly-orca


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
    # Replace None values with the mean of the column
    if y is not None:  # we are in train
        # Notice that we don't need to remove samples with none in the response vector, because none means there wasn't
        # a cancellation

        # Concatenate the modified X and y back into a DataFrame
        data = pd.concat([X, y], axis=1)
    else:
        data = X

    dummie_variables = ['hotel_area_code', 'hotel_chain_code', "original_payment_method", "original_payment_type",
                        "original_payment_currency", "customer_nationality", "accommadation_type_name",
                        "guest_nationality_country_name"]

    "hotel_id"
    # todo: hotel_area_code, hotel_city_code, hotl_country_name are "on the same category". reminder: hotel_country_code = name
    # todo: hotel_brand_code, hotel_chain_code are "on the same category"
    # remove hotel_id because there is too much, and we have the hotl_chain_code which is informative for this data

    data = pd.get_dummies(data, columns=dummie_variables, prefix="dummy ")

    # delete features which are not informative
    data = data.drop(["h_booking_id", "h_customer_id", "language", "hotel_id", "hotel_brand_code", "hotel_country_code",
                      "hotel_city_code"], axis=1)

    # create new feature - the number of days between check in date and booking data, and delete the original column of booking date
    data["booking_to_checkin_duration"] = (
        calculate_date_difference(data["checkin_date"], data["booking_datetime"])).astype(int)
    # create new feature - the number of days between check in date and check out data, and delete the original columns
    data["duration_of_stay"] = (calculate_date_difference(data["checkin_date"], data["checkout_date"])).astype(int)
    data["time_between_creation_and_purchase"] = (
        calculate_date_difference(data["hotel_live_date"], data["booking_datetime"])).astype(int)
    data = data.drop(["checkin_date", "checkout_date", "booking_datetime", "hotel_live_date"], axis=1)

    # invert to binary - charge_option. todo: check if the scala is ok
    data["charge_option"] = np.where(data["charge_option"] == "Pay Now", 1,
                                     np.where(data["charge_option"] == "Pay Later", 2, 3))
    data["is_user_logged_in"] = data["is_user_logged_in"].astype(int)
    data["is_first_booking"] = data["is_first_booking"].astype(int)
    # todo: to create a new feature from the origin_country_code, for examples 1 if the origin... == passport blabla
    data = data.drop(["origin_country_code"], axis=1)
    # data = np.where(check_cancellation_policy(data["cancellation_policy_code"]) == True,
    #                 cancel_code_to_numeric(data["cancellation_policy_code"], data["duration_of_stay"]), 0)
    data = data.drop(["cancellation_policy_code"], axis=1)

    # Fill missing values with the mean of each column
    for feature in data.drop(["cancellation_datetime"], axis=1):  # need to do this on data.drop[all features we don't \ can't fill with the mean value]
        data[feature].fillna(data[feature].mean())

    if y is not None:  # we are in train
        return data.drop(columns=["cancellation_datetime"]), data["cancellation_datetime"]
    else:  # we are in test
        data = data.reindex(columns=train.columns, fill_value=0)  # make the dummy columns in test to be
        # suitable to the train data, and also drop columns which is in test and not in train (id, date...)
        return data


if __name__ == '__main__':
    # read the data and split it into design matrix and response vector
    df = pd.read_csv(
        r"C:\Users\97252\OneDrive - Yezreel Valley College\Desktop\שנה ב\IML\האקתון\Hackaton\agoda_cancellation_train.csv")
    train, test = train_test_split(df, test_size=0.2)
    X_train, y_train = train.loc[:, train.columns != "cancellation_datetime"], train["cancellation_datetime"]
    X_test, y_test = test.loc[:, test.columns != "cancellation_datetime"], test["cancellation_datetime"]
    preprocess_data(X_train, y_train)