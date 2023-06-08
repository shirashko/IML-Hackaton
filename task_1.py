import pandas as pd
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import numpy as np

pd.options.mode.chained_assignment = None

saved_means = None  # saves the mean value of the features in the train set for preprocess test sets


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
        if seg == len(segments) - 1 or 'N' in segments[seg]:
            days_next = 0
        else:
            days_next = int(segments[seg + 1][:segments[seg + 1].find('D')])
        total_value -= cancellation_segment_to_numeric(segments[seg], staying_time, days_next)
    return total_value


def cancellation_segment_to_numeric(segment, staying_time, days_next):
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
    if 'D' in segment:
        if 'P' in segment:
            P_index = segment.find('P')
            sum_precentage = int(segment[D_index + 1: P_index]) / 100
        else:
            N_index = segment.find('N')
            sum_precentage = int(segment[D_index + 1: N_index]) / staying_time
    else:
        P_index = segment.find('P')
        return int(segment[:P_index]) / 100
    return sum_precentage * days_current


def check_cancellation_policy(code):
    """
    Checks if a given cancellation code matches the defined cancellation policy format.

    Args:
        code (str): The cancellation code to be checked.

    Returns:
        bool: True if the cancellation code matches the defined format, False otherwise.

    """
    code = str(code)
    pattern = r'^(\d+D\d+[PN])(\d+D\d+[PN])*(\d+D\d+[PN])?(_\d+P)?$'
    return re.match(pattern, code) is not None


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
        The date strings in vector1 and vector2 should be in the format "%Y-%m-%d %H:%M:%S".
        The function uses pandas.to_datetime() to convert the date strings to datetime objects,
        and then calculates the absolute difference in days using vectorized operations.
        """
    format_str = "%Y-%m-%d %H:%M:%S"
    datetime1 = pd.to_datetime(vector1, format=format_str)
    datetime2 = pd.to_datetime(vector2, format=format_str)
    difference = (datetime2 - datetime1).dt.days
    return abs(difference)


def preprocess_data(X: pd.DataFrame, y: pd.Series = None):
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
    global saved_means  # to save the mean values of each feature in the train
    data = X

    # inserting new features that quantify durations
    data = new_duration_features(data)

    # process feature
    data = process_cancellation_policy_code(data)

    # delete features which are not informative (for some we extracted new features from first)
    data = remove_features(data)

    # convert from boolean to integer 0, 1
    data = convert_boolean_features_to_int(data)

    dummy_variables = ["original_payment_type", "charge_option", "checkin_day", "checkin_month"]

    # Handling missing values
    if y is not None:  # train
        # Fill missing values for each feature in the test with the mean value in the training set
        for feature in data.drop(dummy_variables, axis=1).columns:
            data[feature] = data[feature].fillna(data[feature].mean())
    else:  # test
        # For every missing value (null), substitute it with the mean value of the feature in the training set
        for feature in data.drop(dummy_variables, axis=1).columns:
            data[feature].fillna(value=saved_means.loc[feature, "means"], inplace=True)

    # handling categorical variables
    data = pd.get_dummies(data, columns=dummy_variables, prefix=dummy_variables, dtype=int)

    if y is not None:  # train
        saved_means = pd.DataFrame(data.mean(), columns=["means"])  # save the means for the test
        y = y.notna().astype(int)
        return data, y
    else:  # test
        data = data.reindex(columns=saved_means.index, fill_value=0)  # Make the test suitable to train data features
        return data


def convert_boolean_features_to_int(data):
    data[["is_user_logged_in", "is_first_booking"]] = data[["is_user_logged_in", "is_first_booking"]].astype(int)
    return data


def extract_day_and_month(series):
    # Convert the series to datetime if not already in datetime format
    series = pd.to_datetime(series)

    # Extract the day of the week and month
    days_of_week = series.dt.day_name()
    months = series.dt.month_name()

    return days_of_week, months


def process_cancellation_policy_code(data):
    # process cancellation_policy_code feature
    codes = data["cancellation_policy_code"]
    # Create a boolean mask for the rows that match the cancellation policy format
    mask = codes.apply(check_cancellation_policy)
    # Apply the cancel_code_to_numeric function to the matching rows
    for idx in codes.loc[mask].index:
        staying_time = int(data.loc[idx, "duration_of_stay"])
        codes.loc[idx] = cancel_code_to_numeric(codes.loc[idx], staying_time)
    # Set the non-matching rows to 0
    codes.loc[~mask] = 0
    data["cancellation_policy_code"] = codes
    return data


def remove_features(data):
    data = data.drop(["h_booking_id", "h_customer_id", "language", "hotel_id", "hotel_brand_code", "hotel_country_code",
                      "origin_country_code", "checkin_date", "checkout_date", "booking_datetime",
                      "hotel_live_date", "guest_nationality_country_name", 'hotel_area_code', 'hotel_chain_code',
                      "original_payment_currency", "customer_nationality", 'hotel_city_code', "original_selling_amount",
                      "no_of_extra_bed"], axis=1)
    return data


def new_duration_features(data):
    data["booking_to_checkin_duration"] = (
        calculate_date_difference(data["checkin_date"], data["booking_datetime"])).astype(int)
    data["duration_of_stay"] = (calculate_date_difference(data["checkin_date"], data["checkout_date"])).astype(int)
    data["time_between_creation_and_purchase"] = (
        calculate_date_difference(data["hotel_live_date"], data["booking_datetime"])).astype(int)
    data["original_payment_method"] = np.where(data["original_payment_method"] == 'UNKNOWN', 1, 0)
    data["accommadation_type_name"] = np.where(data["original_payment_method"].isin(['Hotel', 'Apartment']), 1, 0)
    data["checkin_day"], data["checkin_month"] = extract_day_and_month(data["checkin_date"])
    return data


def create_train_validation_test_sets():
    train, test = train_test_split(df, test_size=0.2)

    train, validation = train_test_split(train, test_size=0.2)

    x_train, y_train = train.loc[:, train.columns != "cancellation_datetime"], train["cancellation_datetime"]

    x_val, y_val = validation.loc[:, validation.columns != "cancellation_datetime"], validation["cancellation_datetime"]

    x_test, y_test = test.loc[:, test.columns != "cancellation_datetime"], test["cancellation_datetime"]

    return x_train, y_train, x_val, y_val, x_test, y_test


def preprocess_data_sets():
    global y_train_processed, y_val_processed
    # Preprocess the data
    x_train_processed, y_train_processed = preprocess_data(x_train, y_train)
    x_val_processed = preprocess_data(x_val)
    y_val_processed = y_val.notna().astype(
        int)  # make response vector with 1 where there was a cancellation, 0 otherwise
    return x_train_processed, y_train_processed, x_val_processed, y_val_processed


if __name__ == '__main__':
    # Read the data
    df = pd.read_csv("agoda_cancellation_train.csv")

    # Preprocess the data & split it into design matrix and response vector for train, validation, test
    x_train, y_train, x_val, y_val, x_test, y_test = create_train_validation_test_sets()
    processed_x_train, processed_y_train, processed_x_val, processed_y_val = preprocess_data_sets()

    # Initialize and train the Random Forest classifier
    clf = RandomForestClassifier()
    clf.fit(processed_x_train, processed_y_train)

    # Make predictions on the validation set
    y_pred = clf.predict(processed_x_val)

    # Create a DataFrame with the booking IDs and predicted cancellations
    predictions = pd.DataFrame({'ID': x_val['h_booking_id'], 'cancellation': y_pred})

    # Calculate F1 score
    f1 = f1_score(processed_y_val, y_pred, average='macro')
    print("F1 Score:", f1)

    # Save the predictions to a CSV file
    predictions.to_csv('agoda_cancellation_prediction.csv', index=False)
