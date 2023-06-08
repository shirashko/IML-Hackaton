from typing import NoReturn
import pandas as pd
from sklearn.model_selection import train_test_split
import re
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
import matplotlib.pyplot as plt
import numpy as np

saved_data = {}  # saves the names and the mean value of the features in the train set for preprocess test sets


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
    global saved_data  # to save the mean values of each feature in the train
    data = X

    # inserting new features that quantify durations
    data = new_duration_features(data)

    # delete features which are not informative (for some we extracted new features from first)
    data = remove_features(data)
    data = data.drop(["cancellation_policy_code"], axis=1)  # removing temporarily until ori finish

    # convert from boolean to integer 0, 1
    data[["is_user_logged_in", "is_first_booking"]] = data[["is_user_logged_in", "is_first_booking"]].astype(int)

    dummy_variables = ['hotel_area_code', 'hotel_chain_code', "original_payment_method", "original_payment_type",
                       "original_payment_currency", "customer_nationality", "accommadation_type_name",
                       "guest_nationality_country_name", "charge_option"]

    # Handling missing values
    if y is not None:  # train
        # Fill missing values for each feature in the test with the mean value in the training set
        for feature in data.drop(dummy_variables, axis=1):
            data[feature].fillna(data[feature].mean())
    else:  # test
        # For every missing value (null), substitute it with the mean value of the feature in the training set
        for feature in data.drop(dummy_variables, axis=1):
            data[feature].fillna(value=saved_data["means"][feature], inplace=True)

    # handling categorical variables
    data = pd.get_dummies(data, columns=dummy_variables, prefix="dummy ", dtype=int)

    if y is not None:  # train
        saved_data["means"] = data.mean()  # save the means for the test
        return data, y
    else:  # test
        print(data.duplicated())
        data = data.reindex(columns=saved_data["means"].index,
                            fill_value=0)  # Make the test suitable to train data features
        return data


def remove_features(data):
    data = data.drop(["h_booking_id", "h_customer_id", "language", "hotel_id", "hotel_brand_code", "hotel_country_code",
                      "hotel_city_code", "origin_country_code", "checkin_date", "checkout_date", "booking_datetime",
                      "hotel_live_date"], axis=1)
    return data


def new_duration_features(data):
    data["booking_to_checkin_duration"] = (
        calculate_date_difference(data["checkin_date"], data["booking_datetime"])).astype(int)
    data["duration_of_stay"] = (calculate_date_difference(data["checkin_date"], data["checkout_date"])).astype(int)
    data["time_between_creation_and_purchase"] = (
        calculate_date_difference(data["hotel_live_date"], data["booking_datetime"])).astype(int)
    return data


if __name__ == '__main__':
    # read the data and split it into design matrix and response vector
    df = pd.read_csv(
        r"C:\Users\97252\OneDrive - Yezreel Valley College\Desktop\שנה ב\IML\האקתון\Hackaton\agoda_cancellation_train.csv")
    train, test = train_test_split(df, test_size=0.2)
    X_train, y_train = train.loc[:, train.columns != "cancellation_datetime"], train["cancellation_datetime"]
    X_test, y_test = test.loc[:, test.columns != "cancellation_datetime"], test["cancellation_datetime"]

    # Preprocess the data
    X_train_processed, y_train_processed = preprocess_data(X_train, y_train)
    X_test_processed = preprocess_data(X_test, None)

    # Split the preprocessed data into training and validation sets
    X_train_processed, X_val_processed, y_train_processed, y_val_processed = train_test_split(X_train_processed,
                                                                                              y_train_processed,
                                                                                              test_size=0.2)

    # Initialize the classifiers
    classifiers = [
        SVC(),
        LogisticRegression(),
        KNeighborsClassifier(),
        DecisionTreeClassifier(),
        LinearDiscriminantAnalysis(),
        QuadraticDiscriminantAnalysis()
    ]

    # Initialize lists to store performance metrics
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    # Train and evaluate each classifier
    for classifier in classifiers:
        # Train the classifier
        classifier.fit(X_train_processed, y_train_processed)

        # Predict on the validation set
        y_val_pred = classifier.predict(X_val_processed)

        # Evaluate the classifier
        accuracy = metrics.accuracy_score(y_val_processed, y_val_pred)
        precision = metrics.precision_score(y_val_processed, y_val_pred, average='macro')
        recall = metrics.recall_score(y_val_processed, y_val_pred, average='macro')
        f1 = metrics.f1_score(y_val_processed, y_val_pred, average='macro')

        # Store the performance metrics
        accuracy_scores.append(accuracy)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

    # Create a bar plot to visualize the performance metrics
    x = np.arange(len(classifiers))
    width = 0.2

    fig, ax = plt.subplots()
    rects1 = ax.bar(x, accuracy_scores, width, label='Accuracy')
    rects2 = ax.bar(x + width, precision_scores, width, label='Precision')
    rects3 = ax.bar(x + 2 * width, recall_scores, width, label='Recall')
    rects4 = ax.bar(x + 3 * width, f1_scores, width, label='F1 Score')

    ax.set_ylabel('Scores')
    ax.set_title('Performance Metrics of Classifiers')
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels([classifier.__class__.__name__ for classifier in classifiers], rotation=45)
    ax.legend()

    plt.tight_layout()
    plt.show()

    # Choose the best performing classifier and train it on the full training set
    best_classifier = classifiers[
        np.argmax(f1_scores)]  # Replace with the classifier that performs the best based on your evaluation
    best_classifier.fit(X_train_processed, y_train_processed)

    # Predict on the test set
    y_test_pred = best_classifier.predict(X_test_processed)

    # Evaluate the best classifier on the test set
    accuracy = metrics.accuracy_score(y_test, y_test_pred)
    precision = metrics.precision_score(y_test, y_test_pred, average='macro')
    recall = metrics.recall_score(y_test, y_test_pred, average='macro')
    f1_score = metrics.f1_score(y_test, y_test_pred, average='macro')

    # Print the evaluation metrics for the best classifier on the test set
    print("Evaluation Metrics on Test Set")
    print(f"Classifier: {best_classifier.__class__.__name__}")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1_score}")

# Notice that we don't need to remove samples with none in the response vector, because none means there wasn't
# a cancellation

"""    if y is not None:  # we are in train
        data = pd.concat([X, y], axis=1) # Concatenate the modified X and y back into a DataFrame
    else:
        data = X"""
# todo: hotel_area_code, hotel_city_code, hotl_country_name are "on the same category". reminder: hotel_country_code = name
# todo: hotel_brand_code, hotel_chain_code are "on the same category"
# remove hotel_id because there is too much, and we have the hotl_chain_code which is informative for this data
# todo: to create a new feature from the origin_country_code, for examples 1 if the origin... == passport blabla
# todo: in the train, the date columns arn't None, but in the test it could be so it will be good to handle this case

# data = np.where(check_cancellation_policy(data["cancellation_policy_code"]) == True,
#                 cancel_code_to_numeric(data["cancellation_policy_code"], data["duration_of_stay"]), 0)

# todo drop_duplicates in train?
