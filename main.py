import numpy as np
import pandas as pd


def main():
    """implement a method called main that inputs a relative path to the task1 test set and task2 list of
    dates (inpd.dateformat-YYYY-MM-DD)."""
    # read the test data
    test_data1 = pd.read_csv(r"C:\Users\97252\OneDrive - Yezreel Valley College\Desktop\שנה ב\IML\האקתון\Hackaton\Agoda_Test_1.csv")
    # without the “cancellation_datetime” feature, 38 columns
    print(test_data1.head())
    test_data2 = pd.read_csv(r"C:\Users\97252\OneDrive - Yezreel Valley College\Desktop\שנה ב\IML\האקתון\Hackaton\Agoda_Test_2.csv")
    # without the “cancellation_datetime” feature & "original_selling_amount" feature, 37 columns
    print(test_data2.head())


if __name__ == "__main__":
    main()