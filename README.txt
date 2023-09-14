# Project README

## Overview
This README file provides comprehensive information about the project, including its file structure and the purpose of each file. It serves as a guide for understanding and effectively using the project.

## Files in the Project

### USERS.txt
This file contains user information, including usernames. It is used for managing user accounts and permissions within the project.

### Documentation

#### project.pdf
This document provides an in-depth explanation of the solutions for tasks 1 and 2. It includes detailed descriptions, methodologies, and any significant findings or insights discovered during the project.

#### agoda_cancellation_policy.pdf
This document presents the optimal cancellation and pricing policy solutions for task 4.

#### agoda_churn_prediction_model.pdf
This file contains the churn prediction model solutions for task 3. It includes a description of the most relevant features for cancellation prediction.

### Python Scripts

#### main.py
Serving as the main entry point of the program, this Python file orchestrates the execution of different tasks and facilitates the flow of data between various components of the project.

#### task_1.py
This Python file contains the model for predicting the original booking price (task 1). It is self-contained and can be executed independently, utilizing the necessary input data to generate predictions.

#### task_2.py
This Python file implements the model for predicting booking cancellations (task 2). It can be executed independently, accepting the required input data and generating predictions accordingly.

### Data Files

#### agoda_cancellation_prediction.csv
This CSV file contains the predictions generated for task 1, forecasting whether a booking will be canceled or not. It includes the booking IDs and corresponding cancellation predictions (1 for cancellation, 0 for no cancellation).

#### agoda_cost_of_cancellation.csv
This CSV file presents the predictions for task 2, estimating the expected selling amount in the event of a cancellation. It includes the booking IDs and the predicted selling amounts. For non-canceled bookings, the predicted selling amount is denoted as -1.

### Dependencies

#### requirements.txt
This file lists the specific requirements and dependencies of the Python environment necessary to run the project. It ensures that the required packages are installed correctly.

Please refer to the relevant sections for detailed information on each file and its purpose within the project.
