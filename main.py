# Import necessary libraries
import pandas as pd  # Used for data manipulation and analysis
import numpy as np  # Provides support for arrays and mathematical operations
import matplotlib.pyplot as plt  # Used for data visualization
from sklearn.model_selection import train_test_split  # Splits dataset into training and testing sets
from sklearn.linear_model import LinearRegression  # Implements linear regression model
from sklearn.metrics import mean_squared_error, mean_absolute_error  # Evaluates model performance

# Load the dataset from a CSV file
df = pd.read_csv("Student_Performance.csv")

# Remove rows with missing values
df.dropna(inplace=True)

# Drop the "Extracurricular Activities" column from the dataset
df.drop(["Extracurricular Activities"], axis=1, inplace=True)

# Convert all column names to lowercase for consistency
df.columns = [col.lower() for col in df.columns]

# Define the target variable (performance index) and feature variables (other columns)
y = np.array(df["performance index"])  # Target variable
x = np.array(df.drop("performance index", axis=1))  # Features

# Split the dataset into training (80%) and testing (20%) sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=101)

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(x_train, y_train)

# Make predictions using the test set
prediction = model.predict(x_test)

# Initialize the menu loop
menu = True

# Interactive menu for user options
while menu == True:
    # Prompt user for input
    choice = input("Enter 'v' to visualise data, 'mse' to view mean squared error loss, "
                   "'mae' to view mean absolute error loss, 'p' to make your own prediction, 'q' if you would like to quit:")
    choice = choice.lower()

    # Ensure valid input
    while choice not in ["v", "mse", "mae", "p", "q"]:
        print("Please enter a valid option!")
        choice = input("Enter 'v' to visualise data, 'mse' to view mean squared error loss, "
                       "'mae' to view mean absolute error loss, 'p' to make your own prediction, 'q' if you would like to quit")

    # Quit the menu
    if choice == "q":
        menu = False

    # Display Mean Squared Error (MSE)
    elif choice == "mse":
        print('mean_squared_error : ', mean_squared_error(prediction, y_test))

    # Display Mean Absolute Error (MAE)
    elif choice == "mae":
        print('mean_absolute_error : ', mean_absolute_error(y_test, prediction))

    # Allow user to input custom values for prediction
    elif choice == "p":
        print("Welcome to the prediction menu!\n")
        stats = []
        while True:
            try:
                # Collect input from user and validate it
                studied = int(input("Enter the number of hours you studied:"))
                score = float(input("Please enter your previous test percentage score:"))
                sleep = int(input("Please enter the integer number of hours you slept:"))
                papers = int(input("Please enter the number of practice question papers you have completed:"))
                break
            except ValueError:
                print("Please enter a valid number!")

        # Convert input into NumPy array format
        stats = np.array([[studied, score, sleep, papers]])

        # Make a prediction using the trained model
        print(f"Your performance index on a scale of 10-100 is {model.predict(stats)[0]}")

    # Data visualization menu
    elif choice == "v":
        print("Welcome to the data visualisation menu!\n")
        visualisation = True

        while visualisation == True:
            # Prompt user for visualization type
            v_choice = input("Please enter what you would like: 't' - see trend between pieces of data, "
                             "'a' - see actual vs predicted scores graph, e - exit: ")
            v_choice = v_choice.lower()

            # Ensure valid input
            while v_choice not in ["t", "a", "e"]:
                print("Please enter a valid option!")
                v_choice = input("Please enter what you would like: 't' - see trend between pieces of data, "
                                 "'a' - see actual vs predicted scores graph, e - exit: ")

            # Exit visualization menu
            if v_choice == "e":
                visualisation = False

            # Plot actual vs predicted scores
            elif v_choice == "a":
                plt.figure(figsize=(8, 6))
                plt.scatter(y_test, prediction, color='blue')  # Scatter plot of actual vs predicted values
                plt.xlabel("Actual Scores")
                plt.ylabel("Predicted Scores")
                plt.title("Actual vs Predicted Scores")
                plt.show()

            # Plot trend between two selected variables
            elif v_choice == "t":
                varx = input("Which metric do you want on the x-axis (hours studied, previous scores, sleep hours, "
                             "sample question papers practiced, performance index): ").lower()

                # Validate user input
                while varx not in df.columns:
                    print("Please enter a valid column in the dataset")
                    varx = input("Which metric do you want on the x-axis (hours studied, previous scores, sleep hours, "
                                 "sample question papers practiced, performance index): ").lower()

                varY = input("Which metric do you want on the y-axis (hours studied, previous scores, sleep hours, "
                             "sample question papers practiced, performance index): ").lower()

                # Validate user input
                while varY not in df.columns:
                    print("Please enter a valid column in the dataset")
                    varY = input("Which metric do you want on the y-axis (hours studied, previous scores, sleep hours, "
                                 "sample question papers practiced, performance index): ").lower()

                # Create scatter plot for selected metrics
                plt.figure(figsize=(8, 6))
                plt.scatter(df[varx], df[varY], color='blue')
                plt.xlabel(varx)
                plt.ylabel(varY)
                plt.title("Your custom plot")
                plt.show()
