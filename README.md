# student-performance-predictor

This project implements a linear regression model to predict student performance based on various factors such as hours studied, previous test scores, sleep hours, and practice papers completed. It also includes an interactive menu for data visualization and custom predictions.

## Features
- **Data Preprocessing**: Cleans and prepares the dataset.
- **Linear Regression Model**: Trains and evaluates a regression model using scikit-learn.
- **User Interaction**: Provides a menu for users to:
  - Visualize data trends.
  - View model performance metrics (MSE & MAE).
  - Make custom predictions based on user input.

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/Student-Performance-Prediction.git
   cd Student-Performance-Prediction
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Place your dataset (`Student_Performance.csv`) in the project directory.

## Usage
Run the script to start the interactive menu:
```sh
python script.py
```

### Menu Options
- **`v`**: Visualize data (trends & actual vs. predicted scores).
- **`mse`**: Display Mean Squared Error (MSE).
- **`mae`**: Display Mean Absolute Error (MAE).
- **`p`**: Enter custom values to predict a student's performance index.
- **`q`**: Quit the program.

### Data Visualization
- **Trend Analysis**: Select two variables from the dataset to plot a scatter graph.
- **Actual vs. Predicted Scores**: View how well the model's predictions align with actual values.

## Dataset
The dataset should be in CSV format and include the following columns:
- `Hours Studied`
- `Previous Scores`
- `Sleep Hours`
- `Sample Question Papers Practiced`
- `Performance Index`

Ensure that the column names match these exactly or modify the script accordingly.

## Dependencies
- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

## License
This project is open-source and available under the [MIT License](LICENSE).

