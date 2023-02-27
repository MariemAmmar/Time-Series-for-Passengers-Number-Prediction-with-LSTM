# Time Series Flight Passengers Forecasting Using LSTM Neural Network
This code uses LSTM neural networks to forecast time series data. Specifically, it uses the international airline passengers dataset, which contains the number of international airline passengers each month from January 1949 to December 1960.

## Code Overview

### **The first part involves these following steps:**
* Load the dataset using pandas.
* Normalize the dataset using the MinMaxScaler from sklearn.
* Split the dataset into training and test sets.
* Create a function to convert the dataset into a supervised learning problem.
* Reshape the input data to be in the format expected by the LSTM network.
* Define and compile the LSTM network.
* Train the LSTM network on the training data.
* Make predictions on both the training and test data.
* Invert the predictions to get them back into the original scale.
* Calculate the root mean squared error of the predictions.
* Plot the original dataset along with the predicted values for the training and test sets.

#### **The second part includes the implementation of two techniques time steps technique and window method:**
This implementation of LSTM neural network models for the international airline passengers dataset uses the window method to frame the time series as a supervised learning problem and predicts the next value based on the previous ones where the previous time steps are used to predict the next time step.
The first implementation uses the window method and reshapes the input to be [samples, time steps, features]. The second implementation uses the time step technique and reshapes the input to be [samples, features, time steps]. Both implementations achieve similar results.
The code also includes data preprocessing steps such as normalizing the data using the MinMaxScaler and splitting the data into training and testing sets. 
The code defines a Sequential model with an LSTM layer of 4 neurons and a Dense layer with 1 neuron. The model is compiled using mean squared error as the loss function and the Adam optimization algorithm.
The code trains the model using the training data and makes predictions for both training and testing data. It then calculates the root mean squared error (RMSE) of the predictions. Finally, the code plots the original dataset, training predictions, and testing predictions using Matplotlib.
The root mean squared error is used as a metric to evaluate the performance of the model. The lower the RMSE, the better the model performs. The RMSE is calculated for both the training and testing sets.


## Dependencies
This code requires the following libraries to be installed:

* pandas
* matplotlib
* numpy
* keras
* scikit-learn

## Dataset
The dataset used in this code can be found in the file international-airline-passengers.csv. It contains 144 observations of the number of international airline passengers each month from January 1949 to December 1960.

## Running the Code
To run the code, simply execute the code cells in a Python environment with the necessary dependencies installed. The output will be a plot showing the original dataset along with the predicted values for the training and test sets.
