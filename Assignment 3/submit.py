import numpy as np
import pickle as pkl


# Define your prediction method here
# df is a dataframe containing timestamps, weather data and potentials
def my_predict(df):
    # Separate the data-sets
    x_data = np.array(df.iloc[:, 1:7].values)

    # Load your model file
    model_file = open("./model.pkl", "rb")
    model = pkl.load(model_file)

    # Make two sets of predictions, one for O3 and another for NO2
    y_pred = model.predict(x_data)

    # Return both sets of predictions
    return (y_pred[:, 0], y_pred[:, 1])
