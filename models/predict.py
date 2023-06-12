""" Rohan Sanda 2023 """

""" Wrote a test script to load a model and perform predictions on test data.
    I have since incorporated this script into the Trainer class so its obselete."""
import torch
import numpy as np
import pickle
from model import NeuralNetworkMod, cnn_1d_mod
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

MODEL_TYPE = "cnn_1d_mod"

def load_model(model_file):
    model = cnn_1d_mod()  # Replace with your neural network model class
    model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
    model.eval()
    return model

def predict(test_data_file, label_file, model):
    # Load test data from pickle file
    with open(test_data_file, 'rb') as file:
        test_data = pickle.load(file)

    # Load labels from pickle file
    with open(label_file, 'rb') as file:
        labels = pickle.load(file)

    # Extract diastolic and systolic blood pressure from labels
    systolic_bp = labels[:, 0]
    diastolic_bp = labels[:, 1]

    # Convert test data to torch.Tensor
    test_data_tensor = torch.from_numpy(test_data).float()

    # Perform predictions
    with torch.no_grad():
        if MODEL_TYPE == "cnn_1d_mod":
            systolic_pred, diastolic_pred = model(test_data_tensor.unsqueeze(1))
            systolic_pred = systolic_pred.squeeze(1).numpy()
            diastolic_pred = diastolic_pred.squeeze(1).numpy()
        else:
            systolic_pred, diastolic_pred = model(test_data_tensor)
            # Convert predictions to numpy arrays
            systolic_pred = systolic_pred.numpy()
            diastolic_pred = diastolic_pred.numpy()

    # Calculate R2 score, MSE, and MAE for diastolic blood pressure
    diastolic_r2 = r2_score(diastolic_bp, diastolic_pred)
    diastolic_mse = mean_squared_error(diastolic_bp, diastolic_pred)
    diastolic_mae = mean_absolute_error(diastolic_bp, diastolic_pred)

    # Calculate R2 score, MSE, and MAE for systolic blood pressure
    systolic_r2 = r2_score(systolic_bp, systolic_pred)
    systolic_mse = mean_squared_error(systolic_bp, systolic_pred)
    systolic_mae = mean_absolute_error(systolic_bp, systolic_pred)

    return diastolic_r2, diastolic_mse, diastolic_mae, systolic_r2, systolic_mse, systolic_mae

# Example usage
test_data_file = '/Users/rohansanda/Desktop/cs229_proj/data/test3_flat_testset_segments.pickle'
label_file = '/Users/rohansanda/Desktop/cs229_proj/data/test3_flat_testset_bps.pickle'
model_file = '/Users/rohansanda/Desktop/cs229_proj/6_7_losses/cnn/test3_flat_conv10.pt'

model = load_model(model_file)
diastolic_r2, diastolic_mse, diastolic_mae, systolic_r2, systolic_mse, systolic_mae = predict(test_data_file, label_file, model)

print("Diastolic R2 Score:", diastolic_r2)
print("Diastolic MSE:", diastolic_mse)
print("Diastolic MAE:", diastolic_mae)
print("Systolic R2 Score:", systolic_r2)
print("Systolic MSE:", systolic_mse)
print("Systolic MAE:", systolic_mae)
