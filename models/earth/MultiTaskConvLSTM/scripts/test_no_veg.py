# example_inference
import torch
from MultiTaskConvLSTM import ConvLSTMNetwork
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from utils import (
    mse, mae, nash_sutcliffe_efficiency, r2_score, pearson_correlation,
    spearman_correlation, percentage_error, percentage_bias,
    kendall_tau, spatial_correlation
)
import torch.optim as optim


device = 'cpu'

height = 81
width = 97

set_lookback = 1
set_forecast_horizon = 1

#Define variables for evaluation
batch_size = 16
time_steps_out = set_forecast_horizon
channels = 8

#Variable names
#Variable names
variable_names = ['10 metre U wind component', '10 metre V wind component', '2 metre dewpoint temperature', '2 metre temperature', 'Total column rain water', 'Total precipitation', 'Time-integrated surface latent heat net flux']

# Adjust input_dim and output_channels according to your data specifics
model = ConvLSTMNetwork(
    input_dim=8 * set_lookback, 
    hidden_dims=[8, 32, 64], 
    kernel_size=(3,3), 
    num_layers=3, 
    output_channels=64 * set_forecast_horizon, 
    batch_first=True
).to(device)

# Define separate loss functions
loss_fn = nn.MSELoss()       # For regression output
bce_loss_fn = nn.BCELoss()       # For classification output

optimizer = optim.AdamW(model.parameters(), lr = 0.005) 

checkpoint = torch.load("./MultiTaskConvLSTM_no_veg_variables.pth", map_location = device)
model.load_state_dict(checkpoint['model_state_dict'])

# If you want to move the model to the GPU (optional, depending on your setup)
model.to(device)  # Assuming you have a variable `device` for CUDA or CPU

# Ensure that the model is in evaluation mode if you're using it for inference
model.eval()

print("Model loaded successfully")


threshold = 0.1
precip_index = 10

def evaluate(model, test_loader, reg_loss_fn, class_loss_fn, device, variable_names, height, width):
    """
    Evaluate the model on the test set for both regression and classification tasks.
    """
    model.eval()  # Set the model to evaluation model

    # input_to_true = {'zero_to_non_zero': 0, 'non_zero_to_zero': 0}
    # input_to_pred_REG = {'zero_to_non_zero': 0, 'non_zero_to_zero': 0}
    # input_to_pred_CLASS = {'zero_to_non_zero': 0, 'non_zero_to_zero': 0}

    test_reg_loss = 0.0
    test_class_loss = 0.0
    test_total_loss = 0.0

    y_true_reg = []  # List to store true values for regression
    y_pred_reg = []  # List to store predicted values for regression

    y_pred_reg2 = []

    y_true_class = []  # List to store true values for classification
    y_pred_class = []  # List to store predicted probabilities for classification

    # Disable gradient computation
    with torch.no_grad():
        for X_test, y_test, y_zero_test in tqdm(test_loader, desc="Evaluating on Test Set"):
            # Move the batch to the device
            X_test, y_test, y_zero_test = X_test.to(device), y_test.to(device), y_zero_test.to(device)

            # Reshape inputs and targets
            batch_size, time_steps_in, channels_in, grid_points = X_test.shape
            batch_size, time_steps_out, channels_out, grid_points = y_test.shape
            X_test = X_test.view(batch_size, time_steps_in, channels_in, height, width)
            y_test = y_test.view(batch_size, time_steps_out, channels_out, height, width)
            y_zero_test = y_zero_test.view(batch_size, time_steps_out, channels_out, height, width)

            # Forward pass
            regression_output, classification_output = model(X_test)

            classification_predictions = (classification_output > 0.7).float()

            # Compute regression loss
            reg_loss = reg_loss_fn(regression_output, y_test)

            # Compute classification loss
            class_loss = class_loss_fn(classification_output, y_zero_test)

            # Total loss
            total_loss = reg_loss + class_loss

            regression_output2 = torch.where(classification_predictions == 0, regression_output, classification_predictions)

            # Accumulate losses
            test_reg_loss += reg_loss.item() * X_test.size(0)
            test_class_loss += class_loss.item() * X_test.size(0)
            test_total_loss += total_loss.item() * X_test.size(0)

            # Collect true and predicted values for regression and classification
            y_true_reg.append(y_test.cpu())
            y_pred_reg.append(regression_output.cpu())
            y_pred_reg2.append(regression_output2.cpu())
            y_true_class.append(y_zero_test.cpu())
            y_pred_class.append(classification_output.cpu())

    # Normalize losses by the total dataset size
    test_reg_loss /= len(test_loader)
    test_class_loss /= len(test_loader)
    test_total_loss /= len(test_loader)

    print(f"Test Regression Loss: {test_reg_loss:.16f}")
    print(f"Test Classification Loss: {test_class_loss:.16f}")
    print(f"Test Total Loss: {test_total_loss:.16f}")

    y_true_reg_flat = torch.cat(y_true_reg, dim=0).flatten()  # Keep as PyTorch tensor
    y_pred_reg_flat = torch.cat(y_pred_reg, dim=0).flatten()  # Keep as PyTorch tensor
    y_true_class_flat = torch.cat(y_true_class, dim=0).flatten()  # Keep as PyTorch tensor
    y_pred_class_flat = torch.cat(y_pred_class, dim=0).flatten()  # Keep as PyTorch tensor

    # Compute regression metrics
    regression_metrics = {
        "MSE": mse(y_true_reg_flat, y_pred_reg_flat),
        "MAE": mae(y_true_reg_flat, y_pred_reg_flat),
        "NSE": nash_sutcliffe_efficiency(y_true_reg_flat, y_pred_reg_flat),
        "R2": r2_score(y_true_reg_flat, y_pred_reg_flat),
        "Pearson": pearson_correlation(y_true_reg_flat, y_pred_reg_flat),
        "Spearman": spearman_correlation(y_true_reg_flat, y_pred_reg_flat),
        "NSE": nash_sutcliffe_efficiency(y_true_reg_flat, y_pred_reg_flat),
        "Percentage Error": percentage_error(y_true_reg_flat, y_pred_reg_flat),
        "Percentage Bias": percentage_bias(y_true_reg_flat, y_pred_reg_flat),
        "Kendall Tau": kendall_tau(y_true_reg_flat, y_pred_reg_flat),
        "Spatial Correlation": spatial_correlation(y_true_reg_flat, y_pred_reg_flat)}

    print("\nRegression Metrics:")
    for metric, value in regression_metrics.items():
        print(f"{metric}: {value:.16f}")


    # Compute classification metrics
    classification_metrics = {
        "Accuracy": accuracy_score(y_true_class_flat, (y_pred_class_flat > 0.7)),
        "Precision": precision_score(y_true_class_flat, (y_pred_class_flat > 0.7)),
        "Recall": recall_score(y_true_class_flat, (y_pred_class_flat > 0.7)),
        "F1": f1_score(y_true_class_flat, (y_pred_class_flat > 0.7)),
        "ROC-AUC": roc_auc_score(y_true_class_flat, y_pred_class_flat),
    }

    print("\nClassification Metrics:")
    for metric, value in classification_metrics.items():
        print(f"{metric}: {value:.16f}")

    torch.save({
        'y_true_reg': y_true_reg_flat,
        'y_pred_reg': y_pred_reg_flat,
        'y_true_class': y_true_class_flat,
        'y_pred_class': y_pred_class_flat,
    }, 'results')

    return test_total_loss, regression_metrics, classification_metrics


"""
EXPECTED DATALOADER BATCH FORMAT (normalized_test_data):

Each batch must be a tuple: (X_batch, y_batch, y_zero_batch)

X_batch contains the previous hours variables. y_batch contains the next hour's precipitation.
y_zero_batch contains the next hour's precipitation thresholded as 0 for precipiation <=0.1mm/h and 
1 for precipitation >0.1mm.

Shapes BEFORE reshaping inside `evaluate`:
    X_batch:   (B, T_in, C_in, G)        # G = H*W = 81*97 = 7857
    y_batch:   (B, T_out, C_out, G)
    y_zero_batch: (B, T_out, C_out, G)   # binary 0/1 "zero-precip" targets

    If your preprocessing produces (B,T, C, H, W), reshape to (B, T, C, H*W) before inference.

DTypes:
    X_batch, y_batch:       torch.float32
    y_zero_batch:           torch.float32 (will be used with BCELoss)

Reshaping done in 'evaluate':
    X_test = X_batch.view(B, T_in, C_in, H, W)         -> (B, T_in, C_in, 81, 97)
    y_test = y_batch.view(B, T_out, C_out, H, W)       -> (B, T_out, C_out, 81, 97)
    y_zero_test = y_zero_batch.view(B, T_out, C_out, H, W)

Model input:
    model expects X_test shaped (B, T_in, input_dim, H, W)
    where input_dim == 9 * set_lookback  (with set_lookback=1 -> input_dim=9)

Notes:
  • Make sure G == H*W (i.e., 7857 for 81x97).
  • C_out for precipitation should be 1 (one target channel), and y_zero_batch
    is the 0/1 mask for “zero precipitation” at each pixel & time.
  • y_zero_batch should be probabilities/labels in {0,1} for BCELoss.
"""

normalized_test_data = torch.load("data/normalized_test_data_no_veg_input.pth")


test_total_loss, regression_metrics, classification_metrics = evaluate(
    model=model,
    test_loader=normalized_test_data,
    reg_loss_fn=loss_fn,
    class_loss_fn=bce_loss_fn,
    device=device,
    variable_names=variable_names,
    height=height,
    width=width,
)