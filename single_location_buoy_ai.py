import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

"""
TO DO:

 - graph the real RMSE for the unscaled numbers from the model
 - Remove December 2023 from the dataset since it is missing a lot of values

"""

"""
ABOUT THE AI

The input to the model will be approximately one month(changed to week) of time series data 
One month = 30 days = 30 * 24 = 720 hours
This is around 720 time steps in total

With the 5 parameters in and the month, day, hour, and minute, that will be 9 values in


"""

# LSTM parameters
hidden_size = 25
num_layers = 1

# size of the linear layer after the LSTM layer
linear_size = 100

# general model parameters
learning_rate = 0.0001
batch_size = 40
num_epochs = 50
seq_length = 24 * 30  # length of the input
pred_length = 24  # number of time series rows predicted by the model (24 = 1 day)
dropout = 0.1

# sets device to run computations on gpu
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# load in the csv file of hourly datapoints
df_buoy_dataset = pd.read_csv(
    'C:\\Users\HP\PycharmProjects\deepLearning\\Neural Networks Project\Data Manipulation\csv_dataset_all.csv',
    index_col=0,
    parse_dates=True)

# select all columns as features
X_data = df_buoy_dataset.values

# select first 5 and last 2 columns for prediction
# don't include the date and time in the prediction
target_cols = list(df_buoy_dataset.columns[:5]) + list(df_buoy_dataset.columns[-2:])
y_data = df_buoy_dataset[target_cols].values

# make the train, test, validate sets
# length of validation set: last 2 months
# length of test set: second to last 2 months
# length of train set: the first 20 months

# hours in a month(30 days): 720
n_total = len(X_data)
n_train = 14400  # roughly 20 months
n_test = 1440  # roughly 2 months
n_val = n_total - n_train - n_test

# actually split the data
X_train, y_train = X_data[:n_train], y_data[:n_train]
X_val, y_val = X_data[n_train:n_train + n_val], y_data[n_train:n_train + n_val]
X_test, y_test = X_data[n_train + n_val:], y_data[n_train + n_val:]


# this function is used to scale everything except the cos and sin prescaled values
def scale_and_combine(
        arr,  # input array (X or y)
        scaler,  # standardScaler
        scale_idx,  # indices of columns to scale
        exclude_idx,  # indices of columns to exclude from scaling
):
    arr_scale = arr[:, scale_idx]
    arr_excl = arr[:, exclude_idx]
    arr_scaled = scaler.transform(arr_scale)
    arr_full = np.concatenate([arr_scaled, arr_excl], axis=1)
    reorder = np.argsort(scale_idx + exclude_idx)
    return arr_full[:, reorder]


# lists the names of the columns in the dataframe that are cos and sin scaled
sin_cos_cols = [
    'month_sin', 'month_cos', 'day_sin', 'day_cos',
    'hour_sin', 'hour_cos', 'WDIR_sin', 'WDIR_cos'
]

# grabs out the index of which columns to scale and which columns not to scale
feature_columns = df_buoy_dataset.columns.tolist()
x_scale_idx = [feature_columns.index(col) for col in feature_columns if col not in sin_cos_cols]
x_exclude_idx = [feature_columns.index(col) for col in sin_cos_cols]

# fit the input data scaler to the training dataset
scaler_X = StandardScaler()
scaler_X.fit(X_train[:, x_scale_idx])

# transform each of the splits using the custom scale function to make sure that
# the cos and sin columns are rescaled
X_train_scaled = scale_and_combine(X_train, scaler_X, x_scale_idx, x_exclude_idx)
X_val_scaled = scale_and_combine(X_val, scaler_X, x_scale_idx, x_exclude_idx)
X_test_scaled = scale_and_combine(X_test, scaler_X, x_scale_idx, x_exclude_idx)

# runs through the same process for the X dataset, but the y dataset does have month, day, or hour
# lists the names of the sin and cos columns to not scale
y_sin_cos_cols = ['WDIR_sin', 'WDIR_cos']

# reuses the previous array, but renames for a separate use
target_columns = target_cols

# extracts which indexes are to be scaled and not scaled
y_scale_idx = [target_columns.index(col) for col in target_columns if col not in y_sin_cos_cols]
y_exclude_idx = [target_columns.index(col) for col in y_sin_cos_cols]

# fits the y scaler on the y training data
scaler_y = StandardScaler()
scaler_y.fit(y_train[:, y_scale_idx])

# scales all the splits
y_train_scaled = scale_and_combine(y_train, scaler_y, y_scale_idx, y_exclude_idx)
y_val_scaled = scale_and_combine(y_val, scaler_y, y_scale_idx, y_exclude_idx)
y_test_scaled = scale_and_combine(y_test, scaler_y, y_scale_idx, y_exclude_idx)

# save the scalers locally for later use
joblib.dump(scaler_X, 'scaler_X.pkl')
joblib.dump(scaler_y, 'scaler_y.pkl')

# this function is used within the dataset class for PyTorch
# it creates a sequence of the data according to one of the input parameters at the top of the file
# it creates a sequence of length seq_length for input to the LSTM
# it also creates a sequence of length pred_length as the real values to compare to the model prediction
def create_sequences(X_data, y_data, seq_length, pred_length):
    xs = []
    ys = []
    for i in range(len(X_data) - seq_length - pred_length + 1):
        x = X_data[i:(i + seq_length)]
        y = y_data[(i + seq_length):(i + seq_length + pred_length)]
        xs.append(x)
        ys.append(y)
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


# this is the dataset class required by PyTorch to use a custom dataset
# it requires the user to creates a few class functions if custom data is used
# the __init__ creates one input and output to the model
# the __len__ function returns the length of the dataset
# the __getitem__ returns a specific item at the given index
class TimeSeriesWindowDataset(Dataset):
    def __init__(self, X_data, y_data, seq_length, pred_length=1, flatten_target=True):
        self.X, self.y = create_sequences(X_data, y_data, seq_length, pred_length)
        self.flatten_target = flatten_target

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = torch.from_numpy(self.X[idx])  # [seq_length, num_features]
        y = torch.from_numpy(self.y[idx])  # [pred_length, num_targets]
        if self.flatten_target:
            y = y.reshape(-1)  # [pred_length * num_targets]
        return X, y


# build the three datasets using the dataset class
train_dataset = TimeSeriesWindowDataset(X_train_scaled, y_train_scaled, seq_length, pred_length)
val_dataset = TimeSeriesWindowDataset(X_val_scaled, y_val_scaled, seq_length, pred_length)
test_dataset = TimeSeriesWindowDataset(X_test_scaled, y_test_scaled, seq_length, pred_length)

# cast the tensors to the GPU
X_test_tensor = torch.Tensor(X_test).to(device)
y_test_tensor = torch.Tensor(y_test).to(device)


# this class is the architecture of the model
# the __init__ function typically describes what the layers are (type, activation function, size, etc)
# along with this it may also describe in what order the layers are to make the forward function simpler
# the forward function is run whenever a forward pass through the model needs to be made
# how the layers interact with each other is described in the forward function
class LSTMForecast(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, pred_length, num_targets, dropout=0.2):
        super().__init__()

        # first, the lstm layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # there is dropout after the LSTM layer to help generalize the model
        self.dropout = nn.Dropout(dropout)

        # this is a combination function that runs the layers within in the given order
        self.head = nn.Sequential(

            # this linear function is used to give the model a small amount more of complexity while not adding more to the LSTM
            nn.Linear(hidden_size, linear_size),

            # RELU activation for the linear layer
            nn.ReLU(),

            # dropout of the linear layer
            nn.Dropout(dropout),

            # final output linear layer
            nn.Linear(linear_size, pred_length * num_targets)
        )

    def forward(self, x):

        # pass the input into the LSTM and get the final hidden state from it
        out, _ = self.lstm(x)
        out = out[:, -1, :]

        # dropout on the LSTM
        out = self.dropout(out)

        # runs through the self.head described above
        out = self.head(out)

        # returns output of model
        return out


# get dimensions from data
input_size = X_train_scaled.shape[1]
num_targets = y_train_scaled.shape[1]

# create the model instance
model = LSTMForecast(
    input_size=input_size,
    hidden_size=hidden_size,
    num_layers=num_layers,
    pred_length=pred_length,
    num_targets=num_targets,
    dropout=dropout
).to(device)

# define the loss function
criterion = nn.SmoothL1Loss()

# define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# put the datasets into dataloader for batching
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# training function
def train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=20):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None

    # uses limited epoch number for stopping criteria
    for epoch in range(num_epochs):

        # sets the model into training mode so that it can be changed
        model.train()
        epoch_train_loss = 0

        # runs the training for the epoch over all batches within the dataset
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # sets all gradients to zero
            optimizer.zero_grad()

            # runs a forward pass through the model
            y_pred = model(X_batch)

            # calculates the loss of that pass
            loss = criterion(y_pred, y_batch)

            # does a backward pass of the model, performing backpropagation on all learnable parameters
            loss.backward()

            # updates all parameters with respect to the gradient
            optimizer.step()

            # remembers the training loss for this batch
            epoch_train_loss += loss.item()

        # calculates the average training loss for that epoch
        epoch_train_loss /= len(train_loader)

        # remembers this epoch's training loss
        train_losses.append(epoch_train_loss)

        # validation section
        # puts the model into evaluation mode
        model.eval()
        epoch_val_loss = 0

        # does not calculate gradients when making a forward pass through the model
        with torch.no_grad():

            # for all values within the validation set
            for X_val, y_val in val_loader:

                # send the values to the GPU
                X_val, y_val = X_val.to(device), y_val.to(device)

                # run a forward pass
                val_pred = model(X_val)

                # find the loss of the validation set
                val_loss = criterion(val_pred, y_val)

                # remember the loss
                epoch_val_loss += val_loss.item()

        # average the loss for the validation set
        epoch_val_loss /= len(val_loader)

        # remember this loss as a validation loss
        val_losses.append(epoch_val_loss)

        print(f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {epoch_train_loss:.5f} | Val Loss: {epoch_val_loss:.5f}")

        # save the best model for later use
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_state = model.state_dict()

    # once done with training, load the best model for further use
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return train_losses, val_losses


# testing function
def test(model, test_loader, criterion, device):

    # sets the model to evaluate mode
    model.eval()
    test_loss = 0
    predictions = []
    targets = []

    # does not calculate gradients
    with torch.no_grad():

        # for all of the batches within the given data
        for X_batch, y_batch in test_loader:

            # send those datasets to the GPU
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # run a forward pass
            y_pred = model(X_batch)

            # calculate the total loss
            test_loss += criterion(y_pred, y_batch).item()

            # get the values back to the cpu for use in the rest of the program and appends them to their respective locations
            predictions.append(y_pred.cpu())
            targets.append(y_batch.cpu())

    # calculates the loss
    test_loss /= len(test_loader)

    # concatenates the predictions
    predictions = torch.cat(predictions, dim=0)

    # concatenates the real values
    targets = torch.cat(targets, dim=0)

    print(f"Test Loss: {test_loss:.5f}")
    return test_loss, predictions, targets


# runs the training function
train_losses, val_losses = train(
    model, train_loader, val_loader, criterion, optimizer, device, num_epochs=num_epochs
)

# runs the testing function
test_loss, test_predictions, test_targets = test(model, test_loader, criterion, device)


# this function unscales the y predictions so that they can be viewed in their original units
def unscale_y(y_scaled, scaler, y_scale_idx, y_exclude_idx):
    num_targets = len(y_scale_idx) + len(y_exclude_idx)
    y_scaled = y_scaled.reshape(-1, num_targets)
    y_unscaled = np.zeros_like(y_scaled)
    y_unscaled[:, y_scale_idx] = scaler.inverse_transform(y_scaled[:, y_scale_idx])
    y_unscaled[:, y_exclude_idx] = y_scaled[:, y_exclude_idx]
    return y_unscaled


# get numpy arrays
test_pred_np = test_predictions.cpu().numpy()
test_target_np = test_targets.cpu().numpy()

# unscale the y prediction and y real values
test_pred_unscaled = unscale_y(test_pred_np, scaler_y, y_scale_idx, y_exclude_idx)
test_target_unscaled = unscale_y(test_target_np, scaler_y, y_scale_idx, y_exclude_idx)

# print the RMSE for each parameter
print("Unscaled RMSE per parameter:")
for i, col in enumerate(target_cols):
    rmse = np.sqrt(mean_squared_error(test_target_unscaled[:, i], test_pred_unscaled[:, i]))
    print(f"{col}: {rmse:.4f}")


# function for finding the RMSE of the WDIR which is comprised of cos and sin scaled segments
def angular_rmse(y_true_deg, y_pred_deg):

    diff = np.abs(y_true_deg - y_pred_deg) % 360
    diff = np.where(diff > 180, 360 - diff, diff)
    return np.sqrt(np.mean(diff ** 2))


# get the indices for sin/cos in the target columns
wd_sin_idx = target_cols.index('WDIR_sin')
wd_cos_idx = target_cols.index('WDIR_cos')

# Reconstruct WDIR from sin/cos for both prediction and ground truth
WDIR_pred = (np.degrees(np.arctan2(test_pred_unscaled[:, wd_sin_idx], test_pred_unscaled[:, wd_cos_idx])) % 360)
WDIR_true = (np.degrees(np.arctan2(test_target_unscaled[:, wd_sin_idx], test_target_unscaled[:, wd_cos_idx])) % 360)

# compute and print angular RMSE
WDIR_error = angular_rmse(WDIR_true, WDIR_pred)
print(f"WDIR angular RMSE: {WDIR_error:.2f} degrees")

# THIS SECTION GETS ALL THE DATA FROM BEFORE BUT PRINTS IT IN A WAY THAT ALLOWS EASY COPY/PASTE INTO EXCEL FOR HYPERPARAMETER TUNING

# find the index of the best model according to the validation loss
best_val_epoch = np.argmin(val_losses)

# get the train loss from the same epoch
best_train_loss = train_losses[best_val_epoch]
best_val_loss = val_losses[best_val_epoch]

# find RMSE values as before
rmse_values = [
    np.sqrt(mean_squared_error(test_target_unscaled[:, i], test_pred_unscaled[:, i]))
    for i in range(len(target_cols))
]

all_headers = (
        ["Train_Loss_at_Best_Val", "Best_Val_Loss", "Test_Loss"] +
        list(target_cols) + ["WDIR_angular_RMSE"]
)

all_values = (
        [f"{best_train_loss:.5f}", f"{best_val_loss:.5f}", f"{test_loss:.5f}"] +
        [f"{v:.4f}" for v in rmse_values] + [f"{WDIR_error:.2f}"]
)

# prints out the values for the easy copy/paste (separated by tabs)
print('\t'.join(all_headers))
print('\t'.join(all_values))

# plots the train loss and validation loss on the same graph
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.legend()
plt.show()

# THIS SECTION PLOTS ALL 6 PARAMETERS ON ONE SUBPLOT AT ONE INDEX POINT WITHIN THE TEST DATASET

# plot 6 of the 7 columns
plot_params = [col for col in target_cols if col != 'WDIR_cos']

# finds how many plots are needed
n_params = len(plot_params)
n_cols = 3
n_rows = 2

# sample index within the test dataset
sample_idx = 500

# finds the indices for wind direction sin and cos
wd_sin_idx = target_cols.index('WDIR_sin')
wd_cos_idx = target_cols.index('WDIR_cos')

# plotting setup
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 8), sharex=True)
axes = axes.flatten()

# plots all of the parameters on one subplot
for plot_idx, param_name in enumerate(plot_params):
    ax = axes[plot_idx]
    # runs this to get the wind direction in degrees
    if param_name == 'WDIR_sin':
        sin_true = test_target_unscaled[sample_idx * pred_length: (sample_idx + 1) * pred_length, wd_sin_idx]
        cos_true = test_target_unscaled[sample_idx * pred_length: (sample_idx + 1) * pred_length, wd_cos_idx]
        sin_pred = test_pred_unscaled[sample_idx * pred_length: (sample_idx + 1) * pred_length, wd_sin_idx]
        cos_pred = test_pred_unscaled[sample_idx * pred_length: (sample_idx + 1) * pred_length, wd_cos_idx]

        # convert both true and prediction to degrees
        angle_true = (np.degrees(np.arctan2(sin_true, cos_true))) % 360
        angle_pred = (np.degrees(np.arctan2(sin_pred, cos_pred))) % 360

        # plots the wind direction
        ax.plot(angle_true, label='Actual WDIR (deg)')
        ax.plot(angle_pred, label='Predicted WDIR (deg)')
        ax.set_title('Wind Direction')
        ax.set_ylabel('Degrees')
        ax.set_xlabel('Hour Ahead')
        ax.legend()
    else:
        # runs the same plotting code for the rest of the parameters
        param_idx = target_cols.index(param_name)
        y_true = test_target_unscaled[sample_idx * pred_length: (sample_idx + 1) * pred_length, param_idx]
        y_pred = test_pred_unscaled[sample_idx * pred_length: (sample_idx + 1) * pred_length, param_idx]
        ax.plot(y_true, label='Actual')
        ax.plot(y_pred, label='Predicted')
        ax.set_title(param_name)
        ax.set_xlabel('Hour Ahead')
        if plot_idx % n_cols == 0:
            ax.set_ylabel('Value')
        ax.legend()

# Hide any unused subplots (not needed for 6 graphs in 3x2, but good practice for dynamic code)
#for i in range(n_params, n_rows * n_cols):
   # fig.delaxes(axes[i])

# plots the full subplot
plt.suptitle(f"Predicted vs Actual for all Parameters\nTest Sample {sample_idx}", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()