""" Converted autoencoder-1.ipynb to a .py file for easier use in other files."""

import numpy as np
import torch
import torch.nn as nn
from torch.nn import MSELoss
import pickle
import matplotlib.pyplot as plt
from sequitur.models import LINEAR_AE, LSTM_AE
from statistics import mean

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

with open('./data/test3_rs_segments.pickle', 'rb') as f:
    segments = pickle.load(f)

with open('./data/test3_rs_bps.pickle', 'rb') as f:
    labels = pickle.load(f)

samples = 10000
X_train = segments[:samples, :]
seg_tensor = torch.tensor(X_train, dtype=torch.float32)  # input for linear autoencoder

encoding_dim = 90

# function for outputing list of hidden layer dimensions for the encoder and decoder
def calculate_h_dims(input_dim, encoding_dim, num_hidden_layers):
    units_per_layer = (input_dim - encoding_dim) // (num_hidden_layers + 1)
    h_dims = [input_dim - units_per_layer]  # first hidden layer
    for i in range(num_hidden_layers - 1):
        h_dims.append(h_dims[-1] - units_per_layer)
    return h_dims

rows, cols = segments.shape
input_dim = cols
num_hidden_layers = 2

h_dims = calculate_h_dims(input_dim, encoding_dim, num_hidden_layers)
h_activ = nn.ReLU()  # activation function for the hidden layers
out_activ = nn.ReLU()  # activation function for the decoder's output layer

linear_model = LINEAR_AE(input_dim, encoding_dim, h_dims, h_activ, out_activ).to(device)

# Autoencoder training algorithm
# Mostly taken from https://github.com/shobrook/sequitur/blob/master/sequitur/quick_train.py

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def instantiate_model(model, train_set, encoding_dim, **kwargs):
    return model(train_set[-1].shape[-1], encoding_dim, **kwargs)


def train_model(model, train_set, verbose, lr, epochs, denoise, clip_value, device=None):
    if device is None:
        device = get_device()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = MSELoss(reduction="sum")
    mean_losses = []
    epoch_lst = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for x in train_set:
            x = x.to(device)
            optimizer.zero_grad()
            x_prime = model(x)  # Forward pass
            loss = criterion(x_prime, x)
            loss.backward()  # Backward pass
            if clip_value is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()
            losses.append(loss.item())
        mean_loss = mean(losses)
        print(f"Epoch: {epoch}, Loss: {mean_loss}")
        epoch_lst.append(epoch)
        mean_losses.append(mean_loss)
    # if len(epochs) == len(losses): 
    #     plt.plot(epoch_lst, mean_losses)
    #     plt.xlabel('Epochs')
    #     plt.ylabel('Losses')
    #     plt.title('Losses for Training over Epochs')
    #     plt.show()
    return mean_losses


def get_encodings(model, train_set, device=None):
    if device is None:
        device = get_device()
    model.eval()
    return [model.encoder(x.to(device)) for x in train_set]


def train_ae(model, train_set, encoding_dim, verbose=True, lr=1e-3, epochs=50, clip_value=1, 
                denoise=False, device=None, **kwargs):
    model = instantiate_model(model, train_set, encoding_dim, **kwargs)
    losses = train_model(model, train_set, verbose, lr, epochs, denoise, clip_value, device)
    encodings = get_encodings(model, train_set, device)
    return model.encoder, model.decoder, encodings, losses


# Get linear encoder, decoder

encoder_lin, decoder_lin, encodings_lin, losses_lin = train_ae(LINEAR_AE, seg_tensor, encoding_dim, h_dims=h_dims,
                                                       h_activ=h_activ, out_activ=out_activ, epochs=1)

# Save losses as a list
with open('losses_lin.pickle', 'wb') as f:
    pickle.dump(losses_lin, f)

# Move the encoder, decoder, and encodings to the CPU before saving
encoder_lin = encoder_lin.to('cpu')
decoder_lin = decoder_lin.to('cpu')
encodings_lin = [encoding.to('cpu') for encoding in encodings_lin]


## I changed this part
# Save encoder, decoder, encodings as NumPy arrays
with open('encoder_lin.pickle', 'wb') as f:
    pickle.dump(encoder_lin.state_dict(), f)

with open('decoder_lin.pickle', 'wb') as f:
    pickle.dump(decoder_lin.state_dict(), f)

with open('encodings_lin.pickle', 'wb') as f:
    pickle.dump([encoding.detach().numpy() for encoding in encodings_lin], f)
