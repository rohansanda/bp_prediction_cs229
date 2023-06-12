""" Rohan Sanda 2023 """
""" Using the sequitur library to train an LSTM autoencoder on the segments """""
import torch
import pickle
import numpy
from sequitur.models import LINEAR_AE, LSTM_AE
from sequitur import quick_train

import matplotlib.pyplot as plt

# Load segments from pickle file
with open('/Users/rohansanda/Desktop/cs229_proj/data/test3_rs_segments.pickle', 'rb') as f:
    segments = pickle.load(f)

num_samples = 100000

train_segs = segments[0:num_samples, :]
print(train_segs.shape)

train_segs_tensor = [torch.tensor(s).float() for s in train_segs]
train_set = train_segs_tensor

# Train the model
encoder, decoder, encodings, losses = quick_train(LSTM_AE, train_set, encoding_dim=45, h_dims=[64], epochs=20, verbose=True)

# Convert the input to a tensor
input_tensor = torch.tensor(train_segs).float()

# Encode the input
z = encoder(input_tensor)

# Reconstruct the input
x_prime = decoder(z, seq_len=num_samples)

# Convert the reconstructed input to a numpy array
x_numpy = x_prime.detach().numpy()

print(x_numpy.shape)

#Save x_numpy, encoder, and decoder into pickle files
with open('x_numpy.pickle', 'wb') as f:
    pickle.dump(x_numpy, f)

with open('encoder.pickle', 'wb') as f:
    pickle.dump(encoder, f)

with open('decoder.pickle', 'wb') as f:
    pickle.dump(decoder, f)
