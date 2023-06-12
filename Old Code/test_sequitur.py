from sequitur.models import LINEAR_AE, LSTM_AE
import torch
from sequitur import quick_train
import pickle

# Load the data
with open('/Users/rohansanda/Desktop/cs229_proj/data/test3_rs_segments.pickle', 'rb') as f:
    segments = pickle.load(f)

# Only work with 20000 segments
train_segs = segments[0:20000, :]
print(segments.shape)

# add dimesion at the end of every segment
# Turn it into a tensor
train_segs = [s.reshape(200, 1) for s in train_segs]
print(train_segs[0].shape)
train_segs_tensor = [torch.tensor(s).float() for s in train_segs]
train_set = train_segs_tensor

# Train the encoder and decoder
encoder, decoder, _, _ = quick_train(LINEAR_AE, train_set, encoding_dim=100, verbose=True)

# Get the embedding for a single vector
x = train_segs_tensor[1800]
z = encoder(x)
x_prime = decoder(z)
print(f"Original: {x}")
print(f"Embedding: {x_prime}")

# To Do:
# 0. Try out LSTM_AE
# 1. Do train/validation/test split
# 2. Train the model
# 3. Implement quick_train.py for easier hyperparameter tuning
# 4. Hyperparameter tuning
# 5. Save embeddings and run them on the baseline models
