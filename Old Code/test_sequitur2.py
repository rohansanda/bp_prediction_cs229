from sequitur.models import LINEAR_AE, LSTM_AE
import torch
from sequitur import quick_train
import pickle

with open('/Users/rohansanda/Desktop/cs229_proj/data/test3_rs_segments.pickle', 'rb') as f:
    segments = pickle.load(f)

train_segs = segments[0:100, :]
train_segs = [s.reshape(200, 1) for s in train_segs]
print(train_segs[0].shape)
train_segs_tensor = [torch.tensor(s).float() for s in train_segs]

train_set = train_segs_tensor # [torch.randn(200, 1) for _ in range(100)]
encoder, decoder, encodings, losses = quick_train(LSTM_AE, train_set, encoding_dim=100, h_dims=[64], epochs=10, verbose=True)



# model = LSTM_AE(
#   input_dim=200,
#   encoding_dim=7,
#   h_dims=[64],
#   h_activ=None,
#   out_activ=None
# )

# x = train_segs_tensor # torch.randn(10, 200) # Sequence of 10 3D vectors
# print(x.shape)
# z = model.encoder(x) # z.shape = [7]
# x_prime = model.decoder(z, seq_len=10) # x_prime.shape = [10, 3]