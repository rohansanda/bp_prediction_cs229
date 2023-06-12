from config import SEED, TRAIN_PARAMS_ANN, TRAIN_PARAMS_CNN, TRAIN_PARAMS_NN_MOD, TRAIN_PARAMS_CNN_MOD, TRAIN_PARAMS_RESNET, DATA_FNAMES, SAVE_OPTIONS, MODEL_TYPE
from trainer import Trainer
from model import NeuralNetwork, cnn_1d, ResNet, cnn_1d_mod, NeuralNetworkMod
import torch
torch.manual_seed(SEED)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Select training params
if MODEL_TYPE == 'NeuralNetwork':
    TRAIN_PARAMS = TRAIN_PARAMS_ANN 
elif MODEL_TYPE == 'NeuralNetworkMod':
    TRAIN_PARAMS = TRAIN_PARAMS_NN_MOD
elif MODEL_TYPE == 'cnn_1d':
    TRAIN_PARAMS = TRAIN_PARAMS_CNN 
elif MODEL_TYPE == 'cnn_1d_mod':
    TRAIN_PARAMS = TRAIN_PARAMS_CNN_MOD
else:
    TRAIN_PARAMS = TRAIN_PARAMS_RESNET

#Hyper parameter Tuning
if TRAIN_PARAMS['tune_hyperparameters']:
    print("Tuning hyperparameters... \n")
    trainer = Trainer(None, TRAIN_PARAMS, DATA_FNAMES, MODEL_TYPE)
    trainer.grid_search(verbose=True, iter_print=1)

#Run the model    
else:
    model = None
    if MODEL_TYPE == 'NeuralNetwork':
        model = NeuralNetwork(200, TRAIN_PARAMS['hidden_sizes'][0], 1, TRAIN_PARAMS['num_layers'][0], TRAIN_PARAMS['activation_functions'][0]).to(device)
    elif MODEL_TYPE == 'NeuralNetworkMod':
        model = NeuralNetworkMod().to(device)
    elif MODEL_TYPE == 'cnn_1d':
        model = cnn_1d().to(device)
    elif MODEL_TYPE == 'cnn_1d_mod':
        model = cnn_1d_mod().to(device)
    else:
        model = ResNet().to(device)
        
    trainer = Trainer(model, TRAIN_PARAMS, DATA_FNAMES, MODEL_TYPE)
    trainer.train(verbose=True, iter_print=1)

    if SAVE_OPTIONS['save']:
        torch.save(model.state_dict(), SAVE_OPTIONS['path'])
        print("Model saved to: ", SAVE_OPTIONS['path'])

